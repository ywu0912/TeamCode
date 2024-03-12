import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops

from .backbone import DGCNN
from .utils import pytorch_utils as pt_utils
from .transformer import Transformer, GenTransformer, CrossTransformer, GenTransformerWithUpdate, \
    CrossTransformerWithUpdate
from .exrpn import EXRPN, SRPN

class TDTrack_Multi_Input_Perception_Space(nn.Module):
    def __init__(self, cfg, log):
        super().__init__()
        self.cfg = cfg
        self.log = log
        self.backbone_net = DGCNN(cfg.backbone_cfg)
        self.transformer = Transformer(cfg.transformer_cfg)

        if not cfg.transformer_cfg.layers_cfg[-1].mask_pred:
            self.fc_mask = nn.ModuleList()
            for feat_dim in cfg.transformer_cfg.feat_dim:
                self.fc_mask.append(
                    pt_utils.Seq(feat_dim)
                    .conv1d(feat_dim, bn=True)
                    .conv1d(feat_dim, bn=True)
                    .conv1d(1, activation=None)
                )
        else:
            self.fc_mask = None

        if not cfg.transformer_cfg.layers_cfg[-1].center_pred:
            self.fc_center = nn.ModuleList()
            for feat_dim in cfg.transformer_cfg.feat_dim:
                self.fc_center.append(
                    pt_utils.Seq(3 + feat_dim)
                    .conv1d(feat_dim, bn=True)
                    .conv1d(feat_dim, bn=True)
                    .conv1d(3 + feat_dim, activation=None)
                )
        else:
            self.fc_center = nn.ModuleList()
            for feat_dim in cfg.transformer_cfg.feat_dim:
                self.fc_center.append(
                    pt_utils.Seq(3 + feat_dim)
                    .conv1d(feat_dim, bn=True)
                    .conv1d(feat_dim, bn=True)
                    .conv1d(feat_dim, activation=None)
                )

        if cfg.srpn_cfg:
            self.rpn_type = 'srpn'
            self.srpn = SRPN(cfg.srpn_cfg)
        elif cfg.exrpn_cfg:
            self.rpn_type = 'exrpn'
            self.exrpn = EXRPN(cfg.exrpn_cfg)

    def forward(self, input_dict):
        l_template_pcd = input_dict['l_template_pcd'] # b,size,n,3 > [b,n,3 ... b,n,3]
        l_template_mask_ref = input_dict['l_template_mask_ref']
        l_template_pcd = [l_template_pcd[:, i, ...] for i in range(l_template_pcd.shape[1])]
        l_template_mask_ref = [l_template_mask_ref[:, i, ...] for i in range(l_template_mask_ref.shape[1])]

        search_pcd = input_dict['search_pcd']
        search_mask_ref = input_dict['search_mask_ref']

        output_dict = {}
        l_template_xyz, l_template_feat, l_template_idx = [], [], [] # multi input for multi scale feature
        l_search_xyz, l_search_feat, l_search_idx = [], [], []  # single input for multi scale feature
        lv_template_xyz, lv_template_feat, lv_search_xyz, lv_search_feat, = [], [], [], []  # for test
        search_output_dict = self.backbone_net(search_pcd)
        for i, template_pcd in enumerate(l_template_pcd):
            template_output_dict = self.backbone_net(template_pcd)
            l_template_xyz.append(template_output_dict['l_xyz'][-1])
            l_template_feat.append(template_output_dict['l_feat'][-1])
            l_template_idx.append(template_output_dict['l_idx'][-1])
            l_search_xyz.append(search_output_dict['l_xyz'][-1])
            l_search_feat.append(search_output_dict['l_feat'][-1])
            l_search_idx.append(search_output_dict['l_idx'][-1])

            lv_template_xyz.append(template_output_dict['l_xyz'][0])
            lv_template_feat.append(template_output_dict['l_feat'][0])
            lv_search_xyz.append(search_output_dict['l_xyz'][0])
            lv_search_feat.append(search_output_dict['l_feat'][0])

        l_search_mask_ref = []
        for search_idx in l_search_idx:
            l_search_mask_ref.append(ops.gather(search_mask_ref, 1, search_idx))
        for i, template_idx in enumerate(l_template_idx):
            l_template_mask_ref[i] = ops.gather(l_template_mask_ref[i], 1, template_idx)

        output_dict.update(
            l_search_xyz=l_search_xyz,
            l_template_xyz=l_template_xyz,
            l_search_mask_gt = [],
            l_template_mask_gt = []
        )

        if self.training:
            for search_idx in l_search_idx:
                output_dict['l_search_mask_gt'].append(ops.gather(input_dict['search_mask_gt'], 1, search_idx))
            l_template_mask_gt = input_dict['l_template_mask_gt']
            l_template_mask_gt = [l_template_mask_gt[:, i, ...] for i in range(l_template_mask_gt.shape[1])]
            for i, template_idx in enumerate(l_template_idx):
                output_dict['l_template_mask_gt'].append((ops.gather(l_template_mask_gt[i], 1, template_idx)))

        trfm_input_dict = dict(
            l_search_xyz=l_search_xyz,
            l_search_feat=l_search_feat,
            l_search_mask_ref=l_search_mask_ref,
            l_template_xyz=l_template_xyz,
            l_template_feat=l_template_feat,
            l_template_mask_ref=l_template_mask_ref,
        )

        if self.training:
            trfm_input_dict.update(
                l_search_mask_gt=output_dict['l_search_mask_gt'],
                l_template_mask_gt=output_dict['l_template_mask_gt'],
            )

        trfm_output_dict = self.transformer(trfm_input_dict)

        l_search_feat = trfm_output_dict.pop('l_search_feat')
        l_template_feat = trfm_output_dict.pop('l_template_feat')

        output_dict.update(trfm_output_dict)
        l_center_xyz, l_refined_bboxes = [], []
        for i in range(len(l_search_feat)):
            if not self.cfg.transformer_cfg.layers_cfg[-1].mask_pred:
                search_mask_pred = self.fc_mask[i](l_search_feat[i]).squeeze(1)
                output_dict.update(search_mask_pred_9=search_mask_pred)
                search_mask_score = search_mask_pred.sigmoid()
            else:
                search_mask_score = trfm_output_dict['l_search_mask_ref'][i]

            if not self.cfg.transformer_cfg.layers_cfg[-1].center_pred:
                search_xyz_feat = ops.cat((l_search_xyz[i].transpose(1, 2).contiguous(), l_search_feat[i]), dim=1)
                offset = self.fc_center[i](search_xyz_feat)
                search_center_xyz = l_search_xyz[i] + offset[:, :3, :].transpose(1, 2).contiguous()
                search_feat = l_search_feat[i] + offset[:, 3:, :]
                output_dict.update(search_center_pred_9=search_center_xyz)
            else:
                search_xyz_feat = ops.cat((l_search_xyz[i].transpose(1, 2).contiguous(), l_search_feat[i]),dim=1)
                offset = self.fc_center[i](search_xyz_feat)
                search_feat = l_search_feat[i] + offset
                search_center_xyz = trfm_output_dict['l_search_center_ref'][i]

            l_center_xyz.append(search_center_xyz)
            rpn_input_dict = dict(
                search_xyz=l_search_xyz[i],
                search_mask_score=search_mask_score,
                search_feat=search_feat,
                search_center_xyz=search_center_xyz
            )

            if self.rpn_type == 'exrpn':
                rpn_output_dict = self.exrpn(rpn_input_dict)
            elif self.rpn_type == 'srpn':
                rpn_output_dict = self.srpn(rpn_input_dict)
            else:
                rpn_output_dict = None
            l_refined_bboxes.append(rpn_output_dict.pop('refined_bboxes'))

        output_dict.update(l_center_xyz = l_center_xyz, l_refined_bboxes = l_refined_bboxes)

        output_dict.update(lv_template_xyz = lv_template_xyz, lv_search_xyz = lv_search_xyz,
                           lv_template_feat = lv_template_feat, lv_search_feat = lv_search_feat)
        return output_dict

class TDTrack_Multi_Input_Perception_Space_Gen(nn.Module):
    def __init__(self, cfg, log):
        super().__init__()
        self.cfg = cfg
        self.log = log
        self.backbone_net = DGCNN(cfg.backbone_cfg)
        self.transformer = GenTransformer(cfg.transformer_cfg)

        if not cfg.transformer_cfg.layers_cfg[-1].mask_pred:
            self.fc_mask = (
                pt_utils.Seq(cfg.transformer_cfg.feat_dim)
                .conv1d(cfg.transformer_cfg.feat_dim, bn=True)
                .conv1d(cfg.transformer_cfg.feat_dim, bn=True)
                .conv1d(1, activation=None)
            )
        else:
            self.fc_mask = None

        if not cfg.transformer_cfg.layers_cfg[-1].center_pred:
            self.fc_center = (
                pt_utils.Seq(3 + cfg.transformer_cfg.feat_dim)
                .conv1d(cfg.transformer_cfg.feat_dim, bn=True)
                .conv1d(cfg.transformer_cfg.feat_dim, bn=True)
                .conv1d(3 + cfg.transformer_cfg.feat_dim, activation=None)
            )
        else:
            self.fc_center = (
                pt_utils.Seq(3 + cfg.transformer_cfg.feat_dim)
                .conv1d(cfg.transformer_cfg.feat_dim, bn=True)
                .conv1d(cfg.transformer_cfg.feat_dim, bn=True)
                .conv1d(cfg.transformer_cfg.feat_dim, activation=None)
            )

        if cfg.srpn_cfg:
            self.rpn_type = 'srpn'
            self.srpn = SRPN(cfg.srpn_cfg)
        elif cfg.exrpn_cfg:
            self.rpn_type = 'exrpn'
            self.exrpn = EXRPN(cfg.exrpn_cfg)

    def forward(self, input_dict):
        l_template_pcd = input_dict['l_template_pcd'] # b,size,n,3 > [b,n,3 ... b,n,3]
        l_template_mask_ref = input_dict['l_template_mask_ref']
        l_template_pcd = [l_template_pcd[:, i, ...] for i in range(l_template_pcd.shape[1])]
        l_template_mask_ref = [l_template_mask_ref[:, i, ...] for i in range(l_template_mask_ref.shape[1])]

        search_pcd = input_dict['search_pcd']
        search_mask_ref = input_dict['search_mask_ref']

        output_dict = {}
        l_template_xyz, l_template_feat, l_template_idx = [], [], [] # multi input for multi scale feature
        for i, template_pcd in enumerate(l_template_pcd):
            template_output_dict = self.backbone_net(template_pcd)
            l_template_xyz.append(template_output_dict['l_xyz'][-1])
            l_template_feat.append(template_output_dict['l_feat'][-1])
            l_template_idx.append(template_output_dict['l_idx'][-1])

        search_output_dict = self.backbone_net(search_pcd)
        search_xyz = search_output_dict['l_xyz'][-1]
        search_feat = search_output_dict['l_feat'][-1]
        search_idx = search_output_dict['l_idx'][-1]

        search_mask_ref = ops.gather(search_mask_ref, 1, search_idx)
        for i, template_idx in enumerate(l_template_idx):
            l_template_mask_ref[i] = ops.gather(l_template_mask_ref[i], 1, template_idx)

        output_dict.update(
            search_xyz=search_xyz,
            l_template_xyz=l_template_xyz,
            l_template_mask_gt = []
        )

        if self.training:
            output_dict['search_mask_gt'] = ops.gather(input_dict['search_mask_gt'], 1, search_idx)
            l_template_mask_gt = input_dict['l_template_mask_gt']
            l_template_mask_gt = [l_template_mask_gt[:, i, ...] for i in range(l_template_mask_gt.shape[1])]
            for i, template_idx in enumerate(l_template_idx):
                output_dict['l_template_mask_gt'].append((ops.gather(l_template_mask_gt[i], 1, template_idx)))

        trfm_input_dict = dict(
            search_xyz=search_xyz,
            search_feat=search_feat,
            search_mask_ref=search_mask_ref,
            l_template_xyz=l_template_xyz,
            l_template_feat=l_template_feat,
            l_template_mask_ref=l_template_mask_ref,
        )

        if self.training:
            trfm_input_dict.update(
                search_mask_gt=output_dict['search_mask_gt'],
                l_template_mask_gt=output_dict['l_template_mask_gt'],
            )

        trfm_output_dict = self.transformer(trfm_input_dict)

        search_feat = trfm_output_dict.pop('search_feat')
        template_feat = trfm_output_dict.pop('template_feat')

        output_dict.update(trfm_output_dict)
        if not self.cfg.transformer_cfg.layers_cfg[-1].mask_pred:
            search_mask_pred = self.fc_mask(search_feat).squeeze(1)
            output_dict.update(search_mask_pred_9=search_mask_pred)
            search_mask_score = search_mask_pred.sigmoid()
        else:
            search_mask_score = trfm_output_dict['search_mask_ref']

        if not self.cfg.transformer_cfg.layers_cfg[-1].center_pred:
            search_xyz_feat = ops.cat((search_xyz.transpose(1, 2).contiguous(), search_feat), dim=1)
            offset = self.fc_center(search_xyz_feat)
            search_center_xyz = search_xyz + offset[:, :3, :].transpose(1, 2).contiguous()
            search_feat = search_feat + offset[:, 3:, :]
            output_dict.update(search_center_pred_9=search_center_xyz)
        else:
            search_xyz_feat = ops.cat((search_xyz.transpose(1, 2).contiguous(), search_feat),dim=1)
            offset = self.fc_center(search_xyz_feat)
            search_feat = search_feat + offset
            search_center_xyz = trfm_output_dict['search_center_ref']

        output_dict.update(center_xyz=search_center_xyz)

        rpn_input_dict = dict(
            search_xyz=search_xyz,
            search_mask_score=search_mask_score,
            search_feat=search_feat,
            search_center_xyz=search_center_xyz
        )

        if self.rpn_type == 'exrpn':
            rpn_output_dict = self.exrpn(rpn_input_dict)
        elif self.rpn_type == 'srpn':
            rpn_output_dict = self.srpn(rpn_input_dict)
        else:
            rpn_output_dict = None
        output_dict.update(rpn_output_dict)

        # output_dict.update(s_xyz=search_xyz, t_xyz=template_xyz)
        return output_dict

class TDTrack_Multi_Input_Perception_Space_Cross(nn.Module):
    def __init__(self, cfg, log):
        super().__init__()
        self.cfg = cfg
        self.log = log
        self.backbone_net = DGCNN(cfg.backbone_cfg)
        self.transformer = CrossTransformer(cfg.transformer_cfg)

        if not cfg.transformer_cfg.layers_cfg[-1].mask_pred:
            self.fc_mask = (
                pt_utils.Seq(cfg.transformer_cfg.feat_dim)
                .conv1d(cfg.transformer_cfg.feat_dim, bn=True)
                .conv1d(cfg.transformer_cfg.feat_dim, bn=True)
                .conv1d(1, activation=None)
            )
        else:
            self.fc_mask = None

        if not cfg.transformer_cfg.layers_cfg[-1].center_pred:
            self.fc_center = (
                pt_utils.Seq(3 + cfg.transformer_cfg.feat_dim)
                .conv1d(cfg.transformer_cfg.feat_dim, bn=True)
                .conv1d(cfg.transformer_cfg.feat_dim, bn=True)
                .conv1d(3 + cfg.transformer_cfg.feat_dim, activation=None)
            )
        else:
            self.fc_center = (
                pt_utils.Seq(3 + cfg.transformer_cfg.feat_dim)
                .conv1d(cfg.transformer_cfg.feat_dim, bn=True)
                .conv1d(cfg.transformer_cfg.feat_dim, bn=True)
                .conv1d(cfg.transformer_cfg.feat_dim, activation=None)
            )

        if cfg.srpn_cfg:
            self.rpn_type = 'srpn'
            self.srpn = SRPN(cfg.srpn_cfg)
        elif cfg.exrpn_cfg:
            self.rpn_type = 'exrpn'
            self.exrpn = EXRPN(cfg.exrpn_cfg)

    def forward(self, input_dict):
        l_template_pcd = input_dict['l_template_pcd'] # b,size,n,3 > [b,n,3 ... b,n,3]
        l_template_mask_ref = input_dict['l_template_mask_ref']
        l_template_pcd = [l_template_pcd[:, i, ...] for i in range(l_template_pcd.shape[1])]
        l_template_mask_ref = [l_template_mask_ref[:, i, ...] for i in range(l_template_mask_ref.shape[1])]

        search_pcd = input_dict['search_pcd']
        search_mask_ref = input_dict['search_mask_ref']

        output_dict = {}
        l_template_xyz, l_template_feat, l_template_idx = [], [], [] # multi input for multi scale feature
        for i, template_pcd in enumerate(l_template_pcd):
            template_output_dict = self.backbone_net(template_pcd)
            l_template_xyz.append(template_output_dict['l_xyz'][-1])
            l_template_feat.append(template_output_dict['l_feat'][-1])
            l_template_idx.append(template_output_dict['l_idx'][-1])

        search_output_dict = self.backbone_net(search_pcd)
        search_xyz = search_output_dict['l_xyz'][-1]
        search_feat = search_output_dict['l_feat'][-1]
        search_idx = search_output_dict['l_idx'][-1]

        search_mask_ref = ops.gather(search_mask_ref, 1, search_idx)
        for i, template_idx in enumerate(l_template_idx):
            l_template_mask_ref[i] = ops.gather(l_template_mask_ref[i], 1, template_idx)

        output_dict.update(
            search_xyz=search_xyz,
            l_template_xyz=l_template_xyz,
            l_template_mask_gt = []
        )

        if self.training:
            output_dict['search_mask_gt'] = ops.gather(input_dict['search_mask_gt'], 1, search_idx)
            l_template_mask_gt = input_dict['l_template_mask_gt']
            l_template_mask_gt = [l_template_mask_gt[:, i, ...] for i in range(l_template_mask_gt.shape[1])]
            for i, template_idx in enumerate(l_template_idx):
                output_dict['l_template_mask_gt'].append((ops.gather(l_template_mask_gt[i], 1, template_idx)))

        trfm_input_dict = dict(
            search_xyz=search_xyz,
            search_feat=search_feat,
            search_mask_ref=search_mask_ref,
            l_template_xyz=l_template_xyz,
            l_template_feat=l_template_feat,
            l_template_mask_ref=l_template_mask_ref,
        )

        if self.training:
            trfm_input_dict.update(
                search_mask_gt=output_dict['search_mask_gt'],
                l_template_mask_gt=output_dict['l_template_mask_gt'],
            )

        trfm_output_dict = self.transformer(trfm_input_dict)

        search_feat = trfm_output_dict.pop('search_feat')
        template_feat = trfm_output_dict.pop('template_feat')

        output_dict.update(trfm_output_dict)
        if not self.cfg.transformer_cfg.layers_cfg[-1].mask_pred:
            search_mask_pred = self.fc_mask(search_feat).squeeze(1)
            output_dict.update(search_mask_pred_9=search_mask_pred)
            search_mask_score = search_mask_pred.sigmoid()
        else:
            search_mask_score = trfm_output_dict['search_mask_ref']

        if not self.cfg.transformer_cfg.layers_cfg[-1].center_pred:
            search_xyz_feat = ops.cat((search_xyz.transpose(1, 2).contiguous(), search_feat), dim=1)
            offset = self.fc_center(search_xyz_feat)
            search_center_xyz = search_xyz + offset[:, :3, :].transpose(1, 2).contiguous()
            search_feat = search_feat + offset[:, 3:, :]
            output_dict.update(search_center_pred_9=search_center_xyz)
        else:
            search_xyz_feat = ops.cat((search_xyz.transpose(1, 2).contiguous(), search_feat),dim=1)
            offset = self.fc_center(search_xyz_feat)
            search_feat = search_feat + offset
            search_center_xyz = trfm_output_dict['search_center_ref']

        output_dict.update(center_xyz=search_center_xyz)

        rpn_input_dict = dict(
            search_xyz=search_xyz,
            search_mask_score=search_mask_score,
            search_feat=search_feat,
            search_center_xyz=search_center_xyz
        )

        if self.rpn_type == 'exrpn':
            rpn_output_dict = self.exrpn(rpn_input_dict)
        elif self.rpn_type == 'srpn':
            rpn_output_dict = self.srpn(rpn_input_dict)
        else:
            rpn_output_dict = None
        output_dict.update(rpn_output_dict)

        # output_dict.update(s_xyz=search_xyz, t_xyz=template_xyz)
        return output_dict

