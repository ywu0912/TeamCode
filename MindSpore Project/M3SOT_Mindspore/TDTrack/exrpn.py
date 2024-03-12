import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
from functools import partial

from .utils import pytorch_utils as pt_utils

NORM_DICT = {
    "batch_norm": nn.BatchNorm1d,
    "id": nn.Identity,
    "layer_norm": nn.LayerNorm,
}

ACTIVATION_DICT = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leakyrelu": partial(nn.LeakyReLU, negative_slope=0.1),
}

WEIGHT_INIT_DICT = {
    "xavier_uniform": nn.init.xavier_uniform_,
}


class LocalTransformerLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = nn.MultiheadAttention(cfg.feat_dim, cfg.num_heads, cfg.attn_dropout)

        self.pre_norm = NORM_DICT[cfg.norm](cfg.feat_dim)

        self.radius = cfg.radius

        if cfg.pos_emb_cfg:
            if cfg.pos_emb_cfg.type == 'mlp':
                self.pos_emb = (
                    pt_utils.Seq(3)
                    .conv1d(cfg.feat_dim, bn=True)
                    .conv1d(cfg.feat_dim, activation=None)
                )
            elif cfg.pos_emb_cfg.type == 'sin':
                pass
            elif cfg.pos_emb_cfg.type == 'fourier':
                pass
            else:
                raise NotImplementedError(
                    'pos_emb == %s has not been implemented.' % cfg.pos_emb_cfg.type)

        if cfg.mask_emb:
            self.mask_emb = (
                pt_utils.Seq(1)
                .conv1d(cfg.feat_dim, activation=None)
            )

        if cfg.center_emb:
            if cfg.fixed_sigma_n2:
                self.sigma_n2 = cfg.sigma_n2
            else:
                self.sigma_n2 = nn.Parameter(cfg.sigma_n2*ops.ones(1), requires_grad=True)

            self.center_emb = (
                pt_utils.Seq(1)
                .conv1d(cfg.feat_dim, activation=None)
            )

        self.dropout = nn.Dropout(cfg.dropout)

        self.cfg = cfg

    def with_pos_embed(self, tensor, pe):
        return tensor if pe is None else tensor + pe

    def with_mask_embed(self, tensor, me):
        return tensor if me is None else tensor + me

    def with_center_embed(self, tensor, ce):
        return tensor if ce is None else tensor + ce

    def _compute_attn_mask(self, xyz, radius):
        with ops.no_grad():
            dist = ops.cdist(xyz, xyz, p=2)
            mask = (dist >= radius)
        return mask

    def forward(self, input_dict):

        # print(self.sigma_n2)
        # assert False

        feat = input_dict.pop('feat')
        xyz = input_dict.pop('xyz')
        center_xyz = input_dict.pop('center_xyz')
        mask_ref = input_dict.pop('mask_ref')
        attn_mask = self._compute_attn_mask(center_xyz, self.radius)

        if self.cfg.pos_emb_cfg:
            pe = xyz.permute(0, 2, 1).contiguous()
            pe = self.pos_emb(pe).permute(2, 0, 1)
        else:
            pe = None

        if self.cfg.mask_emb:
            me = mask_ref.unsqueeze(1)
            me = self.mask_emb(me).permute(2, 0, 1)
        else:
            me = None

        if self.cfg.center_emb:
            center_dist2 = ops.sum(center_xyz * center_xyz, dim=-1)
            gauss_center_score = ops.exp(- 0.5 * self.sigma_n2 * center_dist2)
            ce = gauss_center_score.unsqueeze(1)
            ce = self.center_emb(ce).permute(2, 0, 1)
        else:
            ce = None

        if self.cfg.mask_emb == 'mask_trfm':
            x = feat.permute(2, 0, 1)
            xx = self.pre_norm(x)
            q = k = self.with_pos_embed(xx, pe)
            v = xx
            xx = self.attn(q, k, v, attn_mask=attn_mask)[0]
            v = me
            mx = self.attn(q, k, v, attn_mask=attn_mask)[0]
            if self.cfg.center_emb:
                v = ce
                cx = self.attn(q, k, v, attn_mask=attn_mask)[0]
            else:
                cx = 0.0
            x = x + self.dropout(xx) + mx + cx
            # x = x + xx + mx + cx
        elif self.cfg.mask_emb == 'mask_gate':
            x = feat.permute(2, 0, 1)
            xx = self.pre_norm(x)
            q = k = self.with_pos_embed(xx, pe)
            v = me
            mx = self.attn(q, k, v, attn_mask=attn_mask)[0]  # N,B,C
            mx = self.mask_norm(xx * ops.sigmoid(mx))
            v = mask_ref.unsqueeze(1).permute(
                2, 0, 1).repeat(1, 1, xx.shape[2]) * xx
            fx = self.attn(q, k, v, attn_mask=attn_mask)[0]  # N,B,C
            fx = self.feat_norm(fx + self.dropout(xx))
            v = ce
            cx = self.attn(q, k, v, attn_mask=attn_mask)[0]  # N,B,C
            cx = self.center_norm(xx * ops.sigmoid(cx))
            x = fx + mx + cx
        else:
            x = feat.permute(2, 0, 1)
            xx = self.pre_norm(x)
            q = k = self.with_pos_embed(xx, pe)
            v = self.with_center_embed(xx, ce)
            xx = self.attn(q, k, v, attn_mask=attn_mask)[0]
            x = x + self.dropout(xx)

        output_dict = dict(
            feat=x.permute(1, 2, 0),
            xyz=xyz,
            center_xyz=center_xyz,
            mask_ref=mask_ref,
        )

        return output_dict


class VoteTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList()
        for layer_cfg in cfg.layers_cfg:
            self.layers.append(LocalTransformerLayer(layer_cfg))
        self.cfg = cfg

    def forward(self, input_dict):

        output_dict = dict()
        for i, layer in enumerate(self.layers):
            input_dict = layer(input_dict)
        output_dict.update(input_dict)
        return output_dict

class EXRPN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.transformer = VoteTransformer(cfg.transformer_cfg)
        self.fc_refine = (
            pt_utils.Seq(cfg.feat_dim + 3)
                .conv1d(cfg.feat_dim, bn=True)
                .conv1d(cfg.feat_dim, bn=True)
                .conv1d(3 + 1 + 1, activation=None)
        )
        self.cfg = cfg

    def forward(self, input_dict):
        search_xyz = input_dict.pop('search_xyz')
        search_feat = input_dict.pop('search_feat')
        search_mask_score = input_dict.pop('search_mask_score')
        search_center_xyz = input_dict.pop('search_center_xyz')
        output_dict = {}
        trfm_input_dict = dict(
            feat=search_feat,
            mask_ref=search_mask_score,
            xyz=search_xyz,
            center_xyz=search_center_xyz
        )

        trfm_output_dict = self.transformer(trfm_input_dict)
        search_feat = trfm_output_dict.pop('feat')
        # assert ops.isnan(search_xyz_feat).int().sum() == 0
        search_xyz_feat = ops.cat((search_xyz.transpose(1, 2).contiguous(), search_feat), dim=1)
        # assert ops.isnan(search_xyz_feat).int().sum() == 0
        refined_bboxes = self.fc_refine(search_xyz_feat)
        refined_bboxes = refined_bboxes.transpose(1, 2).contiguous()
        refined_bboxes = ops.cat(
            (refined_bboxes[:, :, :3] + search_center_xyz,
             refined_bboxes[:, :, 3:]),
            dim=-1
        )
        output_dict.update(refined_bboxes=refined_bboxes)
        return output_dict

class SRPN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fc_move = nn.ModuleList()
        self.fc_direction = nn.ModuleList()
        for feat_dim in cfg.feat_dim:
            self.fc_move.append(
                pt_utils.Seq(feat_dim + 3)
                .conv1d(feat_dim, bn=True)
                .conv1d(feat_dim, bn=True)
                .conv1d(3+1, activation=None)
            )
            self.fc_direction.append(
                pt_utils.Seq(feat_dim + 3)
                .conv1d(feat_dim, bn=True)
                .conv1d(feat_dim, bn=True)
                .conv1d(1+1, activation=None)
            )

    def forward(self, input_dict, idx):
        search_xyz = input_dict.pop('search_xyz')
        search_feat = input_dict.pop('search_feat')
        search_mask_score = input_dict.pop('search_mask_score')
        search_center_xyz = input_dict.pop('search_center_xyz')
        output_dict = {}
        search_xyz_feat = ops.cat((search_xyz.transpose(1, 2).contiguous(), search_feat),dim=1)
        refined_moves = self.fc_move[idx](search_xyz_feat).transpose(1, 2).contiguous()
        refined_directions = self.fc_direction[idx](search_xyz_feat).transpose(1, 2).contiguous()
        refined_bboxes = ops.cat(
            (refined_moves[:, :, :3] + search_center_xyz,
             refined_directions[:, :, 0].unsqueeze(-1),
             (refined_moves[:, :, -1] + refined_directions[:, :, -1]).unsqueeze(-1)), dim=-1
        )
        output_dict.update(refined_bboxes=refined_bboxes)
        return output_dict
