""" 
rpn.py
Created by zenn at 2021/5/8 20:55
"""
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
from .utils import pytorch_utils as pt_utils

from .utils.pointnet2_modules import PointnetSAModule


class RPN(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.num_proposal = cfg.num_proposal
        self.vote_aggregation = PointnetSAModule(
            radius=0.3,
            nsample=16,
            mlp=[1 + cfg.feat_dim, cfg.feat_dim,
                 cfg.feat_dim, cfg.feat_dim],
            use_xyz=True,
            normalize_xyz=cfg.normalize_xyz)

        self.FC_proposal = (
            pt_utils.Seq(cfg.feat_dim)
            .conv1d(cfg.feat_dim, bn=True)
            .conv1d(cfg.feat_dim, bn=True)
            .conv1d(3 + 1 + 1, activation=None))

    def forward(self, input_dict):
        search_xyz = input_dict.pop('search_xyz')
        search_feat = input_dict.pop('search_feat')
        search_mask_score = input_dict.pop('search_mask_score')
        search_center_xyz = input_dict.pop('search_center_xyz')
        output_dict = {}

        # estimation_cla = self.FC_layer_cla(feature).squeeze(1)
        # score = estimation_cla.sigmoid()

        # xyz_feature = ops.cat(
        #     (xyz.transpose(1, 2).contiguous(), feature), dim=1)

        # offset = self.vote_layer(xyz_feature)
        # vote = xyz_feature + offset
        # vote_xyz = vote[:, 0:3, :].transpose(1, 2).contiguous()
        # vote_feature = vote[:, 3:, :]

        vote_feat = ops.cat(
            (search_mask_score.unsqueeze(1), search_feat), dim=1)

        center_xyz, proposal_features = self.vote_aggregation(
            search_center_xyz, vote_feat, self.num_proposal)
        proposal_offsets = self.FC_proposal(proposal_features)
        refined_bboxes = ops.cat(
            (proposal_offsets[:, 0:3, :] + center_xyz.transpose(1,
             2).contiguous(), proposal_offsets[:, 3:5, :]),
            dim=1)

        refined_bboxes = refined_bboxes.transpose(1, 2).contiguous()

        output_dict.update(
            refined_bboxes=refined_bboxes,
            center_xyz=center_xyz
        )

        return output_dict
