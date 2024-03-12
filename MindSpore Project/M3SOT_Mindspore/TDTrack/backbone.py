import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
import pytorch3d


from .utils import pytorch_utils as pt_utils

def furthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = ops.zeros(B, npoint, dtype=ops.long).to(device)
    distance = ops.ones(B, N).to(device) * 1e10
    farthest = ops.randint(0, N, (B,), dtype=ops.long).to(device)
    batch_indices = ops.arange(B, dtype=ops.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = ops.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = ops.max(distance, -1)[1]
    return centroids

def gather(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    # device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = ops.arange(B, dtype=ops.long).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class EdgeConv(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mlps = cfg.mlps
        mlps[0] = mlps[0]*2
        if cfg.use_xyz:
            mlps[0] += 6
        self.shared_mlp = pt_utils.SharedMLP(mlps, bn=True)
        self.cfg = cfg

    def get_graph_feature(self, new_xyz, new_feat, xyz, feat, k, use_xyz=False):
        bs = xyz.size(0)
        device = ops.device('cuda')
        feat = feat if feat is not None else None
        new_feat = new_feat if new_feat is not None else None
        if use_xyz:
            feat = ops.cat([feat, xyz], dim=-1) if feat is not None else xyz
            new_feat = ops.cat([new_feat, new_xyz], dim=-1) if new_feat is not None else new_xyz
        # b,n,c
        _, knn_idx, _ = pytorch3d.ops.knn_points(new_xyz, xyz, K=k, return_nn=True)

        knn_feat = pytorch3d.ops.knn_gather(feat, knn_idx)
        # b,n1,k,c
        feat_tiled = new_feat.unsqueeze(-2).repeat(1, 1, k, 1)
        edge_feat = ops.cat([knn_feat-feat_tiled, feat_tiled], dim=-1)

        return edge_feat.permute(0, 3, 1, 2).contiguous()

    def forward(self, xyz, feat, npoints):
        """
        Args:
            xyz : b,n,3
            feat : b,n,c
        """
        device = xyz.device
        if self.cfg.sample_method == 'FPS':
            sample_idx = furthest_point_sample(xyz, npoints)
        elif self.cfg.sample_method == 'Random':
            sample_idx = ops.randint(0, xyz.size(1), [xyz.size(0), npoints]).int().to(device)
        elif self.cfg.sample_method == 'Range':
            sample_idx = ops.arange(npoints).repeat(xyz.size(0), 1).int().to(device)
        else:
            raise NotImplementedError('Sample method %s has not been implemented' % self.cfg.sample_method)
        sample_idx = sample_idx.long()
        new_xyz = gather(xyz, sample_idx)
        new_feat = gather(feat, sample_idx) if feat is not None else None
        edge_feat = self.get_graph_feature(new_xyz, new_feat, xyz, feat, self.cfg.nsample, use_xyz=self.cfg.use_xyz)
        # b, 2*c(+6), npoints, nsample
        new_feat = self.shared_mlp(edge_feat)
        new_feat = new_feat.max(dim=-1, keepdim=False)[0]
        new_feat = new_feat.permute(0, 2, 1).contiguous()
        return new_xyz, new_feat, sample_idx

class DGCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.SA_modules = nn.ModuleList()
        for layer_cfg in cfg.layers_cfg:
            self.SA_modules.append(EdgeConv(layer_cfg))

        self.downsample_ratios = cfg.downsample_ratios
        self.cfg = cfg

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pcd):
        npts = pcd.shape[1]

        xyz, features = self._break_up_pc(pcd)

        l_xyz, l_features, l_idxs = [xyz], [features], []
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, li_idxs = self.SA_modules[i](l_xyz[i], l_features[i], npts // self.downsample_ratios[i])
            # print('li_xyz=>', li_xyz.shape, 'li_features=>', li_features.shape, 'li_idxs=>', li_idxs.shape)
            # li_xyz=> ops.Size([64, 512, 3]) li_features=> ops.Size([64, 512, 64]) li_idxs=> ops.Size([64, 512])
            # li_xyz=> ops.Size([64, 256, 3]) li_features=> ops.Size([64, 256, 128]) li_idxs=> ops.Size([64, 256])
            # li_xyz=> ops.Size([64, 128, 3]) li_features=> ops.Size([64, 128, 256]) li_idxs=> ops.Size([64, 128])
            l_xyz.append(li_xyz) # b,n,3
            l_features.append(li_features) # b,c,n
            l_idxs.append(li_idxs) # b,n

        return dict(
            l_xyz=l_xyz[1:],
            l_feat=[fea.permute(0, 2, 1) for fea in l_features[1:]],
            l_idx=l_idxs
        )
