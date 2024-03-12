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


class SelfLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            cfg.feat_dim, cfg.num_heads, cfg.attn_dropout)
        self.pre_norm = NORM_DICT[cfg.norm](cfg.feat_dim)

        if cfg.ffn_cfg:
            self.ffn = nn.Sequential(
                nn.Linear(cfg.feat_dim, cfg.ffn_cfg.hidden_dim,
                          bias=cfg.ffn_cfg.use_bias),
                ACTIVATION_DICT[cfg.ffn_cfg.activation](),
                nn.Dropout(cfg.ffn_cfg.dropout, inplace=True),
                nn.Linear(cfg.ffn_cfg.hidden_dim, cfg.feat_dim,
                          bias=cfg.ffn_cfg.use_bias)

            )
            self.pre_ffn_norm = NORM_DICT[cfg.ffn_cfg.norm](cfg.feat_dim)
            self.ffn_dropout = nn.Dropout(cfg.ffn_cfg.dropout, inplace=True)
        if cfg.pos_emb_cfg:
            if cfg.pos_emb_cfg.type == 'mlp':
                self.s_pos_emb = (
                    pt_utils.Seq(3)
                    .conv1d(cfg.feat_dim, bn=True)
                    .conv1d(cfg.feat_dim, activation=None)
                )
                self.t_pos_emb = (
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

        if cfg.mask_emb == 'mask_vanilla':
            self.mask_emb = (
                pt_utils.Seq(1)
                .conv1d(cfg.feat_dim, activation=None)
            )
        elif cfg.mask_emb == 'mask_trfm':
            # self.mask_norm = NORM_DICT[cfg.norm](cfg.feat_dim)
            # self.feat_norm = NORM_DICT[cfg.norm](cfg.feat_dim)
            self.mask_emb = (
                pt_utils.Seq(1)
                .conv1d(cfg.feat_dim, activation=None)
            )
        elif cfg.mask_emb == 'mask_gate':
            self.mask_norm = NORM_DICT[cfg.norm](cfg.feat_dim)
            self.feat_norm = NORM_DICT[cfg.norm](cfg.feat_dim)

        if cfg.mask_pred:
            self.fc_mask = (
                pt_utils.Seq(cfg.feat_dim)
                .conv1d(cfg.feat_dim, bn=True)
                .conv1d(cfg.feat_dim, bn=True)
                .conv1d(1, activation=None)
            )

        if cfg.center_pred:
            self.fc_center = (
                pt_utils.Seq(cfg.feat_dim+3)
                .conv1d(cfg.feat_dim, bn=True)
                .conv1d(cfg.feat_dim, bn=True)
                .conv1d(3, activation=None)
            )
        self.dropout = nn.Dropout(cfg.dropout)

        self.cfg = cfg

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def with_mask_embed(self, tensor, mask):
        return tensor if mask is None else tensor + mask

    def forward(self, input_dict):

        search_feat = input_dict.pop('search_feat')
        template_feat = input_dict.pop('template_feat')
        search_xyz = input_dict.pop('search_xyz')
        template_xyz = input_dict.pop('template_xyz')
        template_mask_ref = input_dict.pop('template_mask_ref')
        search_mask_ref = input_dict.pop('search_mask_ref')
        search_npts = search_feat.shape[-1]
        template_npts = template_feat.shape[-1]
        bs = search_feat.shape[0]
        device = search_feat.device

        feat = ops.cat((search_feat, template_feat), dim=-1)
        mask_ref = ops.cat((search_mask_ref, template_mask_ref), dim=-1)
        xyz = ops.cat((search_xyz, template_xyz), dim=1)
        if self.cfg.pos_emb_cfg:
            if self.cfg.pos_emb_cfg.type == 'mlp':
                s_pe = search_xyz.permute(0, 2, 1).contiguous()
                s_pe = self.s_pos_emb(s_pe).permute(2, 0, 1)
                t_pe = template_xyz.permute(0, 2, 1).contiguous()
                t_pe = self.t_pos_emb(t_pe).permute(2, 0, 1)
                pe = ops.cat((s_pe, t_pe), dim=0)
        else:
            pe = None

        if self.cfg.mask_emb == 'mask_trfm' or self.cfg.mask_emb == 'mask_vanilla':
            me = mask_ref.unsqueeze(1)
            me = self.mask_emb(me).permute(2, 0, 1)
        elif self.cfg.mask_emb == 'mask_gate':
            me = mask_ref.unsqueeze(1).repeat(1, self.cfg.feat_dim, 1)
            me = me.permute(2, 0, 1)

        if self.cfg.mask_emb == 'mask_trfm':
            x = feat.permute(2, 0, 1)
            xx = self.pre_norm(x)
            q = k = self.with_pos_embed(xx, pe)
            v = xx
            px, attn_w = self.attn(q, k, v, attn_mask=None)
            v = me
            mx = self.attn(q, k, v, attn_mask=None)[0]
            x = x + self.dropout(px) + mx
        elif self.cfg.mask_emb == 'mask_vanilla':
            x = feat.permute(2, 0, 1)
            xx = self.pre_norm(x)
            q = k = self.with_pos_embed(xx, pe)
            v = xx + me
            xx, attn_w = self.attn(q, k, v, attn_mask=None)
            x = x + self.dropout(xx)

        if self.cfg.ffn_cfg:
            xx = self.pre_ffn_norm(x)
            xx = self.ffn(xx)
            x = x + self.ffn_dropout(xx)

        feat = x.permute(1, 2, 0)

        search_feat, template_feat = ops.split(feat, (search_npts, template_npts), dim=-1)

        attn_w, _ = ops.split(attn_w, (search_npts, template_npts), dim=1)
        search_attn_w, template_attn_w = ops.split(
            attn_w, (search_npts, template_npts), dim=-1)

        output_dict = dict(
            search_feat=search_feat,
            template_feat=template_feat,
            search_xyz=search_xyz,
            template_xyz=template_xyz,
            search_mask_ref=search_mask_ref,
            template_mask_ref=template_mask_ref
        )

        output_dict.update(input_dict)
        # output_dict.update(search_attn_w=search_attn_w, template_attn_w=template_attn_w)

        if self.cfg.mask_pred:
            aux_x = self.fc_mask(feat)
            mask_pred = aux_x.squeeze(1)
            search_mask_pred, template_mask_pred = ops.split(
                mask_pred, (search_npts, template_npts), dim=1)

            output_dict.update(
                search_mask_ref=search_mask_pred.sigmoid(),
                template_mask_ref=template_mask_pred.sigmoid(),
                search_mask_pred=search_mask_pred,
                template_mask_pred=template_mask_pred,
            )

        if self.cfg.center_pred and (self.training or self.cfg.is_last):
            aux_x = self.fc_center(ops.cat([feat, xyz.permute(0, 2, 1)], dim=1))
            center_pred = aux_x.permute(0, 2, 1) + xyz
            search_center_pred, template_center_pred = ops.split(
                center_pred, (search_npts, template_npts), dim=1)

            output_dict.update(
                search_center_ref=search_center_pred,
                template_center_ref=template_center_pred,
                search_center_pred=search_center_pred,
                template_center_pred=template_center_pred
            )
        return output_dict

class CrossLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            cfg.feat_dim, cfg.num_heads, cfg.attn_dropout)
        self.pre_norm = NORM_DICT[cfg.norm](cfg.feat_dim)

        if cfg.ffn_cfg:
            self.ffn = nn.Sequential(
                nn.Linear(cfg.feat_dim, cfg.ffn_cfg.hidden_dim, bias=cfg.ffn_cfg.use_bias),
                ACTIVATION_DICT[cfg.ffn_cfg.activation](),
                nn.Dropout(cfg.ffn_cfg.dropout, inplace=True),
                nn.Linear(cfg.ffn_cfg.hidden_dim, cfg.feat_dim, bias=cfg.ffn_cfg.use_bias)
            )
            self.pre_ffn_norm = NORM_DICT[cfg.ffn_cfg.norm](cfg.feat_dim)
            self.ffn_dropout = nn.Dropout(cfg.ffn_cfg.dropout, inplace=True)
        if cfg.pos_emb_cfg:
            if cfg.pos_emb_cfg.type == 'mlp':
                self.s_pos_emb = (
                    pt_utils.Seq(3)
                    .conv1d(cfg.feat_dim, bn=True)
                    .conv1d(cfg.feat_dim, activation=None)
                )
                self.t_pos_emb = (
                    pt_utils.Seq(3)
                    .conv1d(cfg.feat_dim, bn=True)
                    .conv1d(cfg.feat_dim, activation=None)
                )
            else:
                raise NotImplementedError(
                    'pos_emb == %s has not been implemented.' % cfg.pos_emb_cfg.type)

        if cfg.mask_emb == 'mask_trfm':
            self.mask_emb = (
                pt_utils.Seq(1)
                .conv1d(cfg.feat_dim, activation=None)
            )
        elif cfg.mask_emb == 'mask_gate':
            self.mask_norm = NORM_DICT[cfg.norm](cfg.feat_dim)
            self.feat_norm = NORM_DICT[cfg.norm](cfg.feat_dim)

        if cfg.mask_pred:
            self.fc_mask = (
                pt_utils.Seq(cfg.feat_dim)
                .conv1d(cfg.feat_dim, bn=True)
                .conv1d(cfg.feat_dim, bn=True)
                .conv1d(1, activation=None)
            )

        if cfg.center_pred:
            self.fc_center = (
                pt_utils.Seq(cfg.feat_dim+3)
                .conv1d(cfg.feat_dim, bn=True)
                .conv1d(cfg.feat_dim, bn=True)
                .conv1d(3, activation=None)
            )
        self.dropout = nn.Dropout(cfg.dropout)

        self.cfg = cfg

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def with_mask_embed(self, tensor, mask):
        return tensor if mask is None else tensor + mask

    def forward(self, input_dict):
        search_feat = input_dict.pop('search_feat')
        template_feat = input_dict.pop('template_feat')
        search_xyz = input_dict.pop('search_xyz')
        template_xyz = input_dict.pop('template_xyz')
        template_mask_ref = input_dict.pop('template_mask_ref')
        search_mask_ref = input_dict.pop('search_mask_ref')

        if self.cfg.pos_emb_cfg:
            if self.cfg.pos_emb_cfg.type == 'mlp':
                s_pe = search_xyz.permute(0, 2, 1).contiguous() # b,n,c -> b,c,n
                s_pe = self.s_pos_emb(s_pe).permute(2, 0, 1) # b,c,n -> n,b,c
                t_pe = template_xyz.permute(0, 2, 1).contiguous()
                t_pe = self.t_pos_emb(t_pe).permute(2, 0, 1)
        else:
            s_pe, t_pe = None, None

        if self.cfg.mask_emb == 'mask_trfm' or self.cfg.mask_emb == 'mask_vanilla':
            t_me = template_mask_ref.unsqueeze(1)
            s_me = search_mask_ref.unsqueeze(1)
            t_me, s_me = self.mask_emb(t_me).permute(2, 0, 1), self.mask_emb(s_me).permute(2, 0, 1)

        if self.cfg.mask_emb == 'mask_trfm':
            t = template_feat.permute(2, 0, 1)
            tt = self.pre_norm(t)
            s = search_feat.permute(2, 0, 1)
            ss = self.pre_norm(s)
            q = self.with_pos_embed(tt, t_pe)
            k = v = self.with_pos_embed(ss, s_pe)
            ps, attn_w = self.attn(q, k, v, attn_mask=None)
            # q = self.with_mask_embed(tt, t_me)
            # k = v = self.with_mask_embed(ss, s_me)
            k = v = s_me
            ms = self.attn(q, k, v, attn_mask=None)[0]
            s = s + self.dropout(ps) + ms

        if self.cfg.ffn_cfg:
            ss = self.pre_ffn_norm(s)
            ss = self.ffn(ss)
            s = s + self.ffn_dropout(ss)

        search_feat = s.permute(1, 2, 0)

        output_dict = dict(
            search_feat=search_feat,
            template_feat=template_feat,
            search_xyz=search_xyz,
            template_xyz=template_xyz,
            search_mask_ref=search_mask_ref,
            template_mask_ref=template_mask_ref
        )

        output_dict.update(input_dict)

        if self.cfg.mask_pred:
            aux_x = self.fc_mask(search_feat)
            search_mask_pred = aux_x.squeeze(1)
            output_dict.update(
                search_mask_ref=search_mask_pred.sigmoid(),
                search_mask_pred=search_mask_pred,
            )

        if self.cfg.center_pred and (self.training or self.cfg.is_last):
            aux_x = self.fc_center(ops.cat([search_feat, search_xyz.permute(0, 2, 1)], dim=1))
            search_center_pred = aux_x.permute(0, 2, 1) + search_xyz

            output_dict.update(
                search_center_ref=search_center_pred,
                search_center_pred=search_center_pred,
            )
        return output_dict

class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer_num = len(cfg.layers_cfg)
        self.layers = nn.ModuleList()
        for idx, layer_cfg in enumerate(cfg.layers_cfg):
            layer_cfg.is_last = (idx == len(cfg.layers_cfg)-1)
            self.layers.append(SelfLayer(layer_cfg))

        self.cfg = cfg

    def forward(self, input_dict):

        output_dict, input_dict_i = dict(), dict()
        l_search_center_ref = []
        # l_search_attn_w, l_template_attn_w = [], []
        for i, search_xyz in enumerate(input_dict['l_search_xyz']):
            input_dict_i['search_xyz'] = input_dict['l_search_xyz'][i]
            input_dict_i['search_feat'] = input_dict['l_search_feat'][i]
            input_dict_i['search_mask_ref'] = input_dict['l_search_mask_ref'][i]
            input_dict_i['template_xyz'] = input_dict['l_template_xyz'][i]
            input_dict_i['template_feat'] = input_dict['l_template_feat'][i]
            input_dict_i['template_mask_ref'] = input_dict['l_template_mask_ref'][i]
            for j, layer in enumerate(self.layers):
                input_dict_i = layer(input_dict_i)
                if self.cfg.layers_cfg[j].mask_pred and self.training:
                    template_mask_pred = input_dict_i.pop('template_mask_pred')
                    search_mask_pred = input_dict_i.pop('search_mask_pred')
                    output_dict.update({
                        'template_mask_pred_%d' % (i * self.layer_num + j): template_mask_pred,
                        'search_mask_pred_%d' % (i * self.layer_num + j): search_mask_pred,
                    })
                if self.cfg.layers_cfg[j].center_pred and self.training:
                    template_center_pred = input_dict_i.pop('template_center_pred')
                    search_center_pred = input_dict_i.pop('search_center_pred')
                    output_dict.update({
                        'template_center_pred_%d' % (i * self.layer_num + j): template_center_pred,
                        'search_center_pred_%d' % (i * self.layer_num + j): search_center_pred,
                    })
            if self.cfg.layers_cfg[-1].center_pred:
                l_search_center_ref.append(input_dict_i['search_center_ref'])
            input_dict['l_search_xyz'][i] = input_dict_i['search_xyz']
            input_dict['l_search_feat'][i] = input_dict_i['search_feat']
            input_dict['l_search_mask_ref'][i] = input_dict_i['search_mask_ref']
            input_dict['l_template_xyz'][i] = input_dict_i['template_xyz']
            input_dict['l_template_feat'][i] = input_dict_i['template_feat']
            input_dict['l_template_mask_ref'][i] = input_dict_i['template_mask_ref']

            # l_search_attn_w.append(input_dict_i['search_attn_w']) # test for visual
            # l_template_attn_w.append(input_dict_i['template_attn_w'])  # test for visual

        output_dict.update(l_search_center_ref = l_search_center_ref)
        # output_dict.update(l_search_attn_w=l_search_attn_w)
        # output_dict.update(l_template_attn_w=l_template_attn_w)
        output_dict.update(input_dict)

        return output_dict

class GenTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer_num = len(cfg.layers_cfg)
        self.layers = nn.ModuleList()
        for idx, layer_cfg in enumerate(cfg.layers_cfg):
            layer_cfg.is_last = (idx == len(cfg.layers_cfg)-1)
            self.layers.append(SelfLayer(layer_cfg))

        self.cfg = cfg

    def forward(self, input_dict):

        output_dict, input_dict_i = dict(), dict()
        for i, _ in enumerate(input_dict['l_template_xyz']):
            input_dict_i['template_xyz'] = input_dict['l_template_xyz'][i]
            input_dict_i['template_feat'] = input_dict['l_template_feat'][i]
            input_dict_i['template_mask_ref'] = input_dict['l_template_mask_ref'][i]
            try:
                input_dict_i['search_xyz'] = input_dict['l_template_xyz'][i+1]
                input_dict_i['search_feat'] = input_dict['l_template_feat'][i+1]
                input_dict_i['search_mask_ref'] = input_dict['l_template_mask_ref'][i+1]
            except:
                input_dict_i['search_xyz'] = input_dict['search_xyz']
                input_dict_i['search_feat'] = input_dict['search_feat']
                input_dict_i['search_mask_ref'] = input_dict['search_mask_ref']

            for j, layer in enumerate(self.layers):
                input_dict_i = layer(input_dict_i)
                if self.cfg.layers_cfg[j].mask_pred and self.training:
                    template_mask_pred = input_dict_i.pop('template_mask_pred')
                    search_mask_pred = input_dict_i.pop('search_mask_pred')
                    output_dict.update({
                        'template_mask_pred_%d' % (i * self.layer_num + j): template_mask_pred,
                        'search_mask_pred_%d' % (i * self.layer_num + j): search_mask_pred,
                    })
                if self.cfg.layers_cfg[j].center_pred and self.training:
                    template_center_pred = input_dict_i.pop('template_center_pred')
                    search_center_pred = input_dict_i.pop('search_center_pred')
                    output_dict.update({
                        'template_center_pred_%d' % (i * self.layer_num + j): template_center_pred,
                        'search_center_pred_%d' % (i * self.layer_num + j): search_center_pred,
                    })
            output_dict.update({
                'template_xyz': input_dict_i['template_xyz'],
                'template_feat': input_dict_i['template_feat'],
                'template_mask_ref': input_dict_i['template_mask_ref'],
                'search_xyz': input_dict_i['search_xyz'],
                'search_feat': input_dict_i['search_feat'],
                'search_mask_ref': input_dict_i['search_mask_ref'],
                'search_center_ref': input_dict_i['search_center_ref'],
            })

        output_dict.update(input_dict)
        return output_dict

class GenTransformerWithUpdate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer_num = len(cfg.layers_cfg)
        self.layers = nn.ModuleList()
        for idx, layer_cfg in enumerate(cfg.layers_cfg):
            layer_cfg.is_last = (idx == len(cfg.layers_cfg)-1)
            self.layers.append(SelfLayer(layer_cfg))

        self.cfg = cfg

    def forward(self, input_dict):

        output_dict, input_dict_i = dict(), dict()
        is_s_t = False
        for i, _ in enumerate(input_dict['l_template_xyz']):
            if not is_s_t:
                input_dict_i['template_xyz'] = input_dict['l_template_xyz'][i]
                input_dict_i['template_feat'] = input_dict['l_template_feat'][i]
                input_dict_i['template_mask_ref'] = input_dict['l_template_mask_ref'][i]
            try:
                input_dict_i['search_xyz'] = input_dict['l_template_xyz'][i+1]
                input_dict_i['search_feat'] = input_dict['l_template_feat'][i+1]
                input_dict_i['search_mask_ref'] = input_dict['l_template_mask_ref'][i+1]
            except:
                input_dict_i['search_xyz'] = input_dict['search_xyz']
                input_dict_i['search_feat'] = input_dict['search_feat']
                input_dict_i['search_mask_ref'] = input_dict['search_mask_ref']

            for j, layer in enumerate(self.layers):
                input_dict_i = layer(input_dict_i)
                if self.cfg.layers_cfg[j].mask_pred and self.training:
                    template_mask_pred = input_dict_i.pop('template_mask_pred')
                    search_mask_pred = input_dict_i.pop('search_mask_pred')
                    output_dict.update({
                        'template_mask_pred_%d' % (i * self.layer_num + j): template_mask_pred,
                        'search_mask_pred_%d' % (i * self.layer_num + j): search_mask_pred,
                    })
                if self.cfg.layers_cfg[j].center_pred and self.training:
                    template_center_pred = input_dict_i.pop('template_center_pred')
                    search_center_pred = input_dict_i.pop('search_center_pred')
                    output_dict.update({
                        'template_center_pred_%d' % (i * self.layer_num + j): template_center_pred,
                        'search_center_pred_%d' % (i * self.layer_num + j): search_center_pred,
                    })
            output_dict.update({
                'template_xyz': input_dict_i['template_xyz'],
                'template_feat': input_dict_i['template_feat'],
                'template_mask_ref': input_dict_i['template_mask_ref'],
                'search_xyz': input_dict_i['search_xyz'],
                'search_feat': input_dict_i['search_feat'],
                'search_mask_ref': input_dict_i['search_mask_ref'],
                'search_center_ref': input_dict_i['search_center_ref'],
            })

            is_s_t = True
            input_dict_i['template_xyz'] = input_dict_i['search_xyz']
            input_dict_i['template_feat'] = input_dict_i['search_feat']
            input_dict_i['template_mask_ref'] = input_dict_i['search_mask_ref']

        output_dict.update(input_dict)
        return output_dict

class CrossTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer_num = len(cfg.layers_cfg)
        self.layers = nn.ModuleList()
        for idx, layer_cfg in enumerate(cfg.layers_cfg):
            layer_cfg.is_last = (idx == len(cfg.layers_cfg)-1)
            self.layers.append(CrossLayer(layer_cfg))

        self.cfg = cfg

    def forward(self, input_dict):

        output_dict, input_dict_i = dict(), dict()
        is_s_t = False
        for i, _ in enumerate(input_dict['l_template_xyz']):
            if not is_s_t:
                input_dict_i['template_xyz'] = input_dict['l_template_xyz'][i]
                input_dict_i['template_feat'] = input_dict['l_template_feat'][i]
                input_dict_i['template_mask_ref'] = input_dict['l_template_mask_ref'][i]
            try:
                input_dict_i['search_xyz'] = input_dict['l_template_xyz'][i+1]
                input_dict_i['search_feat'] = input_dict['l_template_feat'][i+1]
                input_dict_i['search_mask_ref'] = input_dict['l_template_mask_ref'][i+1]
            except:
                input_dict_i['search_xyz'] = input_dict['search_xyz']
                input_dict_i['search_feat'] = input_dict['search_feat']
                input_dict_i['search_mask_ref'] = input_dict['search_mask_ref']

            for j, layer in enumerate(self.layers):
                input_dict_i = layer(input_dict_i)
                if self.cfg.layers_cfg[j].mask_pred and self.training:
                    search_mask_pred = input_dict_i.pop('search_mask_pred')
                    output_dict.update({'search_mask_pred_%d' % (i * self.layer_num + j): search_mask_pred})

                if self.cfg.layers_cfg[j].center_pred and self.training:
                    search_center_pred = input_dict_i.pop('search_center_pred')
                    output_dict.update({'search_center_pred_%d' % (i * self.layer_num + j): search_center_pred})

            output_dict.update({
                'template_xyz': input_dict_i['template_xyz'],
                'template_feat': input_dict_i['template_feat'],
                'template_mask_ref': input_dict_i['template_mask_ref'],
                'search_xyz': input_dict_i['search_xyz'],
                'search_feat': input_dict_i['search_feat'],
                'search_mask_ref': input_dict_i['search_mask_ref'],
                'search_center_ref': input_dict_i['search_center_ref'],
            })

        output_dict.update(input_dict)
        return output_dict

class CrossTransformerWithUpdate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer_num = len(cfg.layers_cfg)
        self.layers = nn.ModuleList()
        for idx, layer_cfg in enumerate(cfg.layers_cfg):
            layer_cfg.is_last = (idx == len(cfg.layers_cfg)-1)
            self.layers.append(CrossLayer(layer_cfg))

        self.cfg = cfg

    def forward(self, input_dict):

        output_dict, input_dict_i = dict(), dict()
        is_s_t = False
        for i, _ in enumerate(input_dict['l_template_xyz']):
            if not is_s_t:
                input_dict_i['template_xyz'] = input_dict['l_template_xyz'][i]
                input_dict_i['template_feat'] = input_dict['l_template_feat'][i]
                input_dict_i['template_mask_ref'] = input_dict['l_template_mask_ref'][i]
            try:
                input_dict_i['search_xyz'] = input_dict['l_template_xyz'][i+1]
                input_dict_i['search_feat'] = input_dict['l_template_feat'][i+1]
                input_dict_i['search_mask_ref'] = input_dict['l_template_mask_ref'][i+1]
            except:
                input_dict_i['search_xyz'] = input_dict['search_xyz']
                input_dict_i['search_feat'] = input_dict['search_feat']
                input_dict_i['search_mask_ref'] = input_dict['search_mask_ref']

            for j, layer in enumerate(self.layers):
                input_dict_i = layer(input_dict_i)
                if self.cfg.layers_cfg[j].mask_pred and self.training:
                    search_mask_pred = input_dict_i.pop('search_mask_pred')
                    output_dict.update({'search_mask_pred_%d' % (i * self.layer_num + j): search_mask_pred})

                if self.cfg.layers_cfg[j].center_pred and self.training:
                    search_center_pred = input_dict_i.pop('search_center_pred')
                    output_dict.update({'search_center_pred_%d' % (i * self.layer_num + j): search_center_pred})

            output_dict.update({
                'template_xyz': input_dict_i['template_xyz'],
                'template_feat': input_dict_i['template_feat'],
                'template_mask_ref': input_dict_i['template_mask_ref'],
                'search_xyz': input_dict_i['search_xyz'],
                'search_feat': input_dict_i['search_feat'],
                'search_mask_ref': input_dict_i['search_mask_ref'],
                'search_center_ref': input_dict_i['search_center_ref'],
            })

            is_s_t = True
            input_dict_i['template_xyz'] = input_dict_i['search_xyz']
            input_dict_i['template_feat'] = input_dict_i['search_feat']
            input_dict_i['template_mask_ref'] = input_dict_i['search_mask_ref']

        output_dict.update(input_dict)
        return output_dict