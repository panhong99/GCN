import torch.nn as nn
from einops import rearrange
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from Mask2Former_models.msdeformattn import MSDeformAttnPixelDecoder
from Mask2Former_models.position_encoding import PositionEmbeddingSine
from Mask2Former_models.swin import Swin_transformer

MultiScaleDeformableAttention=None
# try:
#     import MultiScaleDeformableAttention as MSDA
# except ModuleNotFoundError as e:
#     info_string = (
#         "\n\nPlease compile MultiScaleDeformableAttention CUDA op with the following commands:\n"
#         "\t`cd mask2former/modeling/pixel_decoder/ops`\n"
#         "\t`sh make.sh`\n"
#     )
#     raise ModuleNotFoundError(info_string)


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None
def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()



class Masked_attention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.mh_attention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, query, value, key_pos, attn_mask):
        key = value + key_pos

        out = self.mh_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask
        )[0]

        return self.norm(out + query)


class Self_attention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.mh_attention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, query):
        out = self.mh_attention(
            query=query,
            key=query,
            value=query
        )[0]

        return self.norm(out + query)


class FFN(nn.Module):
    def __init__(self, model_dim, inter_dim):
        super(FFN, self).__init__()

        self.ffn = nn.Sequential(
            nn.Linear(model_dim, inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim, model_dim)
        )

        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        x = self.ffn(x) + x
        return self.norm(x)


class MLP(nn.Module):
    def __init__(self, model_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


class Transformer_decoder_block(nn.Module):
    def __init__(self, model_dim=256, num_heads=8):
        super().__init__()

        self.masked_attention = Masked_attention(model_dim, num_heads)
        self.self_attention = Self_attention(model_dim, num_heads)
        self.ffn = FFN(model_dim, 2 * model_dim)

    def forward(self, query, value, key_pos, attn_mask):
        query = self.masked_attention(query, value, key_pos, attn_mask)
        out = self.self_attention(query)
        out = self.ffn(out)

        return out


class Transformer_decoder(nn.Module):
    def __init__(self, n_class=10, L=3, num_query=100, num_features=3, model_dim=256, num_heads=8):
        super().__init__()

        self.num_features = num_features
        self.num_heads = num_heads
        self.transformer_block = nn.ModuleList(
            [Transformer_decoder_block(model_dim=model_dim, num_heads=num_heads) for _ in range(L * 3)])
        self.query = nn.Parameter(torch.rand(num_query, 1, model_dim))

        self.from_features_linear = nn.ModuleList(
            [nn.Conv2d(model_dim, model_dim, kernel_size=1) for _ in range(num_features)])
        self.from_features_bias = nn.ModuleList([nn.Embedding(1, model_dim) for _ in range(num_features)])
        self.pos_emb = PositionEmbeddingSine(model_dim // 2, normalize=True)

        self.decoder_norm = nn.LayerNorm(model_dim)
        self.classfication_module = nn.Linear(model_dim, n_class)
        self.segmentation_module = MLP(model_dim)

    def forward_prediction_heads(self, mask_embed, pix_emb, decoder_layer_size=None):
        mask_embed = self.decoder_norm(mask_embed)
        mask_embed = mask_embed.transpose(0, 1)  # b, 100, 256
        outputs_class = self.classfication_module(mask_embed)
        mask_embed = self.segmentation_module(mask_embed)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, pix_emb)

        if decoder_layer_size is not None:
            attn_mask = F.interpolate(outputs_mask, size=decoder_layer_size, mode="bilinear", align_corners=False)
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                             1) < 0.5).bool()  # head 수 만큼 복사한다. bool 형으로 넣어야 한다. True인 곳이 무시할 픽셀
            attn_mask = attn_mask.detach()
        else:
            attn_mask = None
        return outputs_class, outputs_mask, attn_mask

    def forward(self, features, mask_features):
        query= self.query.expand(self.query.shape[0], features[0].shape[0], self.query.shape[2]) # batch 만큼 복사

        predictions_class = []
        predictions_mask = []

        for i in range(self.num_features):
            b, c, h, w = features[i].shape

            kv = self.from_features_linear[i](features[i])  + self.from_features_bias[i].weight[:, :, None, None]
            kv = rearrange(kv, 'b c h w-> (h w) b c')

            key_pos = self.pos_emb(b, h, w, features[i].device, None)
            key_pos = rearrange(key_pos, 'b c h w -> (h w) b c')

            for j in range(3):
                outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(query, mask_features, decoder_layer_size=(h, w))
                # axial training을 위해 중간 결과를 저장한다.
                predictions_class.append(outputs_class)
                predictions_mask.append(outputs_mask)

                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False # 중간 추출된 mask가 아무것도 가리키지 않을 경우 global context attention으로 처리한다.
                query = self.transformer_block[i * 3 + j](query, kv, key_pos, attn_mask)

        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(query, mask_features, decoder_layer_size=None)
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)



        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': {
                'pred_logits' : predictions_class,
                'pred_masks': predictions_mask,
            }
        }
        return out


# Copyright (c) Facebook, Inc. and its affiliates.
# Modified from https://github.com/facebookresearch/Mask2Former/blob/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a/mask2former/modeling/pixel_decoder/msdeformattn.py

import copy
import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.nn.init import normal_
from detectron2.layers import ShapeSpec
import warnings
from torch.nn.init import xavier_uniform_, constant_

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0
class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 128

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        #try:
        #    output = MSDeformAttnFunction.apply(
        #        value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        #except:
        #    # CPU
        output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        # # For FLOPs calculation only
        # output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output

# MSDeformAttn Transformer encoder in deformable detr
class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self,
                 d_model=256,
                 nhead=8,
                 num_encoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation="relu",
                 num_feature_levels=4,
                 enc_n_points=4,
                 ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                              mask_flatten)

        return memory, spatial_shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 d_ffn=1024,
                 dropout=0.1,
                 activation="relu",
                 n_levels=4,
                 n_heads=8,
                 n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index,
                              padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class Mask2Former(nn.Module):
    def __init__(self):
        super(Mask2Former,self).__init__()
        # small swin tramsformer
        self.backbone = Swin_transformer(
            patch_size = 4,
            window_size = 8,
            merge_size = 2,
            model_dim =96,
            num_layers_in_stage = [2,2,6,2]
        )
        ## Pixel Decoder configuration
        pixel_decoder_config = {}
        pixel_decoder_config['input_shape'] = {}
        pixel_decoder_config['input_shape']['res2'] = ShapeSpec(channels=96, height=None, width=None, stride=4)
        pixel_decoder_config['input_shape']['res3'] = ShapeSpec(channels=192, height=None, width=None, stride=8)
        pixel_decoder_config['input_shape']['res4'] = ShapeSpec(channels=384, height=None, width=None, stride=16)
        pixel_decoder_config['input_shape']['res5'] = ShapeSpec(channels=768, height=None, width=None, stride=32)

        pixel_decoder_config['transformer_dropout'] = 0.0
        pixel_decoder_config['transformer_nheads'] = 8
        pixel_decoder_config['transformer_dim_feedforward'] = 1024
        pixel_decoder_config['transformer_enc_layers'] = 6
        pixel_decoder_config['conv_dims'] = 256
        pixel_decoder_config['mask_dim'] = 256
        pixel_decoder_config['norm'] = 'GN'
        pixel_decoder_config['transformer_in_features'] = ['res3', 'res4', 'res5']
        pixel_decoder_config['common_stride'] = 4

        self.pixel_decoder = MSDeformAttnPixelDecoder(
            input_shape = pixel_decoder_config['input_shape'],
            transformer_dropout = pixel_decoder_config['transformer_dropout'],
            transformer_nheads = pixel_decoder_config['transformer_nheads'],
            transformer_dim_feedforward = pixel_decoder_config['transformer_dim_feedforward'],
            transformer_enc_layers = pixel_decoder_config['transformer_enc_layers'],
            conv_dim = pixel_decoder_config['conv_dims'],
            mask_dim = pixel_decoder_config['mask_dim'],
            norm = pixel_decoder_config['norm'],
            transformer_in_features = pixel_decoder_config['transformer_in_features'],
            common_stride = pixel_decoder_config['common_stride'])
        transformer_decoder_config = {}
        transformer_decoder_config['n_class'] = 10
        transformer_decoder_config['L'] = 3
        transformer_decoder_config['num_query'] = 100
        transformer_decoder_config['num_features'] = 3
        transformer_decoder_config['model_dim'] = 256
        transformer_decoder_config['num_heads'] = 8
        self.transformer_decoder = Transformer_decoder(
            n_class = transformer_decoder_config['n_class'] + 1,
            L = transformer_decoder_config['L'],
            num_query = transformer_decoder_config['num_query'],
            num_features = transformer_decoder_config['num_features'],
            model_dim = transformer_decoder_config['model_dim'],
            num_heads = transformer_decoder_config['num_heads']
        )



    def forward(self,imgs):
        features = self.backbone(imgs)
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)


        out = self.transformer_decoder(multi_scale_features, mask_features)

        mask_pred_results = F.interpolate(
            out["pred_masks"],
            size=(imgs.size(2), imgs.size(3)),
            mode="bilinear",
            align_corners=False,
        )
        return mask_pred_results