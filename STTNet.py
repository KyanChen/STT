from Models.BackBone import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class BoTMultiHeadAttention(nn.Module):
    def __init__(self, in_feature_dim, num_heads=8, dim_head=None, dropout_rate=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head or in_feature_dim // num_heads
        self.scale = self.dim_head ** -0.5

        inner_dim = self.dim_head * self.num_heads
        self.weights_qkv = nn.ModuleList([
            nn.Linear(in_feature_dim, inner_dim, bias=False),
            nn.Linear(in_feature_dim, inner_dim, bias=False),
            nn.Linear(in_feature_dim, inner_dim, bias=False)
        ])

        self.out_layer = nn.Sequential(
            nn.Linear(inner_dim, in_feature_dim),
            nn.Dropout(dropout_rate)
        )
        self.layer_norm = nn.LayerNorm(in_feature_dim)

    def forward(self, q_s, k_s=None, v_s=None, pos_emb=None):
        if k_s is None and v_s is None:
            k_s = v_s = q_s
        elif v_s is None:
            v_s = k_s
        q, k, v = [self.weights_qkv[idx](x) for idx, x in enumerate([q_s, k_s, v_s])]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), [q, k, v])
        content_content_att = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if pos_emb is not None:
            pos_emb = rearrange(pos_emb, 'b n (h d) -> b h n d', h=self.num_heads)
            content_position_att = torch.einsum('b h i d, b h j d -> b h i j', q, pos_emb) * self.scale
            att_mat = content_content_att + content_position_att
        else:
            att_mat = content_content_att
        att_mat = att_mat.softmax(dim=-1)

        atted_x = torch.einsum('b h i j , b h j d -> b h i d', att_mat, v)
        atted_x = rearrange(atted_x, 'b h n d -> b n (h d)')
        atted_x = self.out_layer(atted_x)
        out = self.layer_norm(atted_x + q_s)
        return out


class STTNet(nn.Module):
    def __init__(self, in_channel, n_classes, *args, **kwargs):
        super(STTNet, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.img_size = kwargs['IMG_SIZE']
        # kwargs['backbone'] = res18, res50 or vgg16
        self.res_backbone = get_backbone(
            model_name=kwargs['backbone'], num_classes=None, **kwargs
        )

        # kwargs['out_keys'] = ['block_4'] or ['block_5']
        self.last_block = kwargs['out_keys'][-1]

        if '18' in kwargs['backbone']:
            # 512 256 128 64 32 16
            layer_channels = [64, 64, 128, 256, 512]
            self.reduce_dim_in = 256
            self.reduce_dim_out = 256 // 4
        elif '50' in kwargs['backbone']:
            layer_channels = [64, 256, 512, 1024, 2048]
            self.reduce_dim_in = 1024
            self.reduce_dim_out = 1024 // 16
        elif '16' in kwargs['backbone']:
            layer_channels = [64, 128, 256, 512, 512]
            self.reduce_dim_in = 512
            self.reduce_dim_out = 512 // 8

        if self.last_block == 'block5':
            self.f_map_size = self.img_size[0] // 32
        elif self.last_block == 'block4':
            self.f_map_size = self.img_size[0] // 16

        # kwargs['top_k_s'] = 64
        self.top_k_s = kwargs['top_k_s']
        # kwargs['top_k_c'] = 16
        self.top_k_c = kwargs['top_k_c']
        # kwargs['encoder_pos'] = True or False
        self.encoder_pos = kwargs['encoder_pos']
        # kwargs['decoder_pos'] = True or False
        self.decoder_pos = kwargs['decoder_pos']
        # kwargs['model_pattern'] = ['X', 'A', 'S', 'C'] means different features concatenation
        self.model_pattern = kwargs['model_pattern']

        self.cat_num = len(self.model_pattern)
        if 'A' in self.model_pattern:
            self.cat_num += 1

        self.num_head_s = max(2, min(self.top_k_s // 8, 64))
        self.num_head_c = min(2, min(self.top_k_c // 4, 64))

        self.reduce_channel_b5 = nn.Sequential(
            nn.Conv2d(in_channels=self.reduce_dim_in, out_channels=self.reduce_dim_out, kernel_size=1),
            nn.BatchNorm2d(self.reduce_dim_out),
            nn.LeakyReLU()
        )
        # position embedding
        # if self.encoder_pos or self.decoder_pos:
        self.spatial_embedding_h = nn.Parameter(
            torch.randn(1, self.reduce_dim_out, self.f_map_size, 1), requires_grad=True)
        self.spatial_embedding_w = nn.Parameter(
            torch.randn(1, self.reduce_dim_out, 1, self.f_map_size), requires_grad=True)
        self.channel_embedding = nn.Parameter(
            torch.randn(1, self.reduce_dim_out, self.f_map_size ** 2), requires_grad=True)
        # spatial attention ops

        self.get_s_probability = nn.Sequential(
            nn.Conv2d(self.reduce_dim_out, self.reduce_dim_out // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.reduce_dim_out // 4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.reduce_dim_out // 4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # b5 spatial encoder and decoder
        self.tf_encoder_spatial_b5 = BoTMultiHeadAttention(
            in_feature_dim=self.reduce_dim_out,
            num_heads=self.num_head_s
        )
        self.tf_decoder_spatial_b5 = BoTMultiHeadAttention(
            in_feature_dim=self.reduce_dim_out,
            num_heads=self.num_head_s
        )
        # channel attention ops

        self.get_c_probability = nn.Sequential(
            nn.Conv2d(self.reduce_dim_out, self.reduce_dim_out // 8, kernel_size=self.f_map_size),
            nn.BatchNorm2d(self.reduce_dim_out // 8),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.reduce_dim_out // 8, self.reduce_dim_out, kernel_size=1),
            nn.Sigmoid()
        )

        # b5 channel encoder and decoder
        self.tf_encoder_channel_b5 = BoTMultiHeadAttention(
            in_feature_dim=self.f_map_size ** 2,
            num_heads=self.num_head_c
        )
        self.tf_decoder_channel_b5 = BoTMultiHeadAttention(
            in_feature_dim=self.f_map_size ** 2,
            num_heads=self.num_head_c
        )
        self.before_predict_head_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.reduce_dim_out * self.cat_num, out_channels=self.reduce_dim_in, kernel_size=1),
            nn.BatchNorm2d(self.reduce_dim_in),
            nn.LeakyReLU()
        )

        if self.last_block == 'block5':
            self.pre_pixel_shuffle = nn.PixelShuffle(2)
            # 128, 256, 256
            self.pre_double_conv = DoubleConv(
                in_channels=layer_channels[4] // 4,
                out_channels=layer_channels[3],
                mid_channels=layer_channels[3]
            )

        self.pixel_shuffle1 = nn.PixelShuffle(4)
        # 16, 64, 64
        self.double_conv1 = DoubleConv(
            in_channels=layer_channels[3] // 16,
            out_channels=layer_channels[1],
            mid_channels=layer_channels[3] // 4
        )
        # 4, 16, 16
        self.pixel_shuffle2 = nn.PixelShuffle(4)
        self.double_conv2 = DoubleConv(
            in_channels=layer_channels[1] // 16,
            out_channels=layer_channels[1] // 4,
            mid_channels=layer_channels[1] // 4
        )

        last_channels = layer_channels[1] // 4
        # 16, 32
        # 32, 2
        if '18' in kwargs['backbone']:
            scale_factor = 2
        else:
            scale_factor = 1
        self.predict_head_out = nn.Sequential(
            nn.Conv2d(in_channels=last_channels, out_channels=last_channels * scale_factor, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(last_channels * scale_factor),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=last_channels * scale_factor, out_channels=n_classes, kernel_size=3, stride=1, padding=1),
        )

        self.loss_att_branch = nn.Sequential(
            nn.Conv2d(in_channels=self.reduce_dim_out * 2, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, *args, **kwargs):
        x, endpoints = self.res_backbone(x)

        # reduce channel 512 to 128
        x_reduced_channel = self.reduce_channel_b5(x)  # B 128 h w

        prob_s_map = self.get_s_probability(x_reduced_channel)

        prob_c_map = self.get_c_probability(x_reduced_channel)  # B C 1 1
        x_att_s = x_reduced_channel * prob_s_map
        x_att_c = x_reduced_channel * prob_c_map

        output_cat = []
        if 'X' in self.model_pattern:
            output_cat.append(x_reduced_channel)
        if 'A' in self.model_pattern:
            output_cat.append(x_att_s)
            output_cat.append(x_att_c)

        if 'S' in self.model_pattern:
            # spatial pos embedding
            prob_s_vector = rearrange(prob_s_map, 'b c h w -> b (h w) c')
            x_vec_s = rearrange(x_reduced_channel, 'b c h w -> b (h w) c')
            # get top k, k = 16 * 16 // 4  x_b5_reduced_channel_vector
            _, indices_s = torch.topk(prob_s_vector, k=self.top_k_s, dim=1, sorted=False)  # B K 1
            indices_s = repeat(indices_s, 'b k m -> b k (m c)', c=self.reduce_dim_out)
            x_s_vec_topk = torch.gather(x_vec_s, 1, indices_s)  # B K 128
            if self.encoder_pos or self.decoder_pos:
                s_pos_embedding = self.spatial_embedding_h + self.spatial_embedding_w  # 1 128 16 16
                s_pos_embedding = repeat(s_pos_embedding, 'm c h w -> (b m) c h w', b=x.size(0))
                s_pos_embedding_vec = rearrange(s_pos_embedding, 'b c h w -> b (h w) c')
                s_pos_embedding_vec_topk = torch.gather(s_pos_embedding_vec, 1, indices_s)  # B K 128

            if self.encoder_pos is True:
                pos_encoder = s_pos_embedding_vec_topk
            else:
                pos_encoder = None

            # b5 encoder and decoder op
            tf_encoder_s_x = self.tf_encoder_spatial_b5(
                q_s=x_s_vec_topk, k_s=None, v_s=None, pos_emb=pos_encoder
            )
            if self.decoder_pos is True:
                pos_decoder = s_pos_embedding_vec_topk
            else:
                pos_decoder = None

            tf_decoder_s_x = self.tf_decoder_spatial_b5(
                q_s=x_vec_s, k_s=tf_encoder_s_x, v_s=None,
                pos_emb=pos_decoder
            )  # B (16*16) 128

            # B 128 16 16
            tf_decoder_s_x = rearrange(tf_decoder_s_x, 'b (h w) c -> b c h w', h=self.f_map_size)
            output_cat.append(tf_decoder_s_x)

        if 'C' in self.model_pattern:
            # channel attention ops
            prob_c_vec = rearrange(prob_c_map, 'b c h w -> b c (h w)')
            x_vec_c = rearrange(x_reduced_channel, 'b c h w -> b c (h w)')

            # get top k, k = 128 // 4 = 32
            _, indices_c = torch.topk(prob_c_vec, k=self.top_k_c, dim=1, sorted=True)  # b k 1
            indices_c = repeat(indices_c, 'b k m -> b k (m c)', c=self.f_map_size ** 2)
            x_vec_c_topk = torch.gather(x_vec_c, 1, indices_c)  # B K 256
            if self.encoder_pos or self.decoder_pos:
                c_pos_embedding_vec = repeat(self.channel_embedding, 'm len c -> (m b) len c', b=x.size(0))
                c_pos_embedding_vec_topk = torch.gather(c_pos_embedding_vec, 1, indices_c)  # B K 256

            if self.encoder_pos is True:
                pos_encoder = c_pos_embedding_vec_topk
            else:
                pos_encoder = None
            # b5 encoder and decoder op
            tf_encoder_c_x = self.tf_encoder_channel_b5(
                q_s=x_vec_c_topk, k_s=None, v_s=None,
                pos_emb=pos_encoder
            )
            if self.decoder_pos is True:
                pos_decoder = c_pos_embedding_vec_topk
            else:
                pos_decoder = None
            tf_decoder_c_x = self.tf_decoder_channel_b5(
                q_s=x_vec_c, k_s=tf_encoder_c_x, v_s=None,
                pos_emb=pos_decoder
            )  # B 128 (16*16)

            # B 128 16 16
            tf_decoder_c_x = rearrange(tf_decoder_c_x, 'b c (h w) -> b c h w', h=self.f_map_size)
            output_cat.append(tf_decoder_c_x)

        x_cat = torch.cat(output_cat, dim=1)
        x_cat = self.before_predict_head_conv(x_cat)

        x = self.double_conv1(self.pixel_shuffle1(x_cat))
        x = self.double_conv2(self.pixel_shuffle2(x))
        logits = self.predict_head_out(x)

        att_output = torch.cat([x_att_s, x_att_c], dim=1)
        att_branch_output = self.loss_att_branch(att_output)
        return logits, att_branch_output


