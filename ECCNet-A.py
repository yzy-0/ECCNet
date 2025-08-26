import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from yzyall.yzy.mymodelnew.toolbox.models.segformer.mix_transformer import mit_b2


class ComplexityEstimator(nn.Module):


    def __init__(self, in_channels, reduction=64, kernel_size=3, grid_size=16):
        super().__init__()
        self.grid_size = grid_size
        self.compress = nn.Conv2d(in_channels, reduction, kernel_size=1, bias=True)
        self.complexity_conv = nn.Conv2d(reduction, 1, kernel_size=kernel_size,
                                         padding=kernel_size // 2, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.compress(x)
        heat_map = self.complexity_conv(x)
        heat_map = self.sigmoid(heat_map)

        B, C, H, W = heat_map.shape
        pad_h = (self.grid_size - H % self.grid_size) % self.grid_size
        pad_w = (self.grid_size - W % self.grid_size) % self.grid_size
        heat_map = F.pad(heat_map, (0, pad_w, 0, pad_h), mode='reflect')
        H_padded, W_padded = H + pad_h, W + pad_w
        grid_h, grid_w = H_padded // self.grid_size, W_padded // self.grid_size
        heat_map_grid = heat_map.view(B, C, grid_h, self.grid_size, grid_w, self.grid_size)
        heat_map_grid = heat_map_grid.permute(0, 1, 2, 4, 3, 5).contiguous()
        heat_map_grid = heat_map_grid.view(B, C, grid_h, grid_w, self.grid_size * self.grid_size)
        heat_map_avg = torch.mean(heat_map_grid, dim=-1)
        binary_mask = (heat_map_avg >= 0.5).float()

        binary_mask = binary_mask.unsqueeze(-1).repeat(1, 1, 1, 1, self.grid_size * self.grid_size)
        binary_mask = binary_mask.view(B, C, grid_h, grid_w, self.grid_size, self.grid_size)
        binary_mask = binary_mask.permute(0, 1, 2, 4, 3, 5).contiguous()
        binary_mask = binary_mask.view(B, C, H_padded, W_padded)

        binary_mask = binary_mask[:, :, :H, :W]
        return heat_map[:, :, :H, :W], binary_mask[:, :, :H, :W]

class AdaptiveMixFFN(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3,
                                padding=1, groups=hidden_features)

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W, complexity_mask=None):
        x = self.fc1(x)
        x = self.act(x)
        B, N, C = x.shape
        x_spatial = x.transpose(1, 2).view(B, C, H, W)

        if complexity_mask is not None:
            complexity_mask = complexity_mask.view(B, 1, H, W)
            x_conv = self.dwconv(x_spatial)
            x_spatial = torch.where(complexity_mask > 0, x_conv, x_spatial)
        else:
            x_spatial = self.dwconv(x_spatial)

        x = x_spatial.flatten(2).transpose(1, 2)

        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AdaptiveComputationLoss(nn.Module):
    def __init__(self, lambda_efficiency=0.01, target_ratio=0.7):
        super().__init__()
        self.lambda_efficiency = lambda_efficiency
        self.target_ratio = target_ratio  

    def forward(self, binary_masks):
        if not binary_masks:
            return torch.tensor(0.0).to(binary_masks[0].device) if binary_masks else torch.tensor(0.0)
        loss = 0.0

        for mask in binary_masks:
            high_compute_ratio = torch.mean(mask)
            ratio_diff = torch.abs(high_compute_ratio - self.target_ratio)
            loss += ratio_diff

        return self.lambda_efficiency * loss


class CrossStageConsistencyLoss(nn.Module):
    def __init__(self, lambda_consistency=0.005):
        super().__init__()
        self.lambda_consistency = lambda_consistency
        self.mse_loss = nn.MSELoss()

    def forward(self, complexity_maps):
        if not complexity_maps or len(complexity_maps) <= 1:
            return torch.tensor(0.0).to(complexity_maps[0].device) if complexity_maps else torch.tensor(0.0)

        loss = 0.0

        for i in range(len(complexity_maps) - 1):
            curr_map = complexity_maps[i]
            next_map = F.interpolate(
                complexity_maps[i + 1],
                size=curr_map.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            loss += self.mse_loss(curr_map, next_map)

        return self.lambda_consistency * loss


class ComplexityEstimatorWrapper(nn.Module):
    def __init__(self, stage_idx=0, grid_size=16):
        super().__init__()
        self.stage_idx = stage_idx
        if stage_idx <= 1:  
            self.grid_size = grid_size
        else: 
            self.grid_size = grid_size // 2
    def init_estimator(self, in_channels):
        self.complexity_estimator = ComplexityEstimator(
            in_channels=in_channels,
            grid_size=self.grid_size
        )

    def forward(self, x):
        return self.complexity_estimator(x)


class AdaptiveFeatureProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.complexity_maps = []
        self.binary_masks = []

    def process_features(self, features, binary_masks=None):
        if binary_masks is None or len(binary_masks) == 0:
            return features

        processed_features = []

        for i, feature in enumerate(features):
            if i < len(binary_masks):
                mask = binary_masks[i]

                if mask.shape[2:] != feature.shape[2:]:
                    mask = F.interpolate(
                        mask,
                        size=feature.shape[2:],
                        mode='nearest'
                    )
                processed_feature = feature * (0.5 + 0.5 * mask)
                processed_features.append(processed_feature)
            else:
                processed_features.append(feature)

        return processed_features


class AdaptiveSegFormerHead(nn.Module):
    def __init__(self, in_channels, embedding_dim=256, num_classes=40):
        super().__init__()
        self.in_channels = in_channels
        self.linear_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, embedding_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embedding_dim),
                nn.ReLU(inplace=True)
            )
            for dim in in_channels
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(embedding_dim * len(in_channels), embedding_dim, kernel_size=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        self.dn = nn.Conv2d(embedding_dim * len(in_channels), 64, kernel_size=1)
        self.dn2 = nn.Conv2d(embedding_dim * len(in_channels), 128, kernel_size=1)
        self.dn3 = nn.Conv2d(embedding_dim * len(in_channels), 320, kernel_size=1)
        self.classifier = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, features, input_shape):
        base_size = features[0].shape[2:] 
        resized_features = []
        for i, feature in enumerate(features):
            feature = self.linear_layers[i](feature)
            feature = F.interpolate(
                feature,
                size=base_size,
                mode='bilinear',
                align_corners=False
            )
            resized_features.append(feature)
        fused_features = torch.cat(resized_features, dim=1)
        x_0 = self.dn(fused_features)
        x_1 = self.dn2(fused_features)
        x_2 = self.dn3(fused_features)

        fused_features = self.fusion(fused_features)
        output = F.interpolate(
            fused_features,
            size=input_shape,
            mode='bilinear',
            align_corners=False
        )
        output = self.classifier(output)

        return output, x_0, x_1, x_2


class ECCNet-Aux(nn.Module):
    def __init__(self, num_classes=41):
        super().__init__()
        self.rgb_d = mit_b2()
        embed_dims = [64, 128, 320, 512]

        # embed_dims = self.rgb_mit.embed_dims  # [64, 128, 320, 512]
        self.rgb_complexity_estimators = nn.ModuleList([
            ComplexityEstimatorWrapper(stage_idx=i, grid_size=16)
            for i in range(4) 
        ])
        self.d_complexity_estimators = nn.ModuleList([
            ComplexityEstimatorWrapper(stage_idx=i, grid_size=16)
            for i in range(4)  
        ])
        for i, dim in enumerate(embed_dims):
            self.rgb_complexity_estimators[i].init_estimator(dim)
            self.d_complexity_estimators[i].init_estimator(dim)
        self.feature_processor = AdaptiveFeatureProcessor()
        self.decode_head = AdaptiveSegFormerHead(
            in_channels=embed_dims,
            embedding_dim=256,
            num_classes=num_classes
        )

        self.efficiency_loss = AdaptiveComputationLoss(lambda_efficiency=0.01)
        self.consistency_loss = CrossStageConsistencyLoss(lambda_consistency=0.005)
        self.rgb_complexity_maps = []
        self.rgb_binary_masks = []
        self.d_complexity_maps = []
        self.d_binary_masks = []

    def load_pre_sa(self, pre_model1):
        """加载预训练权重"""
        new_state_dict3 = OrderedDict()
        state_dict = torch.load(pre_model1)['state_dict']
        for k, v in state_dict.items():
            name = k[9:]
            new_state_dict3[name] = v
        self.rgb_d.load_state_dict(new_state_dict3, strict=False)
        print('self.backbone_dmit loading')

    def forward(self, rgb, depth):
        input_shape = (rgb.shape[2], rgb.shape[3])
        self.rgb_complexity_maps = []
        self.rgb_binary_masks = []
        self.d_complexity_maps = []
        self.d_binary_masks = []
        rgb_features = self.rgb_d(rgb)
        d_features = self.rgb_d(depth)

        for i, (rgb_feat, d_feat) in enumerate(zip(rgb_features, d_features)):
            rgb_heat_map, rgb_binary_mask = self.rgb_complexity_estimators[i](rgb_feat)
            self.rgb_complexity_maps.append(rgb_heat_map)
            self.rgb_binary_masks.append(rgb_binary_mask)
            d_heat_map, d_binary_mask = self.d_complexity_estimators[i](d_feat)
            self.d_complexity_maps.append(d_heat_map)
            self.d_binary_masks.append(d_binary_mask)

        rgb_features = self.feature_processor.process_features(rgb_features, self.rgb_binary_masks)
        d_features = self.feature_processor.process_features(d_features, self.d_binary_masks)

        fused_features = []
        for rgb_feat, d_feat in zip(rgb_features, d_features):
            fused_features.append(rgb_feat + d_feat) 
        # print('fused',fused_features[3].shape)
        logits, x_0, x_1, x_2 = self.decode_head(fused_features, input_shape)

        if self.training:
            rgb_efficiency_loss = self.efficiency_loss(self.rgb_binary_masks)
            d_efficiency_loss = self.efficiency_loss(self.d_binary_masks)
            rgb_consistency_loss = self.consistency_loss(self.rgb_complexity_maps)
            d_consistency_loss = self.consistency_loss(self.d_complexity_maps)

            efficiency_loss = rgb_efficiency_loss + d_efficiency_loss
            consistency_loss = rgb_consistency_loss + d_consistency_loss

            return logits, efficiency_loss, consistency_loss, x_0, x_1, fused_features[3]
        else:
            return logits, fused_features



# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = ECCNet-Aux(num_classes=40)

    # 测试输入
    rgb = torch.randn([2, 3, 480, 640])
    d = torch.randn([2, 3, 480, 640])

    # 前向传播
    s = model(rgb, d)

    # 计算参数量
    print("==> Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
    # print("fused", fused_feature)
