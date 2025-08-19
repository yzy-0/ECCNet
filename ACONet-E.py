import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from yzyall.yzy.mymodelnew.toolbox.models.segformer.mix_transformer import mit_b4
from mydesignmodel.y_model2.try_moudle.ML_Decoder import CombinedDecoder


class MaxEntropyHintFusion(nn.Module):
    def __init__(self, channel):
        super(MaxEntropyHintFusion, self).__init__()
        self.channel = channel
        self.channel_adjust_rgb = nn.Conv2d(channel, channel, 1)
        self.channel_adjust_depth = nn.Conv2d(channel, channel, 1)

    def _calc_entropy(self, feat):
        B, C, H, W = feat.size()
        feat = feat.view(B, C, -1) 
        prob = F.softmax(feat, dim=2) 
        entropy = - (prob * torch.log(prob + 1e-10)).sum(dim=2) 
        entropy = entropy.view(B, C, 1, 1)
        global_entropy = entropy.mean(dim=1, keepdim=True)
        prob = prob.view(B, C, H, W)  # [B, channel, H, W]
        local_entropy = - (prob * torch.log(prob + 1e-10)).sum(dim=1, keepdim=True)  # [B, 1, H, W]

        return global_entropy, local_entropy

    def _enhance_feature(self, feat, global_entropy, local_entropy, level_weight):
        global_weight = global_entropy * level_weight
        feat_global = feat * global_weight 
        local_weight = local_entropy * (1 - level_weight) 
        feat_local = feat * local_weight 
        enhanced_feat = feat_global + feat_local
        return enhanced_feat

    def forward(self, rgb_feat, depth_feat):
        rgb_feat = self.channel_adjust_rgb(rgb_feat)
        depth_feat = self.channel_adjust_depth(depth_feat)
        rgb_global_entropy, rgb_local_entropy = self._calc_entropy(rgb_feat)
        depth_global_entropy, depth_local_entropy = self._calc_entropy(depth_feat)

        level_weight = torch.sigmoid(torch.tensor(self.channel / 512.0)).to(rgb_feat.device) 

        rgb_enhanced = self._enhance_feature(rgb_feat, rgb_global_entropy, rgb_local_entropy, level_weight)
        depth_enhanced = self._enhance_feature(depth_feat, depth_global_entropy, depth_local_entropy, level_weight)
        total_entropy = rgb_global_entropy + depth_global_entropy + 1e-10  
        w_rgb = rgb_global_entropy / total_entropy  # [B, 1, 1, 1]
        w_depth = depth_global_entropy / total_entropy  # [B, 1, 1, 1]
        w_rgb = w_rgb + (1 - level_weight) * 0.1 
        w_depth = w_depth - (1 - level_weight) * 0.1
        w_rgb = torch.clamp(w_rgb, 0, 1)
        w_depth = torch.clamp(w_depth, 0, 1)

        fused = w_rgb * rgb_enhanced + w_depth * depth_enhanced  # [B, channel, H, W]
        residual = (rgb_feat + depth_feat)
        residual_weight = 1 - level_weight
        output = fused + residual_weight * residual

        return output, (rgb_local_entropy + depth_local_entropy)/2

class Expert_seg(nn.Module):
    def __init__(self, num_class=41, embed_dims=[64, 128, 320, 512]):
        super(Expert_seg, self).__init__()

        self.channels = embed_dims
        self.rgb_d = mit_b4()
        self.cs1 = MaxEntropyHintFusion(64)
        self.cs2 = MaxEntropyHintFusion(128)
        self.po1 = MaxEntropyHintFusion(320)
        self.po2 = MaxEntropyHintFusion(512)

        self.decoder = CombinedDecoder()

    def forward(self, rgb, dep):
        rgb_list = self.rgb_d(rgb) 
        dep_list = self.rgb_d(dep) 

        CS1 = self.cs1(rgb_list[0], dep_list[0])[0]  # [B, 64, H1, W1]
        CS2 = self.cs2(rgb_list[1], dep_list[1])[0]  # [B, 128, H2, W2]
        po1 = self.po1(rgb_list[2], dep_list[2])[0]  # [B, 320, H3, W3]
        po2 = self.po2(rgb_list[3], dep_list[3])[0]  # [B, 512, H4, W4]
        out , x_0, x_1, x_2 = self.decoder(po2, po1, CS2, CS1) 

        return out, x_0, x_1, po2

    def load_pre_sa(self, pre_model1):
        new_state_dict3 = OrderedDict()
        state_dict = torch.load(pre_model1)['state_dict']
        for k, v in state_dict.items():
            name = k[9:]
            new_state_dict3[name] = v
        self.rgb_d.load_state_dict(new_state_dict3, strict=False)
        print('self.backbone_dmit loading')

if __name__ == '__main__':
    net = Expert_seg()
    rgb = torch.randn([2, 3, 480, 640])
    d = torch.randn([2, 3, 480, 640])
    s = net(rgb, d)
    from mydesignmodel.yzy_model.FindTheBestDec.model.FLOP import CalParams
    CalParams(net, rgb, d)
    print("==> Total params: %.2fM" % (sum(p.numel() for p in net.parameters()) / 1e6))
    print("s.shape:", s[0].shape)
