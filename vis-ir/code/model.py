import torch
import torch.nn as nn
from utils import *


class Structure_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(48, 32, kernel_size=5, padding=2, bias=True)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, padding=2, bias=True)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=5, padding=2, bias=True)

        self.conv6 = nn.Conv2d(32, 32, kernel_size=5, padding=2, bias=True)
        self.conv7 = nn.Conv2d(32, 32, kernel_size=5, padding=2, bias=True)

        self.conv8 = nn.Conv2d(96, 64, kernel_size=3, padding=1, bias=True)
        self.conv9 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv10 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.avg_pool_2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.sa_conv1_1 = nn.Conv2d(2, 8, 7, padding=3, bias=True)
        self.sa_conv1_2 = nn.Conv2d(8, 1, 3, padding=1, bias=True)
        self.sa_conv2_1 = nn.Conv2d(2, 8, 7, padding=3, bias=True)
        self.sa_conv2_2 = nn.Conv2d(8, 1, 3, padding=1, bias=True)
        self.sa_conv3_1 = nn.Conv2d(2, 8, 7, padding=3, bias=True)
        self.sa_conv3_2 = nn.Conv2d(8, 1, 3, padding=1, bias=True)
        self.sa_conv4_1 = nn.Conv2d(2, 8, 7, padding=3, bias=True)
        self.sa_conv4_2 = nn.Conv2d(8, 1, 3, padding=1, bias=True)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)


    def forward(self, x):
        out1 = nn.functional.leaky_relu(self.conv1(x))
        out2 = nn.functional.leaky_relu(self.conv2(out1))

        out3 = self.conv3(torch.cat((out1, out2), dim = 1))
        avg_out3 = torch.mean(out3, dim=1, keepdim=True)
        max_out3, _ = torch.max(out3, dim=1, keepdim=True)
        attention3 = self.sa_conv1_1(torch.cat([avg_out3, max_out3], dim=1))
        attention3 = self.sa_conv1_2(nn.functional.leaky_relu(attention3))
        out3= nn.functional.leaky_relu(out3 * self.sigmoid(attention3))

        out3_ds = self.avg_pool_2(out3)
        avg_out3d = torch.mean(out3_ds, dim=1, keepdim=True)
        max_out3d, _ = torch.max(out3_ds, dim=1, keepdim=True)
        attention3d = self.sa_conv2_1(torch.cat([avg_out3d, max_out3d], dim=1))
        attention3d = self.sa_conv2_2(nn.functional.leaky_relu(attention3d))
        out3_ds= nn.functional.leaky_relu(out3_ds * self.sigmoid(attention3d))

        out4 = nn.functional.relu(self.conv4(out3_ds))

        out4_ds = self.avg_pool_2(out4)
        avg_out4d = torch.mean(out4_ds, dim=1, keepdim=True)
        max_out4d, _ = torch.max(out4_ds, dim=1, keepdim=True)
        attention4d = self.sa_conv3_1(torch.cat([avg_out4d, max_out4d], dim=1))
        attention4d = self.sa_conv3_2(nn.functional.leaky_relu(attention4d))
        out4_ds = nn.functional.leaky_relu(out4_ds * self.sigmoid(attention4d))

        out5 = nn.functional.relu(self.conv5(out4_ds))

        out4_us = nn.functional.interpolate(out4, scale_factor=2, mode='bicubic', align_corners=True)
        out5_us = nn.functional.interpolate(out5, scale_factor=4, mode='bicubic', align_corners=True)

        out6 = nn.functional.leaky_relu(self.conv6(out4_us))
        out7 = nn.functional.leaky_relu(self.conv7(out5_us))

        out8 = self.conv8(torch.cat((out6, out7, out3), dim=1))
        avg_out8 = torch.mean(out8, dim=1, keepdim=True)
        max_out8, _ = torch.max(out8, dim=1, keepdim=True)
        attention8 = self.sa_conv4_1(torch.cat([avg_out8, max_out8], dim=1))
        attention8 = self.sa_conv4_2(nn.functional.leaky_relu(attention8))
        out8 = nn.functional.leaky_relu(out8 * self.sigmoid(attention8))

        out9 = nn.functional.leaky_relu(self.conv9(out8))
        out10 = self.conv10(out9)
        feas = nn.functional.leaky_relu(out10)
        return feas


class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, padding=0, bias=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=0, bias=True)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=0, bias=True)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, padding=0, bias=True)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=0, bias=True)
        self.conv6 = nn.Conv2d(96, 64, kernel_size=3, padding=0, bias=True)
        self.conv7 = nn.Conv2d(128, 64, kernel_size=3, padding=0, bias=True)
        self.conv8 = nn.Conv2d(64, 32, kernel_size=3, padding=0, bias=True)
        self.conv9 = nn.Conv2d(32, 16, kernel_size=3, padding=0, bias=True)
        self.conv10 = nn.Conv2d(16, 8, kernel_size=3, padding=0, bias=True)
        self.conv11 = nn.Conv2d(8, 4, kernel_size=3, padding=0, bias=True)
        self.conv12 = nn.Conv2d(4, 3, kernel_size=3, padding=0, bias=True)

        self.Norm3 = nn.GroupNorm(num_groups=1, num_channels=3, eps=0.001, affine=False)
        self.Norm16 = nn.GroupNorm(num_groups=1, num_channels=16, eps=0.001, affine=False)
        self.Norm8 = nn.GroupNorm(num_groups=1, num_channels=8, eps=0.001, affine=False)
        self.Norm4 = nn.GroupNorm(num_groups=1, num_channels=4, eps=0.01, affine=False)
        self.Ins_Norm64 = nn.InstanceNorm2d(num_features=64, eps=0.001, affine=False)
        self.Ins_Norm32 = nn.InstanceNorm2d(num_features=32, eps=0.001, affine=False)
        self.Ins_Norm16 = nn.InstanceNorm2d(num_features=16, eps=0.001, affine=False)

        self.tanh = nn.Tanh()

        self.avg_pool_2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.sa_conv1_1 = nn.Conv2d(2, 8, 5, padding=0, bias=True)
        self.sa_conv1_2 = nn.Conv2d(8, 1, 3, padding=0, bias=True)
        self.sa_conv2_1 = nn.Conv2d(2, 8, 5, padding=0, bias=True)
        self.sa_conv2_2 = nn.Conv2d(8, 1, 3, padding=0, bias=True)
        self.sa_conv3_1 = nn.Conv2d(2, 8, 5, padding=0, bias=True)
        self.sa_conv3_2 = nn.Conv2d(8, 1, 3, padding=0, bias=True)

        self.sigmoid = nn.Sigmoid()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Linear(48, 48//2), nn.ReLU(inplace=True), nn.Linear(48//2, 48), nn.Sigmoid())
        self.fc2 = nn.Sequential(nn.Linear(96, 96 // 3), nn.ReLU(inplace=True), nn.Linear(96 // 3, 96), nn.Sigmoid())


    def forward(self, x, alpha1, beta1, alpha2, beta2, alpha3, beta3, r1, r2, modulation=False):
        x = nn.functional.pad(x, (1, 1, 1, 1), mode='reflect')
        out1 = nn.functional.leaky_relu(self.conv1(x))

        out2 = nn.functional.pad(out1, (1, 1, 1, 1), mode='reflect')
        out2 = nn.functional.leaky_relu(self.conv2(out2))

        out3 = nn.functional.pad(out2, (1, 1, 1, 1), mode='reflect')
        out3 = nn.functional.leaky_relu(self.conv3(out3))

        out13 = torch.cat((out3, out1), 1)
        avg_out13 = self.avg_pool(out13).squeeze(-1).squeeze(-1)
        max_out13 = self.max_pool(out13).squeeze(-1).squeeze(-1)
        avg_attention13 = self.fc(avg_out13)
        max_attention13 = self.fc(max_out13)
        attention13 = avg_attention13.unsqueeze(2).unsqueeze(3) + max_attention13.unsqueeze(2).unsqueeze(3)
        out13_atten = out13 * self.sigmoid(attention13)

        out4 = nn.functional.pad(out13_atten, (1, 1, 1, 1), mode='reflect')
        out4 = self.conv4(out4)
        avg_out4 = torch.mean(out4, dim=1, keepdim=True)
        max_out4, _ = torch.max(out4, dim=1, keepdim=True)
        input = torch.cat([avg_out4, max_out4], dim=1)
        input = nn.functional.pad(input, (2, 2, 2, 2), mode='reflect')
        attention4 = nn.functional.leaky_relu(self.sa_conv1_1(input))
        attention4 = nn.functional.pad(attention4, (1, 1, 1, 1), mode = 'reflect')
        attention4 = self.sa_conv1_2(attention4)
        out4 = nn.functional.leaky_relu(out4 * self.sigmoid(attention4))

        out4_ds = self.avg_pool_2(out4)
        out5 = nn.functional.pad(out4_ds, (1, 1, 1, 1), mode='reflect')
        out5 = nn.functional.leaky_relu(self.conv5(out5))

        out3_ds = self.avg_pool_2(out3)
        out35 = torch.cat((out5, out3_ds), 1)
        avg_out35 = torch.mean(out35, dim=1, keepdim=True)
        max_out35, _ = torch.max(out35, dim=1, keepdim=True)
        input = torch.cat([avg_out35, max_out35], dim=1)
        input = nn.functional.pad(input, (2, 2, 2, 2), mode='reflect')
        attention35 = nn.functional.leaky_relu(self.sa_conv2_1(input))
        attention35 = nn.functional.pad(attention35, (1, 1, 1, 1), mode='reflect')
        attention35 = self.sa_conv2_2(attention35)
        out35 = nn.functional.leaky_relu(out35 * self.sigmoid(attention35))
        out35_us = nn.functional.interpolate(out35, scale_factor=2, mode='bicubic', align_corners=True)

        out6 = nn.functional.pad(out35_us, (1, 1, 1, 1), mode='reflect')
        out6 = nn.functional.leaky_relu(self.conv6(out6))

        input = torch.cat((out6, out4), dim=1)
        input = nn.functional.pad(input, (1, 1, 1, 1), mode='reflect')
        out7 = self.conv7(input)
        avg_out7 = torch.mean(out7, dim=1, keepdim=True)
        max_out7, _ = torch.max(out7, dim=1, keepdim=True)
        input = torch.cat([avg_out7, max_out7], dim=1)
        input = nn.functional.pad(input, (2, 2, 2, 2), mode='reflect')
        attention7 = nn.functional.leaky_relu(self.sa_conv3_1(input))
        attention7 = nn.functional.pad(attention7, (1, 1, 1, 1), mode='reflect')
        attention7 = self.sa_conv3_2(attention7)
        out7 = nn.functional.leaky_relu(out7 * self.sigmoid(attention7))

        out8 = nn.functional.pad(out7, (1, 1, 1, 1), mode='reflect')
        out8 = self.conv8(out8)
        out8 = nn.functional.leaky_relu(out8)

        out9 = nn.functional.pad(out8, (1, 1, 1, 1), mode='reflect')
        out9 = self.conv9(out9)
        out9 = nn.functional.leaky_relu(out9)

        out10 = nn.functional.pad(out9, (1, 1, 1, 1), mode='reflect')
        out10 = self.conv10(out10)
        out10 = self.Norm8(out10)
        out10 = nn.functional.leaky_relu(out10)

        out11 = nn.functional.pad(out10, (1, 1, 1, 1), mode='reflect')
        out11 = self.conv11(out11)
        out11 = self.Norm4(out11)
        out11 = nn.functional.leaky_relu(out11)


        out12 = nn.functional.pad(out11, (1, 1, 1, 1), mode='reflect')
        out12 = self.conv12(out12)
        result = self.tanh(out12)/2+0.5

        middle = result

        if modulation:
            middle = alpha1.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3]) * out12 \
                     + beta1.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3])
            middle = self.tanh(middle) / 2 + 0.5

            middle = alpha2.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3]) * middle \
                     + beta2.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3])

            '''ori'''
            r1 = r1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, out10.size(2), out10.size(3))
            middle_ycbcr = rgb2ycbcr(middle)
            middle_y = middle_ycbcr[:, 0:1, :, :]
            middle_y_mean = torch.mean(middle_y, [2, 3]).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, out10.size(2),
                                                                                            out10.size(3))
            middle_y_adjust = torch.clamp((middle_y - middle_y_mean) * r1 + middle_y_mean, min=0, max=1)
            middle_adjust_contrast = ycbcr2rgb(
                torch.cat((middle_y_adjust, middle_ycbcr[:, 1:2, :, :], middle_ycbcr[:, 2:3, :, :]), 1))

            middle_adjust_contrast = alpha3.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2],
                                                                               out10.shape[3]) * middle_adjust_contrast \
                                     + beta3.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3])

            r2 = r2.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, out10.shape[2], out10.shape[3])
            middle_gray = rgb2gray(middle_adjust_contrast)
            middle_r = middle_adjust_contrast[:, 0:1, :, :]
            middle_g = middle_adjust_contrast[:, 1:2, :, :]
            middle_b = middle_adjust_contrast[:, 2:3, :, :]
            mask = 1 - rgb2gray(
                torch.cat((middle_r - middle_gray, middle_g - middle_gray, middle_b - middle_gray), dim=1))
            middle_r2 = middle_r * (1 + r2 * mask) - middle_gray * (r2 * mask)
            middle_g2 = middle_g * (1 + r2 * mask) - middle_gray * (r2 * mask)
            middle_b2 = middle_b * (1 + r2 * mask) - middle_gray * (r2 * mask)

            result = torch.cat((middle_r2, middle_g2, middle_b2), dim=1)
            result = torch.clamp(result, min=0, max=1)


        return result, middle


class ConditionPriorExtractor(nn.Module):
    """Visible / IR joint prior for MoE routing (matches trained ckpt keys)."""

    def __init__(self):
        super().__init__()
        self.vis_conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True)
        self.vis_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)
        self.ir_conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True)
        self.ir_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)
        self.mlp = nn.Sequential(
            nn.Linear(96, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 32),
        )

    def forward(self, img_vis, img_ir):
        vis = nn.functional.pad(img_vis, (1, 1, 1, 1), mode='reflect')
        ir = nn.functional.pad(img_ir, (1, 1, 1, 1), mode='reflect')
        v = nn.functional.leaky_relu(self.vis_conv1(vis))
        v = nn.functional.leaky_relu(self.vis_conv2(v))
        i = nn.functional.leaky_relu(self.ir_conv1(ir))
        i = nn.functional.leaky_relu(self.ir_conv2(i))
        vis_g = nn.functional.adaptive_avg_pool2d(v, 1).flatten(1)
        ir_g = nn.functional.adaptive_avg_pool2d(i, 1).flatten(1)
        z = torch.cat([vis_g, ir_g, torch.abs(vis_g - ir_g)], dim=1)
        return self.mlp(z)


class MoEFusionBlockNoiseTop2(nn.Module):
    """
    Top-2 sparse MoE at out7 (64 ch). Router outputs 8 logits; clean_logits = z[:,:4]+z[:,4:8],
    raw_noise_std = z[:,4:8]; forward routing uses noisy_logits = clean + N(0,1)*softplus(raw)+eps.
    """

    def __init__(self, noise_epsilon=1e-2):
        super().__init__()
        self.noise_epsilon = float(noise_epsilon)
        # When True and model.eval(): top-k uses clean_logits only (reproducible). Training always uses noise.
        self.deterministic_inference = False
        self.router = nn.Sequential(
            nn.Linear(96, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 8),
        )
        self.experts = nn.ModuleList()
        for _ in range(4):
            self.experts.append(
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                )
            )

    def forward(self, x_feat, c):
        x_g = nn.functional.adaptive_avg_pool2d(x_feat, 1).flatten(1)
        ri = torch.cat([x_g, c], dim=1)
        h = self.router[0](ri)
        h = nn.functional.leaky_relu(h)
        z = self.router[2](h)
        clean_logits = z[:, :4] + z[:, 4:8]
        raw_noise_std = z[:, 4:8]
        std = nn.functional.softplus(raw_noise_std) + self.noise_epsilon
        use_noisy_route = self.training or not self.deterministic_inference
        if use_noisy_route:
            route_logits = clean_logits + torch.randn_like(clean_logits) * std
        else:
            route_logits = clean_logits
        top_v, top_i = torch.topk(route_logits, 2, dim=1)
        masked = torch.full_like(clean_logits, float('-inf'))
        masked.scatter_(1, top_i, top_v)
        gate = nn.functional.softmax(masked, dim=1)
        delta = 0
        for k in range(4):
            delta = delta + gate[:, k : k + 1, None, None] * self.experts[k](x_feat)
        out = x_feat + delta
        aux = {
            'gate': gate,
            'avg_gate': gate.mean(dim=0),
            'clean_logits': clean_logits,
            'raw_noise_std': raw_noise_std,
            'routing_mode': 'noisy' if use_noisy_route else 'clean_deterministic',
        }
        return out, aux


class FusionNetWithNoiseTop2MoE(FusionNet):
    """FusionNet with MoE after out7 attention (noise_top2 / top-2 routing)."""

    def __init__(self):
        super().__init__()
        self.prior_extractor = ConditionPriorExtractor()
        self.moe_block = MoEFusionBlockNoiseTop2()

    def forward(self, x, alpha1, beta1, alpha2, beta2, alpha3, beta3, r1, r2, modulation=False):
        img_vis = x[:, :3, :, :]
        img_ir = x[:, 3:, :, :]
        c = self.prior_extractor(img_vis, img_ir)

        x = nn.functional.pad(x, (1, 1, 1, 1), mode='reflect')
        out1 = nn.functional.leaky_relu(self.conv1(x))

        out2 = nn.functional.pad(out1, (1, 1, 1, 1), mode='reflect')
        out2 = nn.functional.leaky_relu(self.conv2(out2))

        out3 = nn.functional.pad(out2, (1, 1, 1, 1), mode='reflect')
        out3 = nn.functional.leaky_relu(self.conv3(out3))

        out13 = torch.cat((out3, out1), 1)
        avg_out13 = self.avg_pool(out13).squeeze(-1).squeeze(-1)
        max_out13 = self.max_pool(out13).squeeze(-1).squeeze(-1)
        avg_attention13 = self.fc(avg_out13)
        max_attention13 = self.fc(max_out13)
        attention13 = avg_attention13.unsqueeze(2).unsqueeze(3) + max_attention13.unsqueeze(2).unsqueeze(3)
        out13_atten = out13 * self.sigmoid(attention13)

        out4 = nn.functional.pad(out13_atten, (1, 1, 1, 1), mode='reflect')
        out4 = self.conv4(out4)
        avg_out4 = torch.mean(out4, dim=1, keepdim=True)
        max_out4, _ = torch.max(out4, dim=1, keepdim=True)
        input = torch.cat([avg_out4, max_out4], dim=1)
        input = nn.functional.pad(input, (2, 2, 2, 2), mode='reflect')
        attention4 = nn.functional.leaky_relu(self.sa_conv1_1(input))
        attention4 = nn.functional.pad(attention4, (1, 1, 1, 1), mode='reflect')
        attention4 = self.sa_conv1_2(attention4)
        out4 = nn.functional.leaky_relu(out4 * self.sigmoid(attention4))

        out4_ds = self.avg_pool_2(out4)
        out5 = nn.functional.pad(out4_ds, (1, 1, 1, 1), mode='reflect')
        out5 = nn.functional.leaky_relu(self.conv5(out5))

        out3_ds = self.avg_pool_2(out3)
        out35 = torch.cat((out5, out3_ds), 1)
        avg_out35 = torch.mean(out35, dim=1, keepdim=True)
        max_out35, _ = torch.max(out35, dim=1, keepdim=True)
        input = torch.cat([avg_out35, max_out35], dim=1)
        input = nn.functional.pad(input, (2, 2, 2, 2), mode='reflect')
        attention35 = nn.functional.leaky_relu(self.sa_conv2_1(input))
        attention35 = nn.functional.pad(attention35, (1, 1, 1, 1), mode='reflect')
        attention35 = self.sa_conv2_2(attention35)
        out35 = nn.functional.leaky_relu(out35 * self.sigmoid(attention35))
        out35_us = nn.functional.interpolate(out35, scale_factor=2, mode='bicubic', align_corners=True)

        out6 = nn.functional.pad(out35_us, (1, 1, 1, 1), mode='reflect')
        out6 = nn.functional.leaky_relu(self.conv6(out6))

        input = torch.cat((out6, out4), dim=1)
        input = nn.functional.pad(input, (1, 1, 1, 1), mode='reflect')
        out7 = self.conv7(input)
        avg_out7 = torch.mean(out7, dim=1, keepdim=True)
        max_out7, _ = torch.max(out7, dim=1, keepdim=True)
        input = torch.cat([avg_out7, max_out7], dim=1)
        input = nn.functional.pad(input, (2, 2, 2, 2), mode='reflect')
        attention7 = nn.functional.leaky_relu(self.sa_conv3_1(input))
        attention7 = nn.functional.pad(attention7, (1, 1, 1, 1), mode='reflect')
        attention7 = self.sa_conv3_2(attention7)
        out7 = nn.functional.leaky_relu(out7 * self.sigmoid(attention7))

        out7, aux = self.moe_block(out7, c)

        out8 = nn.functional.pad(out7, (1, 1, 1, 1), mode='reflect')
        out8 = self.conv8(out8)
        out8 = nn.functional.leaky_relu(out8)

        out9 = nn.functional.pad(out8, (1, 1, 1, 1), mode='reflect')
        out9 = self.conv9(out9)
        out9 = nn.functional.leaky_relu(out9)

        out10 = nn.functional.pad(out9, (1, 1, 1, 1), mode='reflect')
        out10 = self.conv10(out10)
        out10 = self.Norm8(out10)
        out10 = nn.functional.leaky_relu(out10)

        out11 = nn.functional.pad(out10, (1, 1, 1, 1), mode='reflect')
        out11 = self.conv11(out11)
        out11 = self.Norm4(out11)
        out11 = nn.functional.leaky_relu(out11)

        out12 = nn.functional.pad(out11, (1, 1, 1, 1), mode='reflect')
        out12 = self.conv12(out12)
        result = self.tanh(out12) / 2 + 0.5

        middle = result

        if modulation:
            middle = alpha1.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3]) * out12 \
                     + beta1.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3])
            middle = self.tanh(middle) / 2 + 0.5

            middle = alpha2.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3]) * middle \
                     + beta2.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3])

            r1 = r1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, out10.size(2), out10.size(3))
            middle_ycbcr = rgb2ycbcr(middle)
            middle_y = middle_ycbcr[:, 0:1, :, :]
            middle_y_mean = torch.mean(middle_y, [2, 3]).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, out10.size(2),
                                                                                            out10.size(3))
            middle_y_adjust = torch.clamp((middle_y - middle_y_mean) * r1 + middle_y_mean, min=0, max=1)
            middle_adjust_contrast = ycbcr2rgb(
                torch.cat((middle_y_adjust, middle_ycbcr[:, 1:2, :, :], middle_ycbcr[:, 2:3, :, :]), 1))

            middle_adjust_contrast = alpha3.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2],
                                                                               out10.shape[3]) * middle_adjust_contrast \
                                     + beta3.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3])

            r2 = r2.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, out10.shape[2], out10.shape[3])
            middle_gray = rgb2gray(middle_adjust_contrast)
            middle_r = middle_adjust_contrast[:, 0:1, :, :]
            middle_g = middle_adjust_contrast[:, 1:2, :, :]
            middle_b = middle_adjust_contrast[:, 2:3, :, :]
            mask = 1 - rgb2gray(
                torch.cat((middle_r - middle_gray, middle_g - middle_gray, middle_b - middle_gray), dim=1))
            middle_r2 = middle_r * (1 + r2 * mask) - middle_gray * (r2 * mask)
            middle_g2 = middle_g * (1 + r2 * mask) - middle_gray * (r2 * mask)
            middle_b2 = middle_b * (1 + r2 * mask) - middle_gray * (r2 * mask)

            result = torch.cat((middle_r2, middle_g2, middle_b2), dim=1)
            result = torch.clamp(result, min=0, max=1)

        return result, middle, aux


class MoEFusionBlockNoiseTop3(nn.Module):
    """
    Top-3 sparse MoE at out7 (64 ch) with the same router/expert structure as noise_top2.
    Routing:
      clean_logits = z[:,:4] + z[:,4:8]
      raw_noise_std = z[:,4:8]
      std = softplus(raw_noise_std) + noise_epsilon
      if training or (not deterministic_inference): route_logits = clean + randn * std
      else: route_logits = clean
      topk(route_logits, 3) -> masked softmax -> sparse gates over 3 experts
    """

    def __init__(self, noise_epsilon=1e-2):
        super().__init__()
        self.noise_epsilon = float(noise_epsilon)
        self.deterministic_inference = False
        self.router = nn.Sequential(
            nn.Linear(96, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 8),
        )
        self.experts = nn.ModuleList()
        for _ in range(4):
            self.experts.append(
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                )
            )

    def forward(self, x_feat, c):
        x_g = nn.functional.adaptive_avg_pool2d(x_feat, 1).flatten(1)
        ri = torch.cat([x_g, c], dim=1)
        h = self.router[0](ri)
        h = nn.functional.leaky_relu(h)
        z = self.router[2](h)
        clean_logits = z[:, :4] + z[:, 4:8]
        raw_noise_std = z[:, 4:8]
        std = nn.functional.softplus(raw_noise_std) + self.noise_epsilon
        use_noisy_route = self.training or not self.deterministic_inference
        if use_noisy_route:
            route_logits = clean_logits + torch.randn_like(clean_logits) * std
        else:
            route_logits = clean_logits
        top_v, top_i = torch.topk(route_logits, 3, dim=1)
        masked = torch.full_like(clean_logits, float('-inf'))
        masked.scatter_(1, top_i, top_v)
        gate = nn.functional.softmax(masked, dim=1)
        delta = 0
        for k in range(4):
            delta = delta + gate[:, k : k + 1, None, None] * self.experts[k](x_feat)
        out = x_feat + delta
        aux = {
            'gate': gate,
            'avg_gate': gate.mean(dim=0),
            'clean_logits': clean_logits,
            'raw_noise_std': raw_noise_std,
            'routing_mode': 'noisy' if use_noisy_route else 'clean_deterministic',
        }
        return out, aux


class FusionNetWithNoiseTop3MoE(FusionNet):
    """FusionNet + MoE after out7; same backbone as noise_top2; MoE uses top_k=3 noisy routing."""

    def __init__(self):
        super().__init__()
        self.prior_extractor = ConditionPriorExtractor()
        self.moe_block = MoEFusionBlockNoiseTop3()

    def forward(self, x, alpha1, beta1, alpha2, beta2, alpha3, beta3, r1, r2, modulation=False):
        img_vis = x[:, :3, :, :]
        img_ir = x[:, 3:, :, :]
        c = self.prior_extractor(img_vis, img_ir)

        x = nn.functional.pad(x, (1, 1, 1, 1), mode='reflect')
        out1 = nn.functional.leaky_relu(self.conv1(x))

        out2 = nn.functional.pad(out1, (1, 1, 1, 1), mode='reflect')
        out2 = nn.functional.leaky_relu(self.conv2(out2))

        out3 = nn.functional.pad(out2, (1, 1, 1, 1), mode='reflect')
        out3 = nn.functional.leaky_relu(self.conv3(out3))

        out13 = torch.cat((out3, out1), 1)
        avg_out13 = self.avg_pool(out13).squeeze(-1).squeeze(-1)
        max_out13 = self.max_pool(out13).squeeze(-1).squeeze(-1)
        avg_attention13 = self.fc(avg_out13)
        max_attention13 = self.fc(max_out13)
        attention13 = avg_attention13.unsqueeze(2).unsqueeze(3) + max_attention13.unsqueeze(2).unsqueeze(3)
        out13_atten = out13 * self.sigmoid(attention13)

        out4 = nn.functional.pad(out13_atten, (1, 1, 1, 1), mode='reflect')
        out4 = self.conv4(out4)
        avg_out4 = torch.mean(out4, dim=1, keepdim=True)
        max_out4, _ = torch.max(out4, dim=1, keepdim=True)
        input = torch.cat([avg_out4, max_out4], dim=1)
        input = nn.functional.pad(input, (2, 2, 2, 2), mode='reflect')
        attention4 = nn.functional.leaky_relu(self.sa_conv1_1(input))
        attention4 = nn.functional.pad(attention4, (1, 1, 1, 1), mode='reflect')
        attention4 = self.sa_conv1_2(attention4)
        out4 = nn.functional.leaky_relu(out4 * self.sigmoid(attention4))

        out4_ds = self.avg_pool_2(out4)
        out5 = nn.functional.pad(out4_ds, (1, 1, 1, 1), mode='reflect')
        out5 = nn.functional.leaky_relu(self.conv5(out5))

        out3_ds = self.avg_pool_2(out3)
        out35 = torch.cat((out5, out3_ds), 1)
        avg_out35 = torch.mean(out35, dim=1, keepdim=True)
        max_out35, _ = torch.max(out35, dim=1, keepdim=True)
        input = torch.cat([avg_out35, max_out35], dim=1)
        input = nn.functional.pad(input, (2, 2, 2, 2), mode='reflect')
        attention35 = nn.functional.leaky_relu(self.sa_conv2_1(input))
        attention35 = nn.functional.pad(attention35, (1, 1, 1, 1), mode='reflect')
        attention35 = self.sa_conv2_2(attention35)
        out35 = nn.functional.leaky_relu(out35 * self.sigmoid(attention35))
        out35_us = nn.functional.interpolate(out35, scale_factor=2, mode='bicubic', align_corners=True)

        out6 = nn.functional.pad(out35_us, (1, 1, 1, 1), mode='reflect')
        out6 = nn.functional.leaky_relu(self.conv6(out6))

        input = torch.cat((out6, out4), dim=1)
        input = nn.functional.pad(input, (1, 1, 1, 1), mode='reflect')
        out7 = self.conv7(input)
        avg_out7 = torch.mean(out7, dim=1, keepdim=True)
        max_out7, _ = torch.max(out7, dim=1, keepdim=True)
        input = torch.cat([avg_out7, max_out7], dim=1)
        input = nn.functional.pad(input, (2, 2, 2, 2), mode='reflect')
        attention7 = nn.functional.leaky_relu(self.sa_conv3_1(input))
        attention7 = nn.functional.pad(attention7, (1, 1, 1, 1), mode='reflect')
        attention7 = self.sa_conv3_2(attention7)
        out7 = nn.functional.leaky_relu(out7 * self.sigmoid(attention7))

        out7, aux = self.moe_block(out7, c)

        out8 = nn.functional.pad(out7, (1, 1, 1, 1), mode='reflect')
        out8 = self.conv8(out8)
        out8 = nn.functional.leaky_relu(out8)

        out9 = nn.functional.pad(out8, (1, 1, 1, 1), mode='reflect')
        out9 = self.conv9(out9)
        out9 = nn.functional.leaky_relu(out9)

        out10 = nn.functional.pad(out9, (1, 1, 1, 1), mode='reflect')
        out10 = self.conv10(out10)
        out10 = self.Norm8(out10)
        out10 = nn.functional.leaky_relu(out10)

        out11 = nn.functional.pad(out10, (1, 1, 1, 1), mode='reflect')
        out11 = self.conv11(out11)
        out11 = self.Norm4(out11)
        out11 = nn.functional.leaky_relu(out11)

        out12 = nn.functional.pad(out11, (1, 1, 1, 1), mode='reflect')
        out12 = self.conv12(out12)
        result = self.tanh(out12) / 2 + 0.5

        middle = result

        if modulation:
            middle = alpha1.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3]) * out12 \
                     + beta1.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3])
            middle = self.tanh(middle) / 2 + 0.5

            middle = alpha2.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3]) * middle \
                     + beta2.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3])

            r1 = r1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, out10.size(2), out10.size(3))
            middle_ycbcr = rgb2ycbcr(middle)
            middle_y = middle_ycbcr[:, 0:1, :, :]
            middle_y_mean = torch.mean(middle_y, [2, 3]).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, out10.size(2),
                                                                                            out10.size(3))
            middle_y_adjust = torch.clamp((middle_y - middle_y_mean) * r1 + middle_y_mean, min=0, max=1)
            middle_adjust_contrast = ycbcr2rgb(
                torch.cat((middle_y_adjust, middle_ycbcr[:, 1:2, :, :], middle_ycbcr[:, 2:3, :, :]), 1))

            middle_adjust_contrast = alpha3.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2],
                                                                               out10.shape[3]) * middle_adjust_contrast \
                                     + beta3.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3])

            r2 = r2.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, out10.shape[2], out10.shape[3])
            middle_gray = rgb2gray(middle_adjust_contrast)
            middle_r = middle_adjust_contrast[:, 0:1, :, :]
            middle_g = middle_adjust_contrast[:, 1:2, :, :]
            middle_b = middle_adjust_contrast[:, 2:3, :, :]
            mask = 1 - rgb2gray(
                torch.cat((middle_r - middle_gray, middle_g - middle_gray, middle_b - middle_gray), dim=1))
            middle_r2 = middle_r * (1 + r2 * mask) - middle_gray * (r2 * mask)
            middle_g2 = middle_g * (1 + r2 * mask) - middle_gray * (r2 * mask)
            middle_b2 = middle_b * (1 + r2 * mask) - middle_gray * (r2 * mask)

            result = torch.cat((middle_r2, middle_g2, middle_b2), dim=1)
            result = torch.clamp(result, min=0, max=1)

        return result, middle, aux


class MoEFusionBlockNoiseTop1(nn.Module):
    """
    Same 8-dim router as noise_top2: noisy_logits = clean + randn * (softplus(raw_std)+eps), topk(...,1).
    Training-time top1 stabilization (aux warmup) lives in train_content_fusion.py, not here.
    """

    def __init__(self, noise_epsilon=1e-2):
        super().__init__()
        self.noise_epsilon = float(noise_epsilon)
        self.deterministic_inference = False
        self.router = nn.Sequential(
            nn.Linear(96, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 8),
        )
        self.experts = nn.ModuleList()
        for _ in range(4):
            self.experts.append(
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
                )
            )

    def forward(self, x_feat, c):
        x_g = nn.functional.adaptive_avg_pool2d(x_feat, 1).flatten(1)
        ri = torch.cat([x_g, c], dim=1)
        h = self.router[0](ri)
        h = nn.functional.leaky_relu(h)
        z = self.router[2](h)
        clean_logits = z[:, :4] + z[:, 4:8]
        raw_noise_std = z[:, 4:8]
        std = nn.functional.softplus(raw_noise_std) + self.noise_epsilon
        use_noisy_route = self.training or not self.deterministic_inference
        if use_noisy_route:
            route_logits = clean_logits + torch.randn_like(clean_logits) * std
        else:
            route_logits = clean_logits
        top_v, top_i = torch.topk(route_logits, 1, dim=1)
        masked = torch.full_like(clean_logits, float('-inf'))
        masked.scatter_(1, top_i, top_v)
        gate = nn.functional.softmax(masked, dim=1)
        delta = 0
        for k in range(4):
            delta = delta + gate[:, k : k + 1, None, None] * self.experts[k](x_feat)
        out = x_feat + delta
        aux = {
            'gate': gate,
            'avg_gate': gate.mean(dim=0),
            'clean_logits': clean_logits,
            'raw_noise_std': raw_noise_std,
            'routing_mode': 'noisy' if use_noisy_route else 'clean_deterministic',
        }
        return out, aux


class FusionNetWithNoiseTop1MoE(FusionNet):
    """FusionNet + MoE after out7; same backbone as noise_top2; MoE uses top_k=1 noisy routing."""

    def __init__(self):
        super().__init__()
        self.prior_extractor = ConditionPriorExtractor()
        self.moe_block = MoEFusionBlockNoiseTop1()

    def forward(self, x, alpha1, beta1, alpha2, beta2, alpha3, beta3, r1, r2, modulation=False):
        img_vis = x[:, :3, :, :]
        img_ir = x[:, 3:, :, :]
        c = self.prior_extractor(img_vis, img_ir)

        x = nn.functional.pad(x, (1, 1, 1, 1), mode='reflect')
        out1 = nn.functional.leaky_relu(self.conv1(x))

        out2 = nn.functional.pad(out1, (1, 1, 1, 1), mode='reflect')
        out2 = nn.functional.leaky_relu(self.conv2(out2))

        out3 = nn.functional.pad(out2, (1, 1, 1, 1), mode='reflect')
        out3 = nn.functional.leaky_relu(self.conv3(out3))

        out13 = torch.cat((out3, out1), 1)
        avg_out13 = self.avg_pool(out13).squeeze(-1).squeeze(-1)
        max_out13 = self.max_pool(out13).squeeze(-1).squeeze(-1)
        avg_attention13 = self.fc(avg_out13)
        max_attention13 = self.fc(max_out13)
        attention13 = avg_attention13.unsqueeze(2).unsqueeze(3) + max_attention13.unsqueeze(2).unsqueeze(3)
        out13_atten = out13 * self.sigmoid(attention13)

        out4 = nn.functional.pad(out13_atten, (1, 1, 1, 1), mode='reflect')
        out4 = self.conv4(out4)
        avg_out4 = torch.mean(out4, dim=1, keepdim=True)
        max_out4, _ = torch.max(out4, dim=1, keepdim=True)
        input = torch.cat([avg_out4, max_out4], dim=1)
        input = nn.functional.pad(input, (2, 2, 2, 2), mode='reflect')
        attention4 = nn.functional.leaky_relu(self.sa_conv1_1(input))
        attention4 = nn.functional.pad(attention4, (1, 1, 1, 1), mode='reflect')
        attention4 = self.sa_conv1_2(attention4)
        out4 = nn.functional.leaky_relu(out4 * self.sigmoid(attention4))

        out4_ds = self.avg_pool_2(out4)
        out5 = nn.functional.pad(out4_ds, (1, 1, 1, 1), mode='reflect')
        out5 = nn.functional.leaky_relu(self.conv5(out5))

        out3_ds = self.avg_pool_2(out3)
        out35 = torch.cat((out5, out3_ds), 1)
        avg_out35 = torch.mean(out35, dim=1, keepdim=True)
        max_out35, _ = torch.max(out35, dim=1, keepdim=True)
        input = torch.cat([avg_out35, max_out35], dim=1)
        input = nn.functional.pad(input, (2, 2, 2, 2), mode='reflect')
        attention35 = nn.functional.leaky_relu(self.sa_conv2_1(input))
        attention35 = nn.functional.pad(attention35, (1, 1, 1, 1), mode='reflect')
        attention35 = self.sa_conv2_2(attention35)
        out35 = nn.functional.leaky_relu(out35 * self.sigmoid(attention35))
        out35_us = nn.functional.interpolate(out35, scale_factor=2, mode='bicubic', align_corners=True)

        out6 = nn.functional.pad(out35_us, (1, 1, 1, 1), mode='reflect')
        out6 = nn.functional.leaky_relu(self.conv6(out6))

        input = torch.cat((out6, out4), dim=1)
        input = nn.functional.pad(input, (1, 1, 1, 1), mode='reflect')
        out7 = self.conv7(input)
        avg_out7 = torch.mean(out7, dim=1, keepdim=True)
        max_out7, _ = torch.max(out7, dim=1, keepdim=True)
        input = torch.cat([avg_out7, max_out7], dim=1)
        input = nn.functional.pad(input, (2, 2, 2, 2), mode='reflect')
        attention7 = nn.functional.leaky_relu(self.sa_conv3_1(input))
        attention7 = nn.functional.pad(attention7, (1, 1, 1, 1), mode='reflect')
        attention7 = self.sa_conv3_2(attention7)
        out7 = nn.functional.leaky_relu(out7 * self.sigmoid(attention7))

        out7, aux = self.moe_block(out7, c)

        out8 = nn.functional.pad(out7, (1, 1, 1, 1), mode='reflect')
        out8 = self.conv8(out8)
        out8 = nn.functional.leaky_relu(out8)

        out9 = nn.functional.pad(out8, (1, 1, 1, 1), mode='reflect')
        out9 = self.conv9(out9)
        out9 = nn.functional.leaky_relu(out9)

        out10 = nn.functional.pad(out9, (1, 1, 1, 1), mode='reflect')
        out10 = self.conv10(out10)
        out10 = self.Norm8(out10)
        out10 = nn.functional.leaky_relu(out10)

        out11 = nn.functional.pad(out10, (1, 1, 1, 1), mode='reflect')
        out11 = self.conv11(out11)
        out11 = self.Norm4(out11)
        out11 = nn.functional.leaky_relu(out11)

        out12 = nn.functional.pad(out11, (1, 1, 1, 1), mode='reflect')
        out12 = self.conv12(out12)
        result = self.tanh(out12) / 2 + 0.5

        middle = result

        if modulation:
            middle = alpha1.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3]) * out12 \
                     + beta1.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3])
            middle = self.tanh(middle) / 2 + 0.5

            middle = alpha2.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3]) * middle \
                     + beta2.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3])

            r1 = r1.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, out10.size(2), out10.size(3))
            middle_ycbcr = rgb2ycbcr(middle)
            middle_y = middle_ycbcr[:, 0:1, :, :]
            middle_y_mean = torch.mean(middle_y, [2, 3]).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, out10.size(2),
                                                                                            out10.size(3))
            middle_y_adjust = torch.clamp((middle_y - middle_y_mean) * r1 + middle_y_mean, min=0, max=1)
            middle_adjust_contrast = ycbcr2rgb(
                torch.cat((middle_y_adjust, middle_ycbcr[:, 1:2, :, :], middle_ycbcr[:, 2:3, :, :]), 1))

            middle_adjust_contrast = alpha3.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2],
                                                                               out10.shape[3]) * middle_adjust_contrast \
                                     + beta3.unsqueeze(-1).unsqueeze(-1).repeat(1, 3, out10.shape[2], out10.shape[3])

            r2 = r2.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, out10.shape[2], out10.shape[3])
            middle_gray = rgb2gray(middle_adjust_contrast)
            middle_r = middle_adjust_contrast[:, 0:1, :, :]
            middle_g = middle_adjust_contrast[:, 1:2, :, :]
            middle_b = middle_adjust_contrast[:, 2:3, :, :]
            mask = 1 - rgb2gray(
                torch.cat((middle_r - middle_gray, middle_g - middle_gray, middle_b - middle_gray), dim=1))
            middle_r2 = middle_r * (1 + r2 * mask) - middle_gray * (r2 * mask)
            middle_g2 = middle_g * (1 + r2 * mask) - middle_gray * (r2 * mask)
            middle_b2 = middle_b * (1 + r2 * mask) - middle_gray * (r2 * mask)

            result = torch.cat((middle_r2, middle_g2, middle_b2), dim=1)
            result = torch.clamp(result, min=0, max=1)

        return result, middle, aux


class A2V_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=0, bias=True)

        self.fcn_r_1 = nn.Linear(in_features=35, out_features=16)
        self.fcn_r1 = nn.Linear(in_features=16, out_features=1)

        self.fcn_r_2 = nn.Linear(in_features=35, out_features=16)
        self.fcn_r2 = nn.Linear(in_features=16, out_features=1)

        self.fcn_a1 = nn.Linear(in_features=32, out_features=1)
        self.fcn_b1 = nn.Linear(in_features=32, out_features=1)

        self.fcn_a2 = nn.Linear(in_features=32, out_features=1)
        self.fcn_b2 = nn.Linear(in_features=32, out_features=1)

        self.fcn_a3 = nn.Linear(in_features=32, out_features=1)
        self.fcn_b3 = nn.Linear(in_features=32, out_features=1)

        self.reflect_pad = nn.ReflectionPad2d(padding=1)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        feas = []
        out1 = nn.functional.leaky_relu(self.conv1(self.reflect_pad(x)))

        output_size = (1, 1)
        pooling = nn.AdaptiveMaxPool2d(output_size)

        input1 = torch.cat((x, out1), 1)
        a1 = self.fcn_a1(pooling(out1).squeeze(-1).squeeze(-1)) + 1
        b1 = self.fcn_b1(pooling(out1).squeeze(-1).squeeze(-1))

        a2 = self.fcn_a2(pooling(out1).squeeze(-1).squeeze(-1)) + 1
        b2 = self.fcn_b2(pooling(out1).squeeze(-1).squeeze(-1))

        a3 = self.fcn_a3(pooling(out1).squeeze(-1).squeeze(-1)) + 1
        b3 = self.fcn_b3(pooling(out1).squeeze(-1).squeeze(-1))

        r_1 =self.fcn_r_1(pooling(input1).squeeze(-1).squeeze(-1))
        r1 = (self.tanh(self.fcn_r1(nn.functional.leaky_relu(r_1))) + 1) * 3

        r_2 = self.fcn_r_2(pooling(input1).squeeze(-1).squeeze(-1))
        r2 = (self.tanh(self.fcn_r2(nn.functional.leaky_relu(r_2))) + 1) * 3

        return a1, b1, a2, b2, a3, b3, r1, r2
