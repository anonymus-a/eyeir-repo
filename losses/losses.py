import torch
from torch import nn
from torch.autograd import Variable

import numpy as np

from networks.encoders import ResNet18EncoderMS

class PerceptualLoss(nn.Module):
    def __init__(self, pretrained=None):

        super(PerceptualLoss, self).__init__()
        if pretrained is None:
            self.feature_extractor = ResNet18EncoderMS(pretrained=True)
        else:
            self.feature_extractor = pretrained
        self.l1_loss = nn.L1Loss()

    @torch.no_grad()
    def forward(self, input, output):
        vgg_real = self.feature_extractor(input)
        vgg_fake = self.feature_extractor(output)

        p0 = self.l1_loss(vgg_real['input'], vgg_fake['input'])
        p1 = self.l1_loss(vgg_real['shallow'], vgg_fake['shallow']) / 2.6
        p2 = self.l1_loss(vgg_real['low'], vgg_fake['low']) / 4.8
        p3 = self.l1_loss(vgg_real['mid'], vgg_fake['mid']) / 3.7
        p4 = self.l1_loss(vgg_real['deep'], vgg_fake['deep']) / 5.6
        p5 = self.l1_loss(vgg_real['out'], vgg_fake['out']) * 10 / 1.5

        return p0 + p1 + p2 + p3 + p4 + p5


class AngleLoss(nn.Module):
    def __init__(self):
        super(AngleLoss, self).__init__()

    def forward(self, pred, target, mask):
        num_valid = torch.sum(mask[:, 0, :, :])
        angle_loss = - torch.sum(torch.mul(mask, torch.mul(pred, target)), 1) 
        return 1 + torch.sum(angle_loss) / num_valid


class RetinaLoss(nn.Module):
    def __init__(self, mode='gradient'):

        super(RetinaLoss, self).__init__()
        self.l1_diff = nn.L1Loss()
        self.l1_diff_mask = MaskL1Loss()
        self.sigmoid = nn.Sigmoid()
        self.downsample = nn.AvgPool2d(2)
        self.level = 3          
        self.eps = 1e-6
        self.mode = mode
        pass

    @staticmethod
    def compute_gradient(img):
        grad_x = img[:, :, 1:, :] - img[:, :, :-1, :]
        grad_y = img[:, :, :, 1:] - img[:, :, :, :-1]
        return grad_x, grad_y

    def compute_exclusion_loss(self, img1, img2, mask=None):
        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)

            if torch.mean(torch.abs(gradx2)) < self.eps or torch.mean(torch.abs(gradx2)) < self.eps:
                gradx_loss.append(0)
                grady_loss.append(0)
                continue

            alphax = 2.0 * torch.mean(torch.abs(gradx1)) / (torch.mean(torch.abs(gradx2)) + self.eps)
            alphay = 2.0 * torch.mean(torch.abs(grady1)) / (torch.mean(torch.abs(grady2)) + self.eps)

            if torch.isnan(alphax) or torch.isnan(alphay):
                gradx_loss.append(0)
                grady_loss.append(0)
                continue

            gradx1_s = (self.sigmoid(gradx1) * 2) - 1
            grady1_s = (self.sigmoid(grady1) * 2) - 1
            gradx2_s = (self.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (self.sigmoid(grady2 * alphay) * 2) - 1

            gradx_loss.append(torch.mean(torch.mul(torch.pow(gradx1_s, 2), torch.pow(gradx2_s, 2)) ** 0.25))
            grady_loss.append(torch.mean(torch.mul(torch.pow(grady1_s, 2), torch.pow(grady2_s, 2)) ** 0.25))

            img1 = self.downsample(img1)
            img2 = self.downsample(img2)

        loss = 0.5 * (sum(gradx_loss) / float(len(gradx_loss)) + sum(grady_loss) / float(len(grady_loss)))

        print(loss)

        return loss

    def compute_gradient_loss(self, img1, img2, mask=None):
        losses = []
        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)

            loss = 0.5 * (self.l1_diff(gradx1, gradx2) + self.l1_diff(grady1, grady2))
            losses.append(loss)

        loss = 0 if len(losses) == 0 else sum(losses) / len(losses)
        return loss

    def compute_edge_loss(self, img1, img2, mask=None):
        if mask is None:
            losses = []
            for l in range(self.level):
                gradx1, grady1 = self.compute_gradient(img1)
                gradx2, grady2 = self.compute_gradient(img2)

                gradx1 = torch.abs(gradx1)
                grady1 = torch.abs(grady1)
                gradx2 = torch.abs(gradx2)
                grady2 = torch.abs(grady2)

                loss = 0.5 * (self.l1_diff(gradx1, gradx2) + self.l1_diff(grady1, grady2))
                losses.append(loss)

            loss = 0 if len(losses) == 0 else sum(losses) / len(losses)
            return loss
        else:
            losses = []
            for l in range(self.level):
                gradx1, grady1 = self.compute_gradient(img1)
                gradx2, grady2 = self.compute_gradient(img2)

                gradx1 = torch.abs(gradx1)
                grady1 = torch.abs(grady1)
                gradx2 = torch.abs(gradx2)
                grady2 = torch.abs(grady2)

                loss = 0.5 * (self.l1_diff_mask(gradx1, gradx2, mask[:, :, 1:, :]) + self.l1_diff_mask(grady1, grady2, mask[:, :, :, 1:]))
                losses.append(loss)

            loss = 0 if len(losses) == 0 else sum(losses) / len(losses)
            return loss


    def forward(self, img_b, img_r, mode=None, mask=None):
        if mode is None:
            mode = self.mode
        # with torch.no_grad():
        if mode == 'exclusion':
            loss = self.compute_exclusion_loss(img_b, img_r, mask)
        elif mode == 'gradient':
            loss = self.compute_gradient_loss(img_b, img_r, mask)
        elif mode == 'edge':
            loss = self.compute_edge_loss(img_b, img_r, mask)
        else:
            raise NotImplementedError("mode should in [exclusion/gradient]")
        return loss

class SecondOrderGradLoss(nn.Module):
    def __init__(self):
        super(SecondOrderGradLoss, self).__init__()
        self.laplace = LaplaceFilter_5D()

    def forward(self, pred, target, mask):
        assert pred.dim() == target.dim(), "inconsistent dimensions in SecondOrderGradLoss"
        lap_pred, mask_lap = self.laplace(pred, mask)
        lap_target, _ = self.laplace(target, mask)
        diff = (lap_pred - lap_target) * mask_lap
        tot_loss = torch.sum(torch.abs(diff)) / torch.sum(mask_lap + 1e-6)
        return tot_loss

class LaplaceFilter_5D(nn.Module):
    def __init__(self):
        super(LaplaceFilter_5D, self).__init__()
        self.edge_conv = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
        edge = np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 2, 1, 0],
            [1, 2, -16, 2, 1],
            [0, 1, 2, 1, 0],
            [0, 0, 1, 0, 0],
        ])
        edge_k = edge
        edge_k = torch.from_numpy(edge_k).float().view(1, 1, 5, 5)
        self.edge_conv.weight = nn.Parameter(edge_k)

        if True:
            self.mask_conv = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
            mask_k = np.array([
                [0, 0, 0.077, 0, 0],
                [0, 0.077, 0.077, 0.077, 0],
                [0.077, 0.077, 0.077, 0.077, 0.077],
                [0, 0.077, 0.077, 0.077, 0],
                [0, 0, 0.077, 0, 0]
            ])
            mask_k = torch.from_numpy(mask_k).float().view(1, 1, 5, 5)
            self.mask_conv.weight = nn.Parameter(mask_k)

        for param in self.parameters():
            param.requires_grad = False

    def apply_laplace_filter(self, x, mask=None):
        out = self.edge_conv(x)
        if mask is not None:
            out_mask = self.mask_conv(mask)
            out_mask[out_mask < 0.95] = 0
            out_mask[out_mask >= 0.95] = 1
            out = torch.mul(out, out_mask)
        else:
            out_mask = None
        return out, out_mask

    def forward(self, x, mask=None):
        out, out_mask = self.apply_laplace_filter(x[:, 0:1, :, :], mask[:, 0:1, :, :] if mask is not None else None)
        for idx in range(1, x.size(1)):
            d_out, d_out_mask = self.apply_laplace_filter(x[:, idx:idx+1, :, :],
                                                          mask[:, idx:idx+1, :, :] if mask is not None else None)
            out = torch.cat((out, d_out), 1)
            if d_out_mask is not None:
                out_mask = torch.cat((out_mask, d_out_mask), 1)

        return out, out_mask

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, pred, target, mask=None):
        if mask is None:
            return torch.mean(torch.pow(pred - target, 2))
        else:
            N = torch.sum(mask)
            diff = pred - target
            diff = torch.mul(diff, mask)
            return torch.sum(torch.pow(diff, 2)) / N 



class MultiScaleGradientLoss(nn.Module):
    def __init__(self, order=1, scale_step=2):
        super(MultiScaleGradientLoss, self).__init__()
        if order == 1:
            self.gradient_loss = L1ImageGradientLoss(step=1)
        elif order == 2:
            self.gradient_loss = SecondOrderGradLoss()
        self.step = scale_step

    def forward(self, pred, target, mask):
        step = self.step

        prediction_1 = pred[:,:,::step,::step]
        prediction_2 = prediction_1[:,:,::step,::step]
        prediction_3 = prediction_2[:,:,::step,::step]

        mask_1 = mask[:,:,::step,::step]
        mask_2 = mask_1[:,:,::step,::step]
        mask_3 = mask_2[:,:,::step,::step]

        gt_1 = target[:,:,::step,::step]
        gt_2 = gt_1[:,:,::step,::step]
        gt_3 = gt_2[:,:,::step,::step]

        final_loss = self.gradient_loss(pred, target, mask)
        final_loss += self.gradient_loss(prediction_1, gt_1, mask_1)
        final_loss += self.gradient_loss(prediction_2, gt_2, mask_2)
        final_loss += self.gradient_loss(prediction_3, gt_3, mask_3)
        return final_loss

class L1ImageGradientLoss(nn.Module):
    def __init__(self, step=2):
        super(L1ImageGradientLoss, self).__init__()
        self.step = step

    def forward(self, pred, target, mask):
        step = self.step

        N = torch.sum(mask)
        diff = pred - target
        diff = torch.mul(diff, mask)

        v_gradient = torch.abs(diff[:, :, 0:-step, :] - diff[:, :, step:, :])
        v_mask = torch.mul(mask[:, :, 0:-step, :], mask[:, :, step:, :])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(diff[:, :, :, 0:-step] - diff[:, :, :, step:])
        h_mask = torch.mul(mask[:, :, :, 0:-step], mask[:, :, :, step:])
        h_gradient = torch.mul(h_gradient, h_mask)

        gradient_loss = (torch.sum(h_gradient) + torch.sum(v_gradient)) / 2.0
        gradient_loss = gradient_loss / (N + 1e-6)

        return gradient_loss

class AngleSmoothLoss(nn.Module):
    def __init__(self, weight=1):
        super(AngleSmoothLoss, self).__init__()
        self.weight = weight
    
    def forward(self, pred, mask):

        N = torch.sum(mask[:, 0, :, :])

        h_mask = torch.mul(mask[:, :, :, 0:-2], mask[:, :, :, 2:])
        h_gradient = torch.sum(torch.mul(h_mask, torch.mul(pred[:, :, :, 0:-2], pred[:, :, :, 2:])), 1)
        h_gradient_loss = 1 - torch.sum(h_gradient) / N

        v_mask = torch.mul(mask[:, :, 0:-2, :], mask[:, :, 2:, :])
        v_gradient = torch.sum(torch.mul(v_mask, torch.mul(pred[:, :, 0:-2, :], pred[:, :, 2:, :])), 1)
        v_gradient_loss = 1 - torch.sum(v_gradient) / N

        gradient_loss = h_gradient_loss + v_gradient_loss

        return self.weight * gradient_loss

class LocalSmoothLoss(nn.Module):
    def __init__(self, half_window_size=1, weight=1):
        super(LocalSmoothLoss, self).__init__()
        self.weight = weight
        self.half_window_size = half_window_size
        x = np.arange(-half_window_size, half_window_size + 1)
        y = np.arange(-half_window_size, half_window_size + 1)
        self.X, self.Y = np.meshgrid(x, y)
    
    def forward(self, pred, mask):

        h = pred.size(2)
        w = pred.size(3)
        num_c = pred.size(1)

        half_window_size = self.half_window_size
        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0

        mask_center = mask[:, :,
                      half_window_size + self.Y[half_window_size, half_window_size]:h - half_window_size + self.Y[half_window_size, half_window_size], \
                      half_window_size + self.X[half_window_size, half_window_size]:w - half_window_size + self.X[half_window_size, half_window_size]]

        R_center = pred[:, :, 
                    half_window_size + self.Y[half_window_size, half_window_size]:h - half_window_size + self.Y[half_window_size, half_window_size], \
                    half_window_size + self.X[half_window_size, half_window_size]:w - half_window_size + self.X[half_window_size, half_window_size]]

        c_idx = 0

        for k in range(0, half_window_size * 2 + 1):
            for l in range(0, half_window_size * 2 + 1):
                R_N = pred[:, :, 
                    half_window_size + self.Y[k, l]:h - half_window_size + self.Y[k, l],
                    half_window_size + self.X[k, l]: w - half_window_size + self.X[k, l]]
                mask_N = mask[:, :, half_window_size + self.Y[k, l]:h - half_window_size + self.Y[k, l],
                         half_window_size + self.X[k, l]: w - half_window_size + self.X[k, l]]

                composed_M = torch.mul(mask_N, mask_center)

                r_diff = torch.mul(composed_M, torch.pow(R_center - R_N, 2))
                total_loss = total_loss + torch.mean(r_diff)
                c_idx = c_idx + 1

        loss =  total_loss / (8.0 * num_c)
        return self.weight * loss

class SHRenderLoss(nn.Module):
    def __init__(self, weight=1):
        super(SHRenderLoss, self).__init__()
        self.weight = weight
    
    def compute_render_loss(self, shading, normal, lighting, mask):
        c1 = 0.429043
        c2 = 0.511664
        c3 = 0.743125
        c4 = 0.886227
        c5 = 0.247708

        batch_size = shading.size()[0]
        dim_mask = mask.size()[1]
        if dim_mask == 1:
            mask = mask.repeat(1, 3, 1, 1)
        mask = mask.permute(0, 2, 3, 1).reshape(batch_size, -1, 3)
        
        shading = shading.permute(0, 2, 3, 1).reshape(batch_size, -1, 3)
        normal = normal.permute(0, 2, 3, 1).reshape(batch_size, -1, 3)
        lighting = lighting.reshape(batch_size, 9, 3)
        
        npix = normal.size()[1]
        ones = torch.ones([batch_size, npix]).cuda()
        BN = torch.stack([
            c4 * ones, 
            2 * c2 * normal[..., 1], 
            2 * c2 * normal[..., 2],
            2 * c2 * normal[..., 0], 
            2 * c1 * normal[..., 0] * normal[..., 1], 
            2 * c1 * normal[..., 1] * normal[..., 2], 
            c3 * normal[..., 2]**2 - c5, 
            2 * c1 * normal[..., 2] * normal[..., 0], 
            c1 * (normal[..., 0]**2 - normal[..., 1]**2)
        ], dim=-1) 

        shading_sh = torch.matmul(BN, lighting)
        
        count = torch.sum(mask)
        return (((shading - shading_sh)**2) * mask).sum() / count

    def forward(self, shading, normal, lighting, mask):
        normal = (normal - 0.5) * 2
        norm = torch.norm(normal, p=2, dim=1, keepdim=True) + 1e-6
        normal_normalized = normal / norm
          
        return self.weight * self.compute_render_loss(shading, normal_normalized, lighting, mask)

class FeatsSuppressLoss(nn.Module):
    def __init__(self, weight=1):
        super(FeatsSuppressLoss, self).__init__()
        self.weight = weight
    
    def forward(self, feats_list):
        N = len(feats_list)
        loss = 0
        for feats in feats_list:
            loss += torch.linalg.norm(feats) / feats.numel()

        return loss / N

class OverExposureLoss(nn.Module):
    def __init__(self, weight=1):
        super(OverExposureLoss, self).__init__()
        self.weight = weight

    def forward(self, x, mask):
        N = torch.sum(mask)
        t = torch.zeros_like(x)
        t = t.fill_(0.95)
        loss = (torch.maximum(x - t, torch.zeros_like(x)) * mask).sum() / N
        return self.weight * loss

class SpecularAwareSuppress(nn.Module):
    def __init__(self, weight=1):
        super(SpecularAwareSuppress, self).__init__()
        self.weight = weight

    def specular_aware_suppress(self, x):
        
        def log_translate_m(x, m):
            factor = 0.6
            return factor * torch.log(x + 1 - m + 1e-6) + m
        
        b = x.shape[0]
        ts = []
        for i in range(0, b):
            max_r = x[i, 0, :, :].max()
            r_sp = torch.clip(log_translate_m(x[i, 0, :, :], max_r), 0, 1)
            max_g = x[i, 1, :, :].max()
            g_sp = torch.clip(log_translate_m(x[i, 1, :, :], max_g), 0, 1)
            max_b = x[i, 2, :, :].max()
            b_sp = torch.clip(log_translate_m(x[i, 2, :, :], max_b), 0, 1)
            ts.append(torch.stack([r_sp, g_sp, b_sp], dim=0))
        res = torch.stack(ts, dim=0)
        return res

    def forward(self, pred, mask):
        N = mask.sum()
        pred_suppress = self.specular_aware_suppress(pred)
        mask_dark = (pred_suppress < 1e-3).float()
        loss = torch.sum(pred * mask_dark * mask) / N
        return self.weight * loss

class ChromeSuppressLoss(nn.Module):
    def __init__(self, weight=1):
        super(ChromeSuppressLoss, self).__init__()

        self.weight = weight

    def forward(self, pred, mask):
        mask = mask[:, 0, :, :]
        N = mask.sum()
        R = pred[:, 0, :, :]
        G = pred[:, 1, :, :]
        B = pred[:, 2, :, :]
        loss = ((torch.pow(R - G, 2) + torch.pow(G - B, 2) + torch.pow(B - R, 2)) * mask).sum() / N
        return self.weight * loss

class DarkLoss(nn.Module):
    def __init__(self, DarkLoss_weight=1):
        super(DarkLoss, self).__init__()
        self.DarkLoss_weight = DarkLoss_weight

    def forward(self, x, mask):
        N = torch.sum(mask)
        loss = ((-x) * mask).sum()
        return self.DarkLoss_weight * loss / N

class MaskLoss(nn.Module):
    def __init__(self, reduction):
        super(MaskLoss, self).__init__()
        self.loss = None
        self.reduction = reduction

    def forward(self, x, y, mask):
        if self.loss == None:
            raise ValueError('losses.py: MaskLoss.loss has not been implemented')
        count = torch.sum(mask)
        loss = self.loss(x, y)
        loss = loss * mask
        if self.reduction == 'all':
            return torch.sum(loss) / (count + 1e-6)
        elif self.reduction == 'none':
            return loss

class MaskMSELoss(MaskLoss):
    def __init__(self, reduction='all'):
        super(MaskMSELoss, self).__init__(reduction)
        self.loss = torch.nn.MSELoss(reduction='none')

class MaskL1Loss(MaskLoss):
    def __init__(self, reduction='all'):
        super(MaskL1Loss, self).__init__(reduction)
        self.loss = torch.nn.L1Loss(reduction='none')