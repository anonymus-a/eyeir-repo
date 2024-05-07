import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
import pdb
import torch
import tensorboardX

import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from networks.nets import MultiHeadGenerator, DAInverseNet
from losses.losses import \
    AngleLoss, OverExposureLoss, PerceptualLoss, RetinaLoss, SecondOrderGradLoss, ReconstructionLoss, \
    MultiScaleGradientLoss, AngleSmoothLoss, LocalSmoothLoss, SHRenderLoss, MaskMSELoss, \
    DarkLoss, ChromeSuppressLoss, SpecularAwareSuppress, FeatsSuppressLoss
from utils import color, graphic, utils


class Trainer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self._init_logs()
        self._init_loss()
        self._init_model()

        self.optim = optim.Adam([p for p in self.model.parameters()], lr=cfg.train.lr, betas=cfg.train.betas)
        self.model.train()

    def _init_logs(self):
        self.log_dir = self.cfg.log.root
        self.log_img_dir = self.cfg.log.img
        self.log_img_train = self.cfg.log.img_train
        self.log_img_eval = self.cfg.log.img_eval
        self.log_img_test = self.cfg.log.img_test
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.log_img_dir):
            os.makedirs(self.log_img_dir)
        if not os.path.exists(self.log_img_train):
            os.makedirs(self.log_img_train)
        if not os.path.exists(self.log_img_eval):
            os.makedirs(self.log_img_eval)
        if not os.path.exists(self.log_img_test):
            os.makedirs(self.log_img_test)
        self.logger = tensorboardX.SummaryWriter(self.log_dir)
        shutil.copy(self.cfg.cfg_path, os.path.join(self.log_dir, os.path.basename(self.cfg.cfg_path)))

    def _init_loss(self):
        pass

    def _init_model(self):
        pass

    def _get_loss(self):
        pass

    def load_weight(self, model_path):
        self.model.load_state_dict(torch.load(model_path)['model'])

    def update(self, inputs):
        pass

    def infer(self, inputs, mode='eval'):
        pass

    def save(self, iteration):
        torch.save({'model': self.model.state_dict()}, os.path.join(self.log_dir, f'model_iv_{iteration:05d}.pt'))

    def save_log(self, loss_dict, iteration, key='train'):
        for k in loss_dict.keys():
            if isinstance(loss_dict[k], dict):
                self.save_log(loss_dict[k], iteration, f'{key}/{k}')
            else:
                self.logger.add_scalar(f'{key}/{k}', loss_dict[k], iteration)

    def save_image(self, info, iteration, mode):
        for k in info.keys():
            if k == 'name' or ('loss' in k):
                continue

            if 'lighting' in k:
                pass
            elif len(info[k].shape) > 3:
                img = np.transpose(info[k].detach().cpu().numpy()[0], axes=[1, 2, 0])
            else:
                img = np.transpose(info[k].detach().cpu().numpy(), axes=[1, 2, 0])

            path = {
                'train': self.log_img_train,
                'eval': self.log_img_eval,
                'test': self.log_img_test,
            }[mode]
            
            path = os.path.join(path, str(iteration))
            os.makedirs(path, exist_ok=True)

            if 'lighting' in k:
                np.save(os.path.join(path, f'{info["name"]}-{iteration:05d}-{k}.npy'), info[k].detach().cpu().numpy()[0].reshape([9, 3]))
                continue 

            if k == 'pred_normal' or k == 'gt_normal':
                img = np.uint8(np.clip(img, 0, 1) * 255)
                cv2.imwrite(os.path.join(path, f'{info["name"]}-{iteration:05d}-{k}.png'), img[..., ::-1])
                continue
            
            if 'pred_sphere' in k:
                img = np.vectorize(color.lin2srgb)(img)
                img = np.uint8(np.clip(img, 0, 1) * 255)
                cv2.imwrite(os.path.join(path, f'{info["name"]}-{iteration:05d}-{k}.png'), img[..., ::-1])
                continue

            img = np.vectorize(color.lin2srgb)(img)
            img = np.uint8(np.clip(img, 0, 1) * 255)
            cv2.imwrite(os.path.join(path, f'{info["name"]}-{iteration:05d}-{k}.png'), img[..., ::-1])

class Trainer_ASNL(Trainer):
    def __init__(self, cfg):
        super(Trainer_ASNL, self).__init__(cfg)

    def _init_loss(self):
        self.loss_percep = PerceptualLoss()
        self.loss_mse = nn.MSELoss()
        self.loss_edge = RetinaLoss(mode='gradient') 
        self.loss_multiscale_gradient = MultiScaleGradientLoss(order=1) 
        self.loss_second_order_gradient = SecondOrderGradLoss()
        self.loss_reconstruct = ReconstructionLoss()
        self.loss_render = SHRenderLoss() 

        self.loss_normal = AngleLoss()

    def _init_model(self):
        self.model = MultiHeadGenerator(
            encoder_cfg=self.cfg.model.encoder, 
            decoder_cfg_list=[
                self.cfg.model.decoder.albedo, 
                self.cfg.model.decoder.shading, 
                self.cfg.model.decoder.normal, 
                self.cfg.model.decoder.lighting, 
            ]
        )
        
    def save(self, iteration):
        torch.save({'model': self.model.state_dict()}, os.path.join(self.log_dir, f'model_asnl_{iteration:05d}.pt'))
        
    def _get_loss(
            self,
            t_pred_albedo, t_pred_shading, t_pred_normal, t_pred_lighting,
            t_gt_albedo, t_gt_shading, t_gt_normal, t_gt_lighting
        ):
        loss_dict = {}
        loss_dict['single'] = {
            'loss_p_albedo': self.loss_percep(t_pred_albedo, t_gt_albedo),
            'loss_p_shading': self.loss_percep(t_pred_shading, t_gt_shading),
            'loss_l_albedo': self.loss_edge(t_pred_albedo, t_gt_albedo),
            'loss_l_shading': self.loss_edge(t_pred_shading, t_gt_shading),
            'loss_mse_albedo': self.loss_mse(t_pred_albedo, t_gt_albedo),
            'loss_mse_shading': self.loss_mse(t_pred_shading, t_gt_shading),
            'loss_mse_lighting': self.loss_mse(t_pred_lighting, t_gt_lighting),
            'loss_gradient_albedo': self.loss_multiscale_gradient(torch.log(t_pred_albedo + 1e-6), torch.log(t_gt_albedo + 1e-6), torch.ones_like(t_gt_albedo)),
            'loss_gradient_shading': self.loss_second_order_gradient(t_pred_shading, t_gt_shading, torch.ones_like(t_gt_shading)),
            'loss_reconstruct': self.loss_reconstruct(torch.mul(t_gt_albedo, t_gt_shading), torch.mul(t_pred_albedo, t_pred_shading)),
            'loss_normal': self.loss_normal(t_pred_normal, t_gt_normal, torch.ones_like(t_gt_normal)),
            'loss_render': self.loss_render(t_pred_shading, t_pred_normal, t_pred_lighting, torch.ones_like(t_pred_shading)),
        }

        loss_dict['combined'] = {
            'albedo': 
                self.cfg.loss.albedo_p * loss_dict['single']['loss_p_albedo'] + \
                self.cfg.loss.albedo_l * loss_dict['single']['loss_l_albedo'] + \
                self.cfg.loss.albedo_mse * loss_dict['single']['loss_mse_albedo'] + \
                self.cfg.loss.albedo_gradient * loss_dict['single']['loss_gradient_albedo'],
            'shading': 
                self.cfg.loss.shading_p * loss_dict['single']['loss_p_shading'] + \
                self.cfg.loss.shading_l * loss_dict['single']['loss_l_shading'] + \
                self.cfg.loss.shading_mse * loss_dict['single']['loss_mse_shading'] + \
                self.cfg.loss.shading_gradient * loss_dict['single']['loss_gradient_shading'],
            'normal':
                self.cfg.loss.normal * loss_dict['single']['loss_normal'],
            'lighting':
                self.cfg.loss.lighting * loss_dict['single']['loss_mse_lighting'],
            'render':
                self.cfg.loss.render * loss_dict['single']['loss_render'],
            'reconstruct':
                self.cfg.loss.reconstruct * loss_dict['single']['loss_reconstruct'], 
        }

        loss_dict['total'] = sum(loss_dict['combined'].values())
        return loss_dict

    def update(self, inputs):

        t_gt_albedo = Variable(inputs['albedo']).cuda()
        t_gt_shading = Variable(inputs['shading']).cuda()
        t_gt_specular = Variable(inputs['specular']).cuda()
        t_gt_normal = Variable(inputs['normal']).cuda()
        t_gt_lighting = Variable(inputs['lighting']).cuda()
        
        t_in = torch.clamp(t_gt_albedo * t_gt_shading + t_gt_specular, 0, 1)

        self.optim.zero_grad()

        t_pred = self.model(t_in)

        t_pred_albedo = t_pred[0]
        t_pred_shading = t_pred[1]
        t_pred_normal = graphic.out2normal(t_pred[2])
        t_pred_lighting = t_pred[3]

        loss_dict = self._get_loss(
            t_pred_albedo, t_pred_shading, t_pred_normal, t_pred_lighting,
            t_gt_albedo, t_gt_shading, t_gt_normal, t_gt_lighting
        )
        
        loss_total = loss_dict['total']

        loss_total.backward()
        if self.cfg.train.grad_clip is not None:
            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.cfg.train.grad_clip)

        self.optim.step()

        output_dict = {
                'input': t_in,

                'gt_albedo': t_gt_albedo, 'pred_albedo': t_pred_albedo,
                'gt_shading': t_gt_shading, 'pred_shading': t_pred_shading,
                'gt_normal': t_gt_normal, 'pred_normal': t_pred_normal,
                'gt_lighting': t_gt_lighting, 'pred_lighting': t_pred_lighting,
                 
                'name': 'train_' + os.path.basename(inputs['filename'][0]).split('.')[0]
        }

        return {
            'loss_dict': loss_dict,
            'output_dict': output_dict
        }


    def infer(self, inputs, mode='eval'):
        output_dict = {}

        t_gt_albedo = Variable(inputs['albedo']).cuda()
        t_gt_shading = Variable(inputs['shading']).cuda()
        t_gt_specular = Variable(inputs['specular']).cuda()
        t_gt_normal = Variable(inputs['normal']).cuda()
        t_gt_lighting = Variable(inputs['lighting']).cuda()
        t_in = torch.clamp(t_gt_albedo * t_gt_shading + t_gt_specular, 0, 1)

        t_pred = self.model(t_in)

        t_pred_albedo = t_pred[0]
        t_pred_shading = t_pred[1]
        t_pred_normal = graphic.out2normal(t_pred[2])
        t_pred_lighting = t_pred[3]

        
        nm_sphere = torch.tensor(graphic.render_sphere_nm(100, 1), dtype=torch.float32)
        b = t_pred_albedo.shape[0]
        nm_sphere = torch.tile(nm_sphere, (b, 1, 1, 1))
        t_pred_sphere = graphic.lambSH_layer(nm_sphere.cuda(), t_pred_lighting.reshape([b, 9, 3]), am=torch.ones_like(nm_sphere).cuda())
        t_pred_pseudo_shading = graphic.lambSH_layer(t_pred_normal, t_pred_lighting.reshape([b, 9, 3]), torch_style=True)
        t_gt_sphere = graphic.lambSH_layer(nm_sphere.cuda(), t_gt_lighting.reshape([b, 9, 3]), am=torch.ones_like(nm_sphere).cuda())
        t_gt_pseudo_shading = graphic.lambSH_layer(t_gt_normal, t_gt_lighting.reshape([b, 9, 3]), torch_style=True)

        output_dict.update({
            'input': t_in,
            'gt_albedo': t_gt_albedo, 'pred_albedo': t_pred_albedo,
            'gt_shading': t_gt_shading, 'pred_shading': t_pred_shading,
            'gt_normal': t_gt_normal, 'pred_normal': t_pred_normal,
            'gt_lighting': t_gt_lighting, 'pred_lighting': t_pred_lighting,
            'pred_sphere': t_pred_sphere,
            'pred_shading_pseudo': t_pred_pseudo_shading,
            'gt_sphere': t_gt_sphere,
            'gt_shading_pseudo': t_gt_pseudo_shading,
            'name': f'{mode}_' + os.path.basename(inputs['filename'][0]).split('.')[0]
        })

        ret = {
            'output_dict': output_dict
        }

        if mode == 'eval':
            loss_dict = self._get_loss(
                t_pred_albedo, t_pred_shading, t_pred_normal, t_pred_lighting,
                t_gt_albedo, t_gt_shading, t_gt_normal, t_gt_lighting
            )
            ret.update({
                'loss_dict': loss_dict,
            })
        
        return ret



class Trainer_C(Trainer):
    def __init__(self, cfg):
        super(Trainer_C, self).__init__(cfg)

    def _init_loss(self):
        self.loss_percep = PerceptualLoss()
        self.loss_mse = nn.MSELoss()

    def _init_model(self):
        self.model = MultiHeadGenerator(
            encoder_cfg=self.cfg.model.encoder, 
            decoder_cfg_list=[
                self.cfg.model.decoder.specular, 
            ]
        )
        
    def save(self, iteration):
        torch.save({'model': self.model.state_dict()}, os.path.join(self.log_dir, f'model_c_{iteration:05d}.pt'))
        
    def _get_loss(
            self,
            t_pred_specular, t_gt_specular
        ):
        loss_dict = {}
        loss_dict['single'] = {
            'loss_p_specular': self.loss_percep(t_pred_specular, t_gt_specular),
            'loss_mse_specular': self.loss_mse(t_pred_specular, t_gt_specular)
        }

        loss_dict['combined'] = {
            'specular': 
                self.cfg.loss.specular_p * loss_dict['single']['loss_p_specular'] + \
                self.cfg.loss.specular_mse * loss_dict['single']['loss_mse_specular'],
        }

        loss_dict['total'] = sum(loss_dict['combined'].values())
        return loss_dict

    def update(self, inputs):

        t_gt_albedo = Variable(inputs['albedo']).cuda()
        t_gt_shading = Variable(inputs['shading']).cuda()
        t_gt_specular = Variable(inputs['specular']).cuda()
        
        t_in = torch.clamp(t_gt_albedo * t_gt_shading + t_gt_specular, 0, 1)

        self.optim.zero_grad()

        t_pred = self.model(t_in)
        t_pred_specular = t_pred[0]

        loss_dict = self._get_loss(
            t_pred_specular, t_gt_specular
        )

        loss_total = loss_dict['total']
        if self.cfg.train.grad_clip is not None:
            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.cfg.train.grad_clip)
        loss_total.backward()

        self.optim.step()

        output_dict = {
            'input': t_in,
            'gt_specular': t_gt_specular, 
            'pred_specular': t_pred_specular,
            'name': 'train_' + os.path.basename(inputs['filename'][0]).split('.')[0]
        }

        return {
            'loss_dict': loss_dict,
            'output_dict': output_dict
        }

    def infer(self, inputs, mode='eval'):
        output_dict = {}

        t_gt_albedo = Variable(inputs['albedo']).cuda()
        t_gt_shading = Variable(inputs['shading']).cuda()
        t_gt_specular = Variable(inputs['specular']).cuda()
        
        t_in = torch.clamp(t_gt_albedo * t_gt_shading + t_gt_specular, 0, 1)

        t_pred = self.model(t_in)
        t_pred_specular = t_pred[0]

        output_dict.update({
            'input': t_in,
            'gt_specular': t_gt_specular, 
            'pred_specular': t_pred_specular,
            'name': f'{mode}_' + os.path.basename(inputs['filename'][0]).split('.')[0]
        })

        ret = {
            'output_dict': output_dict
        }

        if mode == 'eval':
            loss_dict = self._get_loss(
                t_pred_specular, t_gt_specular
            )
            ret.update({
                'loss_dict': loss_dict,
            })
        
        return ret


class Trainer_DA(Trainer):
    def __init__(self, cfg):
        super(Trainer_DA, self).__init__(cfg)

    def _init_loss(self):
        self.loss_percep = PerceptualLoss()  
        self.loss_mse = nn.MSELoss()
        self.loss_mask_mse = MaskMSELoss()
        self.loss_edge = RetinaLoss(mode='gradient')  
        self.loss_multiscale_gradient = MultiScaleGradientLoss(order=1) 
        self.loss_second_order_gradient = SecondOrderGradLoss()
        self.loss_reconstruct = ReconstructionLoss()
        self.loss_render = SHRenderLoss()
        self.loss_normal = AngleLoss()
        self.loss_angle_smooth = AngleSmoothLoss(weight=100)
        self.loss_albedo_smooth = LocalSmoothLoss(weight=10)
        self.loss_dark = DarkLoss()
        self.loss_specular_aware_suppress = SpecularAwareSuppress()
        self.loss_chrome = ChromeSuppressLoss()
        self.loss_overexpo = OverExposureLoss()
        self.loss_feats_suppress = FeatsSuppressLoss()
    
    def _load_weight_src(self, path_multi, path_specular):
        state_dict = torch.load(path_multi)['model']
        encoder_state_dict = {k[8:]: v for k, v in state_dict.items() if k[:8] == 'encoder.'}
        decoder_0_ab_state_dict = {k[19:]: v for k, v in state_dict.items() if
                                   k[:19] == 'decoder.decoders.0.' and ('fc_prefix' in k or 'fc_mediate' in k)}
        decoder_0_c_state_dict = {k[19:]: v for k, v in state_dict.items() if
                                  k[:19] == 'decoder.decoders.0.' and not ('fc_prefix' in k or 'fc_mediate' in k)}
        decoder_1_ab_state_dict = {k[19:]: v for k, v in state_dict.items() if
                                   k[:19] == 'decoder.decoders.1.' and ('fc_prefix' in k or 'fc_mediate' in k)}
        decoder_1_c_state_dict = {k[19:]: v for k, v in state_dict.items() if
                                  k[:19] == 'decoder.decoders.1.' and not ('fc_prefix' in k or 'fc_mediate' in k)}
        decoder_2_ab_state_dict = {k[19:]: v for k, v in state_dict.items() if
                                   k[:19] == 'decoder.decoders.2.' and ('fc_prefix' in k or 'fc_mediate' in k)}
        decoder_2_c_state_dict = {k[19:]: v for k, v in state_dict.items() if
                                  k[:19] == 'decoder.decoders.2.' and not ('fc_prefix' in k or 'fc_mediate' in k)}

        decoder_3_state_dict = {k[19:]: v for k, v in state_dict.items() if
                                  k[:19] == 'decoder.decoders.3.'}

        self.model.multi_head.encoder.load_state_dict(encoder_state_dict)
        self.model.multi_head.decoder_0_ab.load_state_dict(decoder_0_ab_state_dict)
        self.model.multi_head.decoder_0_c.load_state_dict(decoder_0_c_state_dict)
        self.model.multi_head.decoder_1_ab.load_state_dict(decoder_1_ab_state_dict)
        self.model.multi_head.decoder_1_c.load_state_dict(decoder_1_c_state_dict)
        self.model.multi_head.decoder_2_ab.load_state_dict(decoder_2_ab_state_dict)
        self.model.multi_head.decoder_2_c.load_state_dict(decoder_2_c_state_dict)
        self.model.multi_head.decoder_3.load_state_dict(decoder_3_state_dict)

        state_dict = torch.load(path_specular)['model']
        encoder_state_dict = {k[8:]: v for k, v in state_dict.items() if k[:8] == 'encoder.'}
        decoder_0_ab_state_dict = {k[19:]: v for k, v in state_dict.items() if
                                   k[:19] == 'decoder.decoders.0.' and ('fc_prefix' in k or 'fc_mediate' in k)}
        decoder_0_c_state_dict = {k[19:]: v for k, v in state_dict.items() if
                                  k[:19] == 'decoder.decoders.0.' and not ('fc_prefix' in k or 'fc_mediate' in k)}
        
        self.model.sep_endecoder.encoder.load_state_dict(encoder_state_dict)
        self.model.sep_endecoder.decoder_0_ab.load_state_dict(decoder_0_ab_state_dict)
        self.model.sep_endecoder.decoder_0_c.load_state_dict(decoder_0_c_state_dict)
    
        self._fix_Es()
    

    def _init_model(self):
        self.model = DAInverseNet(
            encoder_cfg=self.cfg.model.encoder,
            decoder_cfg_list=[
                self.cfg.model.decoder.albedo, 
                self.cfg.model.decoder.shading, 
                self.cfg.model.decoder.normal, 
                self.cfg.model.decoder.lighting, 
            ],
            sep_decoder_cfg_list=[
                self.cfg.model.decoder.specular
            ]
        )
        self._load_weight_src(self.cfg.train.syn_asnl_path, self.cfg.train.syn_c_path)

    def load_weight(self, model_path):
        checkpoint_dict = torch.load(model_path)['model']
        if list(checkpoint_dict.keys())[0].startswith('module.'):
            checkpoint_dict = {k[7:]: v for k, v in checkpoint_dict.items()}

        self.model.load_state_dict(checkpoint_dict)

    def _fix_Es(self):

        for p in self.model.multi_head.decoder_0_ab.parameters():
            p.requires_grad = False
        self.model.multi_head.decoder_0_ab.eval()    

        for p in self.model.multi_head.decoder_1_ab.parameters():
            p.requires_grad = False
        self.model.multi_head.decoder_1_ab.eval()
        
        for p in self.model.multi_head.decoder_2_ab.parameters():
            p.requires_grad = False
        self.model.multi_head.decoder_2_ab.eval()

        for p in self.model.multi_head.decoder_3.parameters():
            p.requires_grad = False
        self.model.multi_head.decoder_3.eval()
        
        for p in self.model.sep_endecoder.decoder_0_ab.parameters():
            p.requires_grad = False
        self.model.sep_endecoder.decoder_0_ab.eval()
        
    def unfix(self, name):
        if name == 'ASNL':
            for p in self.model.multi_head.parameters():
                p.requires_grad = True
            self.model.multi_head.train()
            for p in self.model.sep_endecoder.parameters():
                p.requires_grad = False
            self.model.sep_endecoder.eval()
        elif name == 'C':
            for p in self.model.multi_head.parameters():
                p.requires_grad = False
            self.model.multi_head.eval()
            for p in self.model.sep_endecoder.parameters():
                p.requires_grad = True
            self.model.sep_endecoder.train()
        
        self._fix_Es()
    
    def _get_src_asnl_loss(
            self, 
            src_t_pred_albedo, src_t_pred_shading, src_t_pred_normal, src_t_pred_lighting, 
            src_t_gt_albedo, src_t_gt_shading, src_t_gt_normal, src_t_gt_lighting, 
            src_t_gt_mask_eyeball, src_t_gt_mask_eyebrow,
            feats
        ):
        loss_dict = {}
        loss_dict['single'] = {
            'src_loss_p_albedo': self.loss_percep(src_t_pred_albedo, src_t_gt_albedo),
            'src_loss_p_shading': self.loss_percep(src_t_pred_shading, src_t_gt_shading),
            'src_loss_l_albedo': self.loss_edge(src_t_pred_albedo, src_t_gt_albedo),
            'src_loss_l_shading': self.loss_edge(src_t_pred_shading, src_t_gt_shading),
            'src_loss_mse_albedo': self.loss_mse(src_t_pred_albedo, src_t_gt_albedo),
            'src_loss_mse_shading': self.loss_mse(src_t_pred_shading, src_t_gt_shading),
            'src_loss_mse_lighting': self.loss_mse(src_t_pred_lighting, src_t_gt_lighting),
            'src_loss_mse_shading_eyeball': self.loss_mask_mse(src_t_pred_shading, src_t_gt_shading, src_t_gt_mask_eyeball),
            'src_loss_mse_shading_eyebrow': self.loss_mask_mse(src_t_pred_shading, src_t_gt_shading, src_t_gt_mask_eyebrow),
            'src_loss_gradient_albedo': self.loss_multiscale_gradient(torch.log(src_t_pred_albedo + 1e-6), torch.log(src_t_gt_albedo), torch.ones_like(src_t_gt_albedo)),
            'src_loss_gradient_shading': self.loss_second_order_gradient(src_t_pred_shading, src_t_gt_shading, torch.ones_like(src_t_gt_shading)),
            'src_loss_reconstruct': self.loss_reconstruct(torch.mul(src_t_gt_albedo, src_t_gt_shading), torch.mul(src_t_pred_albedo, src_t_pred_shading)),
            'src_loss_normal': self.loss_normal(src_t_pred_normal, src_t_gt_normal, torch.ones_like(src_t_gt_normal)),
            'src_loss_normal_eyeball': self.loss_normal(src_t_pred_normal, src_t_gt_normal, src_t_gt_mask_eyeball),
            'src_loss_normal_eyebrow': self.loss_normal(src_t_pred_normal, src_t_gt_normal, src_t_gt_mask_eyebrow),
            'src_loss_eyeball_render': self.loss_render(src_t_pred_shading, src_t_pred_normal, src_t_pred_lighting, src_t_gt_mask_eyeball),
            'src_loss_eyebrow_render': self.loss_render(src_t_pred_shading, src_t_pred_normal, src_t_pred_lighting, src_t_gt_mask_eyebrow),
            'src_loss_eyeball_smooth': self.loss_angle_smooth(src_t_pred_normal, src_t_gt_mask_eyeball),
            'src_loss_eyebrow_smooth': self.loss_angle_smooth(src_t_pred_normal, src_t_gt_mask_eyebrow), 
            'src_loss_eyeball_dark': self.loss_dark(src_t_pred_shading, src_t_gt_mask_eyeball),
            'src_loss_eyebrow_dark': self.loss_dark(src_t_pred_shading, src_t_gt_mask_eyebrow),
            'src_loss_render': self.loss_render(src_t_pred_shading, src_t_pred_normal, src_t_pred_lighting, torch.ones_like(src_t_pred_shading)),
            'src_loss_local_smooth': self.loss_albedo_smooth(src_t_pred_albedo, torch.ones_like(src_t_pred_albedo) - src_t_gt_mask_eyeball),
            'src_loss_albedo_eyeball': self.loss_mask_mse(src_t_pred_albedo, src_t_gt_albedo, src_t_gt_mask_eyeball),
            'src_loss_feats_suppress': self.loss_feats_suppress(feats['ASNL'])
        }
        loss_dict['combined'] = {
            'src_albedo': 
                self.cfg.loss.src_albedo_p * loss_dict['single']['src_loss_p_albedo'] + \
                self.cfg.loss.src_albedo_l * loss_dict['single']['src_loss_l_albedo'] + \
                self.cfg.loss.src_albedo_mse * loss_dict['single']['src_loss_mse_albedo'] + \
                self.cfg.loss.src_albedo_gradient * loss_dict['single']['src_loss_gradient_albedo'] + \
                self.cfg.loss.src_albedo_smooth * loss_dict['single']['src_loss_local_smooth'] + \
                self.cfg.loss.src_albedo_mse_eyeball * loss_dict['single']['src_loss_albedo_eyeball'],
            'src_shading': 
                self.cfg.loss.src_shading_p * loss_dict['single']['src_loss_p_shading'] + \
                self.cfg.loss.src_shading_l * loss_dict['single']['src_loss_l_shading'] + \
                self.cfg.loss.src_shading_mse * loss_dict['single']['src_loss_mse_shading'] + \
                self.cfg.loss.src_shading_gradient * loss_dict['single']['src_loss_gradient_shading'] + \
                self.cfg.loss.src_shading_mse_eyeball * loss_dict['single']['src_loss_mse_shading_eyeball'] + \
                self.cfg.loss.src_shading_mse_eyebrow * loss_dict['single']['src_loss_mse_shading_eyebrow'] + \
                self.cfg.loss.src_shading_dark_eyeball * loss_dict['single']['src_loss_eyeball_dark'] + \
                self.cfg.loss.src_shading_dark_eyebrow *  loss_dict['single']['src_loss_eyebrow_dark'],
            'src_normal':
                self.cfg.loss.src_normal * loss_dict['single']['src_loss_normal'] + \
                self.cfg.loss.src_normal_eyeball * loss_dict['single']['src_loss_normal_eyeball'] + \
                self.cfg.loss.src_normal_eyebrow * loss_dict['single']['src_loss_normal_eyebrow'] + \
                self.cfg.loss.src_normal_smooth_eyeball * loss_dict['single']['src_loss_eyeball_smooth'] + \
                self.cfg.loss.src_normal_smooth_eyebrow * loss_dict['single']['src_loss_eyebrow_smooth'],
            'src_lighting':
                self.cfg.loss.src_lighting * loss_dict['single']['src_loss_mse_lighting'],
            'src_render':
                self.cfg.loss.src_render * loss_dict['single']['src_loss_render'] + \
                self.cfg.loss.src_render_eyeball * loss_dict['single']['src_loss_eyeball_render'] + \
                self.cfg.loss.src_render_eyebrow * loss_dict['single']['src_loss_eyebrow_render'],
            'src_reconstruct':
                self.cfg.loss.src_reconstruct * loss_dict['single']['src_loss_reconstruct'], 
            'src_feats_suppress':
                self.cfg.loss.src_feats_suppress * loss_dict['single']['src_loss_feats_suppress']
        }
        loss_dict['total'] = sum(loss_dict['combined'].values())
        return loss_dict

    def _get_tgt_asnl_loss(
            self, 
            tgt_t_pred_albedo, tgt_t_pred_shading, tgt_t_pred_normal, tgt_t_pred_lighting, tgt_t_pred_specular, 
            tgt_t_in, tgt_t_gt_peripheral, tgt_t_gt_eyebrow, tgt_t_gt_skin
        ):
        loss_dict = {}
        loss_dict['single'] = {
            'tgt_loss_reconstruct': self.loss_reconstruct(tgt_t_in, tgt_t_pred_albedo * tgt_t_pred_shading + tgt_t_pred_specular, mask=tgt_t_gt_skin),
            'tgt_loss_eyeball_render': self.loss_render(tgt_t_pred_shading, tgt_t_pred_normal, tgt_t_pred_lighting, tgt_t_gt_peripheral),
            'tgt_loss_eyebrow_render': self.loss_render(tgt_t_pred_shading, tgt_t_pred_normal, tgt_t_pred_lighting, tgt_t_gt_eyebrow),
            'tgt_loss_eyeball_smooth': self.loss_angle_smooth(tgt_t_pred_normal, tgt_t_gt_peripheral),
            'tgt_loss_eyebrow_smooth': self.loss_angle_smooth(tgt_t_pred_normal, tgt_t_gt_eyebrow),
            'tgt_loss_eyeball_dark': self.loss_dark(tgt_t_pred_shading, tgt_t_gt_peripheral),
            'tgt_loss_eyebrow_dark': self.loss_dark(tgt_t_pred_shading, tgt_t_gt_eyebrow),
            'tgt_loss_albedo_smooth_skin': self.loss_albedo_smooth(tgt_t_pred_albedo, tgt_t_gt_skin - tgt_t_gt_eyebrow - tgt_t_gt_peripheral),
            'tgt_loss_render': self.loss_render(tgt_t_pred_shading, tgt_t_pred_normal, tgt_t_pred_lighting, tgt_t_gt_skin),
            # 'tgt_loss_overexpo': self.loss_overexpo(tgt_t_pred_albedo, tgt_t_gt_peripheral)
        }
        loss_dict['combined'] = {
            'tgt_reonstruct': 
                self.cfg.loss.tgt_reconstruct * loss_dict['single']['tgt_loss_reconstruct'],
            'tgt_render': 
                self.cfg.loss.tgt_render * loss_dict['single']['tgt_loss_render'] + \
                self.cfg.loss.tgt_render_eyeball * loss_dict['single']['tgt_loss_eyeball_render'] + \
                self.cfg.loss.tgt_render_eyebrow * loss_dict['single']['tgt_loss_eyebrow_render'],
            'tgt_shading': 
                self.cfg.loss.tgt_shading_dark_eyeball * loss_dict['single']['tgt_loss_eyeball_dark'] + \
                self.cfg.loss.tgt_shading_dark_eyebrow * loss_dict['single']['tgt_loss_eyebrow_dark'],
            'tgt_normal': 
                self.cfg.loss.tgt_normal_smooth_eyeball * loss_dict['single']['tgt_loss_eyeball_smooth'] + \
                self.cfg.loss.tgt_normal_smooth_eyebrow * loss_dict['single']['tgt_loss_eyebrow_smooth'],
            'tgt_albedo': 
                self.cfg.loss.tgt_albedo_smooth * loss_dict['single']['tgt_loss_albedo_smooth_skin'] #+ \
                # self.cfg.loss.tgt_albedo_overexpo * loss_dict['single']['tgt_loss_overexpo']
        }
        loss_dict['total'] = sum(loss_dict['combined'].values())
        return loss_dict
    
    def _get_asnl_loss(
            self, 
            src_t_pred_albedo, src_t_pred_shading, src_t_pred_normal, src_t_pred_lighting, 
            src_t_gt_albedo, src_t_gt_shading, src_t_gt_normal, src_t_gt_lighting, 
            src_t_gt_mask_eyeball, src_t_gt_mask_eyebrow,
            feats,
            tgt_t_pred_albedo, tgt_t_pred_shading, tgt_t_pred_normal, tgt_t_pred_lighting, tgt_t_pred_specular, 
            tgt_t_in, tgt_t_gt_peripheral, tgt_t_gt_eyebrow, tgt_t_gt_skin
        ):
        src_loss_dict = self._get_src_asnl_loss(
            src_t_pred_albedo, src_t_pred_shading, src_t_pred_normal, src_t_pred_lighting, 
            src_t_gt_albedo, src_t_gt_shading, src_t_gt_normal, src_t_gt_lighting, 
            src_t_gt_mask_eyeball, src_t_gt_mask_eyebrow,
            feats
        )

        tgt_loss_dict = self._get_tgt_asnl_loss(
            tgt_t_pred_albedo, tgt_t_pred_shading, tgt_t_pred_normal, tgt_t_pred_lighting, tgt_t_pred_specular, 
            tgt_t_in, tgt_t_gt_peripheral, tgt_t_gt_eyebrow, tgt_t_gt_skin
        )
        
        loss_total = \
            self.cfg.loss.src * src_loss_dict['total'] + \
            self.cfg.loss.tgt * tgt_loss_dict['total']

        loss_dict = {
            'src': src_loss_dict,
            'tgt': tgt_loss_dict,
            'total': loss_total
        }

        return loss_dict

    def update_ASNL(self, src_inputs, tgt_inputs):

        src_t_gt_albedo = Variable(src_inputs['albedo']).cuda()
        src_t_gt_shading = Variable(src_inputs['shading']).cuda()
        src_t_gt_normal = Variable(src_inputs['normal']).cuda()
        src_t_gt_lighting = Variable(src_inputs['lighting']).cuda()
        src_t_gt_mask_eyeball = Variable(src_inputs['mask']).cuda()
        src_t_gt_mask_eyebrow = Variable(src_inputs['eyebrow']).cuda()
        src_t_gt_specular = Variable(src_inputs['specular']).cuda()

        src_t_in = torch.clamp(src_t_gt_albedo * src_t_gt_shading + src_t_gt_specular, 0, 1)
        
        tgt_t_gt_peripheral = Variable(tgt_inputs['peripheral']).cuda()
        tgt_t_gt_skin = Variable(tgt_inputs['skin']).cuda()
        tgt_t_gt_eyebrow = Variable(tgt_inputs['eyebrow']).cuda()
        tgt_t_in = Variable(tgt_inputs['composite']).cuda()

        self.optim.zero_grad()

        src_t_pred, feats = self.model(src_t_in, encode_feat=True)
        src_t_pred_albedo = src_t_pred[0]
        src_t_pred_shading = src_t_pred[1]
        src_t_pred_normal = graphic.out2normal(src_t_pred[2])
        src_t_pred_lighting = src_t_pred[3]
        src_t_pred_specular = src_t_pred[4]

        tgt_t_pred = self.model(tgt_t_in)
        tgt_t_pred_albedo = tgt_t_pred[0]
        tgt_t_pred_shading = tgt_t_pred[1]
        tgt_t_pred_normal = graphic.out2normal(tgt_t_pred[2])
        tgt_t_pred_lighting = tgt_t_pred[3]
        tgt_t_pred_specular = tgt_t_pred[4]
        
        loss_dict = self._get_asnl_loss(
            src_t_pred_albedo, src_t_pred_shading, src_t_pred_normal, src_t_pred_lighting, 
            src_t_gt_albedo, src_t_gt_shading, src_t_gt_normal, src_t_gt_lighting, 
            src_t_gt_mask_eyeball, src_t_gt_mask_eyebrow,
            feats,
            tgt_t_pred_albedo, tgt_t_pred_shading, tgt_t_pred_normal, tgt_t_pred_lighting, tgt_t_pred_specular, 
            tgt_t_in, tgt_t_gt_peripheral, tgt_t_gt_eyebrow, tgt_t_gt_skin
        )

        
        loss_total = loss_dict['total']

        loss_total.backward()
        if self.cfg.train.grad_clip is not None:
            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.cfg.train.grad_clip)

        self.optim.step()

        output_dict = {
            'src_input': src_t_in,

            'src_gt_albedo': src_t_gt_albedo,
            'src_gt_shading': src_t_gt_shading,
            'src_gt_normal': src_t_gt_normal,
            'src_gt_lighting': src_t_gt_lighting, 
            'src_gt_specular': src_t_gt_specular, 
            
            'src_pred_albedo': src_t_pred_albedo,
            'src_pred_shading': src_t_pred_shading,
            'src_pred_composite_sf': torch.clamp(src_t_pred_albedo * src_t_pred_shading, 0, 1),            
            'src_pred_composite': torch.clamp(src_t_pred_albedo * src_t_pred_shading + src_t_pred_specular, 0, 1),            
            'src_pred_normal': src_t_pred_normal,
            'src_pred_lighting': src_t_pred_lighting,
            'src_pred_specular': src_t_pred_specular,

            'src_name': 'train_src_' + os.path.basename(src_inputs['filename'][0]).split('.')[0],

            'tgt_input': tgt_t_in,

            'tgt_pred_albedo': tgt_t_pred_albedo,
            'tgt_pred_shading': tgt_t_pred_shading,
            'tgt_pred_composite_sf': torch.clamp(tgt_t_pred_albedo * tgt_t_pred_shading, 0, 1), 
            'tgt_pred_composite': torch.clamp(tgt_t_pred_albedo * tgt_t_pred_shading + tgt_t_pred_specular, 0, 1), 
            'tgt_pred_normal': tgt_t_pred_normal,
            'tgt_pred_lighting': tgt_t_pred_lighting,
            'tgt_pred_specular': tgt_t_pred_specular,

            'tgt_mask': tgt_t_gt_skin,

            'tgt_name': 'train_tgt_' + os.path.basename(tgt_inputs['filename'][0]).split('.')[0]
        }

        return {
            'loss_dict': loss_dict,
            'output_dict': output_dict
        }

    def _get_src_c_loss(
            self, 
            src_t_pred_albedo, src_t_pred_shading, src_t_pred_specular, 
            src_t_gt_albedo, src_t_gt_shading, src_t_gt_specular,
            feats
        ):
        loss_dict = {}
        loss_dict['single'] = {
            'src_loss_p_specular': self.loss_percep(src_t_pred_specular, src_t_gt_specular),
            'src_loss_mse_specular': self.loss_mse(src_t_pred_specular, src_t_gt_specular),
            'src_loss_reconstruct': self.loss_reconstruct(src_t_gt_albedo * src_t_gt_shading + src_t_gt_specular, src_t_pred_albedo * src_t_pred_shading + src_t_pred_specular),
            'src_loss_feats_suppress': self.loss_feats_suppress(feats['C'])
        }
        loss_dict['combined'] = {
            'src_specular': 
                self.cfg.loss.C_src_specular_p * loss_dict['single']['src_loss_p_specular'] + \
                self.cfg.loss.C_src_specular_mse * loss_dict['single']['src_loss_mse_specular'],
            'src_reconstruct':
                self.cfg.loss.C_src_reconstruct * loss_dict['single']['src_loss_reconstruct'],
            'src_feats_suppress':
                self.cfg.loss.C_src_feats_suppress * loss_dict['single']['src_loss_feats_suppress']

        }
        loss_dict['total'] = sum(loss_dict['combined'].values())

        return loss_dict
    
    def _get_tgt_c_loss(
            self,
            tgt_t_pred_albedo, tgt_t_pred_shading, tgt_t_pred_specular,
            tgt_t_in, tgt_t_gt_peripheral, tgt_t_gt_eyebrow, tgt_t_gt_skin            
            ):

        loss_dict = {}
        loss_dict['single'] = {
            'tgt_loss_reconstruct': self.loss_reconstruct(tgt_t_in, tgt_t_pred_albedo * tgt_t_pred_shading + tgt_t_pred_specular, mask=tgt_t_gt_skin),
            'tgt_loss_specular_chrome': self.loss_chrome(tgt_t_pred_specular, tgt_t_gt_skin - tgt_t_gt_eyebrow - tgt_t_gt_peripheral),
            'tgt_loss_suppress': self.loss_specular_aware_suppress(tgt_t_pred_specular, tgt_t_gt_peripheral)
        }
        loss_dict['combined'] = {
            'tgt_reconstruct': self.cfg.loss.C_tgt_reconstruct * loss_dict['single']['tgt_loss_reconstruct'],
            'tgt_spec': 
                self.cfg.loss.C_tgt_spec_chrome * loss_dict['single']['tgt_loss_specular_chrome'] + \
                self.cfg.loss.C_tgt_spec_aware_suppress * loss_dict['single']['tgt_loss_suppress'],
        }
        loss_dict['total'] = sum(loss_dict['combined'].values())

        return loss_dict
    
    def _get_c_loss(
            self, 
            src_t_pred_albedo, src_t_pred_shading, src_t_pred_specular, 
            src_t_gt_albedo, src_t_gt_shading, src_t_gt_specular,
            feats,
            tgt_t_pred_albedo, tgt_t_pred_shading, tgt_t_pred_specular,
            tgt_t_in, tgt_t_gt_peripheral, tgt_t_gt_eyebrow, tgt_t_gt_skin
        ):
        src_loss_dict = self._get_src_c_loss(
            src_t_pred_albedo, src_t_pred_shading, src_t_pred_specular, 
            src_t_gt_albedo, src_t_gt_shading, src_t_gt_specular,
            feats
        )

        tgt_loss_dict = self._get_tgt_c_loss(
            tgt_t_pred_albedo, tgt_t_pred_shading, tgt_t_pred_specular,
            tgt_t_in, tgt_t_gt_peripheral, tgt_t_gt_eyebrow, tgt_t_gt_skin            
        )

        loss_total = \
            self.cfg.loss.C_src * src_loss_dict['total'] + \
            self.cfg.loss.C_tgt * tgt_loss_dict['total']

        loss_dict = {
            'src': src_loss_dict,
            'tgt': tgt_loss_dict,
            'total': loss_total,
        }

        return loss_dict

    def update_C(self, src_inputs, tgt_inputs):
        src_t_gt_albedo = Variable(src_inputs['albedo']).cuda()
        src_t_gt_shading = Variable(src_inputs['shading']).cuda()
        src_t_gt_normal = Variable(src_inputs['normal']).cuda()
        src_t_gt_lighting = Variable(src_inputs['lighting']).cuda()
        src_t_gt_specular = Variable(src_inputs['specular']).cuda()

        src_t_in = torch.clamp(src_t_gt_albedo * src_t_gt_shading + src_t_gt_specular, 0, 1)

        tgt_t_gt_peripheral = Variable(tgt_inputs['peripheral']).cuda()
        tgt_t_gt_skin = Variable(tgt_inputs['skin']).cuda()
        tgt_t_gt_eyebrow = Variable(tgt_inputs['eyebrow']).cuda()
        tgt_t_in = Variable(tgt_inputs['composite']).cuda()

        self.optim.zero_grad()

        src_t_pred, feats = self.model(src_t_in, encode_feat=True)
        src_t_pred_albedo = src_t_pred[0]
        src_t_pred_shading = src_t_pred[1]
        src_t_pred_normal = graphic.out2normal(src_t_pred[2])
        src_t_pred_lighting = src_t_pred[3]
        src_t_pred_specular = src_t_pred[4]

        tgt_t_pred = self.model(tgt_t_in)
        tgt_t_pred_albedo = tgt_t_pred[0]
        tgt_t_pred_shading = tgt_t_pred[1]
        tgt_t_pred_normal = graphic.out2normal(tgt_t_pred[2])
        tgt_t_pred_lighting = tgt_t_pred[3]
        tgt_t_pred_specular = tgt_t_pred[4]
        
        loss_dict = self._get_c_loss(
            src_t_pred_albedo, src_t_pred_shading, src_t_pred_specular, 
            src_t_gt_albedo, src_t_gt_shading, src_t_gt_specular,
            feats,
            tgt_t_pred_albedo, tgt_t_pred_shading, tgt_t_pred_specular,
            tgt_t_in, tgt_t_gt_peripheral, tgt_t_gt_eyebrow, tgt_t_gt_skin
        )
        loss_total = loss_dict['total']
        loss_total.backward()
        if self.cfg.train.grad_clip is not None:
            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.cfg.train.grad_clip)
        self.optim.step()

        output_dict = {
            'src_input': src_t_in,

            'src_gt_albedo': src_t_gt_albedo,
            'src_gt_shading': src_t_gt_shading,
            'src_gt_normal': src_t_gt_normal,
            'src_gt_lighting': src_t_gt_lighting, 
            'src_gt_specular': src_t_gt_specular, 
            
            'src_pred_albedo': src_t_pred_albedo,
            'src_pred_shading': src_t_pred_shading,
            'src_pred_composite_sf': torch.clamp(src_t_pred_albedo * src_t_pred_shading, 0, 1),            
            'src_pred_composite': torch.clamp(src_t_pred_albedo * src_t_pred_shading + src_t_pred_specular, 0, 1),            
            'src_pred_normal': src_t_pred_normal,
            'src_pred_lighting': src_t_pred_lighting,
            'src_pred_specular': src_t_pred_specular,

            'src_name': 'train_src_' + os.path.basename(src_inputs['filename'][0]).split('.')[0],

            'tgt_input': tgt_t_in,

            'tgt_pred_albedo': tgt_t_pred_albedo,
            'tgt_pred_shading': tgt_t_pred_shading,
            'tgt_pred_composite_sf': torch.clamp(tgt_t_pred_albedo * tgt_t_pred_shading, 0, 1), 
            'tgt_pred_composite': torch.clamp(tgt_t_pred_albedo * tgt_t_pred_shading + tgt_t_pred_specular, 0, 1), 
            'tgt_pred_normal': tgt_t_pred_normal,
            'tgt_pred_lighting': tgt_t_pred_lighting,
            'tgt_pred_specular': tgt_t_pred_specular,

            'tgt_mask': tgt_t_gt_skin,

            'tgt_name': 'train_tgt_' + os.path.basename(tgt_inputs['filename'][0]).split('.')[0]
        }

        return {
            'loss_dict': loss_dict,
            'output_dict': output_dict
        }

    def infer(self, src_inputs=None, tgt_inputs=None, mode='eval', with_tgt_gt=True):
        output_dict = {}
        if src_inputs:
            src_t_gt_albedo = Variable(src_inputs['albedo']).cuda()
            src_t_gt_shading = Variable(src_inputs['shading']).cuda()
            src_t_gt_normal = Variable(src_inputs['normal']).cuda()
            src_t_gt_lighting = Variable(src_inputs['lighting']).cuda()
            src_t_gt_mask_eyeball = Variable(src_inputs['mask']).cuda()
            src_t_gt_mask_eyebrow = Variable(src_inputs['eyebrow']).cuda()
            src_t_gt_specular = Variable(src_inputs['specular']).cuda()
            src_t_in = torch.clamp(src_t_gt_albedo * src_t_gt_shading + src_t_gt_specular, 0, 1)

            src_t_pred, feats = self.model(src_t_in, encode_feat=True)
            
            src_t_pred_albedo = src_t_pred[0]
            src_t_pred_shading = src_t_pred[1]
            src_t_pred_normal = graphic.out2normal(src_t_pred[2])
            src_t_pred_lighting = src_t_pred[3]
            src_t_pred_specular = src_t_pred[4]

            src_nm_sphere = torch.tensor(graphic.render_sphere_nm(100, 1), dtype=torch.float32)
            src_b = src_t_pred_albedo.shape[0]
            src_nm_sphere = torch.tile(src_nm_sphere, (src_b, 1, 1, 1))
            src_t_sphere = graphic.lambSH_layer(src_nm_sphere.cuda(), src_t_pred_lighting.reshape([src_b, 9, 3]),
                                            am=torch.ones_like(src_nm_sphere).cuda())
            src_t_pseudo_shading = graphic.lambSH_layer(src_t_pred_normal, src_t_pred_lighting.reshape([src_b, 9, 3]), torch_style=True)
            src_t_gt_sphere = graphic.lambSH_layer(src_nm_sphere.cuda(), src_t_gt_lighting.reshape([src_b, 9, 3]),
                                                    am=torch.ones_like(src_nm_sphere).cuda())
            src_t_gt_pseudo_shading = graphic.lambSH_layer(src_t_gt_normal, src_t_gt_lighting.reshape([src_b, 9, 3]), torch_style=True)

            output_dict.update({
                'src_input': src_t_in,
                
                'src_gt_albedo': src_t_gt_albedo,
                'src_gt_shading': src_t_gt_shading,
                'src_gt_normal': src_t_gt_normal,
                'src_gt_lighting': src_t_gt_lighting, 
                'src_gt_specular': src_t_gt_specular, 
                
                'src_pred_albedo': src_t_pred_albedo,
                'src_pred_shading': src_t_pred_shading,
                'src_pred_composite_sf': torch.clamp(src_t_pred_albedo * src_t_pred_shading, 0, 1),            
                'src_pred_composite': torch.clamp(src_t_pred_albedo * src_t_pred_shading + src_t_pred_specular, 0, 1),            
                'src_pred_normal': src_t_pred_normal,
                'src_pred_lighting': src_t_pred_lighting,
                'src_pred_specular': src_t_pred_specular,

                'src_pred_sphere': src_t_sphere,
                'src_pred_shading_pseudo': src_t_pseudo_shading,
                'src_gt_sphere': src_t_gt_sphere,
                'src_gt_shading_pseudo': src_t_gt_pseudo_shading,

                'src_name': f'{mode}_src_' + os.path.basename(src_inputs['filename'][0]).split('.')[0],
            })

        if tgt_inputs:
            if with_tgt_gt:
                tgt_t_gt_peripheral = Variable(tgt_inputs['peripheral']).cuda()
                tgt_t_gt_skin = Variable(tgt_inputs['skin']).cuda()
                tgt_t_gt_eyebrow = Variable(tgt_inputs['eyebrow']).cuda()
            tgt_t_in = Variable(tgt_inputs['composite']).cuda()

            tgt_t_pred = self.model(tgt_t_in)
            
            tgt_t_pred_albedo = tgt_t_pred[0]
            tgt_t_pred_shading = tgt_t_pred[1]
            tgt_t_pred_normal = graphic.out2normal(tgt_t_pred[2])
            tgt_t_pred_lighting = tgt_t_pred[3]
            tgt_t_pred_specular = tgt_t_pred[4]

            tgt_nm_sphere = torch.tensor(graphic.render_sphere_nm(100, 1), dtype=torch.float32)
            tgt_b = tgt_t_pred_albedo.shape[0]
            tgt_nm_sphere = torch.tile(tgt_nm_sphere, (tgt_b, 1, 1, 1))
            tgt_t_sphere = graphic.lambSH_layer(tgt_nm_sphere.cuda(), tgt_t_pred_lighting.reshape([tgt_b, 9, 3]),
                                            am=torch.ones_like(tgt_nm_sphere).cuda())
            tgt_t_pseudo_shading = graphic.lambSH_layer(tgt_t_pred_normal, tgt_t_pred_lighting.reshape([tgt_b, 9, 3]), torch_style=True)


            output_dict.update({
                'tgt_input': tgt_t_in,

                'tgt_pred_albedo': tgt_t_pred_albedo,
                'tgt_pred_shading': tgt_t_pred_shading,
                'tgt_pred_composite_sf': torch.clamp(tgt_t_pred_albedo * tgt_t_pred_shading, 0, 1), 
                'tgt_pred_composite': torch.clamp(tgt_t_pred_albedo * tgt_t_pred_shading + tgt_t_pred_specular, 0, 1), 
                'tgt_pred_normal': tgt_t_pred_normal,
                'tgt_pred_lighting': tgt_t_pred_lighting,
                'tgt_pred_specular': tgt_t_pred_specular,

                'tgt_pred_sphere': tgt_t_sphere,
                'tgt_pred_shading_pseudo': tgt_t_pseudo_shading,

                
            })

            if with_tgt_gt:
                output_dict.update({
                    'tgt_mask': tgt_t_gt_skin,
                    'tgt_name': f'{mode}_tgt_' + os.path.basename(tgt_inputs['filename'][0]).split('.')[0]
                })

        ret = {
            'output_dict': output_dict
        }

        if mode == 'eval':

            loss_dict_asnl = self._get_asnl_loss(
                src_t_pred_albedo, src_t_pred_shading, src_t_pred_normal, src_t_pred_lighting, 
                src_t_gt_albedo, src_t_gt_shading, src_t_gt_normal, src_t_gt_lighting, 
                src_t_gt_mask_eyeball, src_t_gt_mask_eyebrow,
                feats,
                tgt_t_pred_albedo, tgt_t_pred_shading, tgt_t_pred_normal, tgt_t_pred_lighting, tgt_t_pred_specular, 
                tgt_t_in, tgt_t_gt_peripheral, tgt_t_gt_eyebrow, tgt_t_gt_skin
            )

            loss_dict_c = self._get_c_loss(
                src_t_pred_albedo, src_t_pred_shading, src_t_pred_specular, 
                src_t_gt_albedo, src_t_gt_shading, src_t_gt_specular,
                feats,
                tgt_t_pred_albedo, tgt_t_pred_shading, tgt_t_pred_specular,
                tgt_t_in, tgt_t_gt_peripheral, tgt_t_gt_eyebrow, tgt_t_gt_skin
            )

            loss_dict = {
                'asnl': loss_dict_asnl,
                'c': loss_dict_c
            }
            ret.update({
                'loss_dict': loss_dict,
            })

        return ret

    def save(self, iteration):
        torch.save({'model': self.model.state_dict()}, os.path.join(self.log_dir, f'model_da_{iteration:05d}.pt'))


    def save_image(self, info, iteration, mode, use_mask=False):
        if use_mask:
            if len(info['tgt_mask'].shape) > 3:  
                mask = np.transpose(info['tgt_mask'].detach().cpu().numpy()[0], axes=[1, 2, 0])
            else:
                mask = np.transpose(info['tgt_mask'].detach().cpu().numpy(), axes=[1, 2, 0])
        for k in info.keys():
            if ('name' in k) or ('loss' in k) or ('mask' in k):
                continue
            if 'lighting' in k:
                pass
            elif len(info[k].shape) > 3: 
                img = np.transpose(info[k].detach().cpu().numpy()[0], axes=[1, 2, 0])
            else:
                img = np.transpose(info[k].detach().cpu().numpy(), axes=[1, 2, 0])

            path = {
                'train': self.log_img_train,
                'eval': self.log_img_eval,
                'test': self.log_img_test,
            }[mode]

            path = os.path.join(path, str(iteration))
            os.makedirs(path, exist_ok=True)
            
            if 'lighting' in k:
                name = info['src_name'] if 'src' in k else info['tgt_name']
                np.save(os.path.join(path, f'{name}-{iteration:05d}-{k}.npy'), info[k].detach().cpu().numpy()[0].reshape([9, 3]))
                continue 

            if 'pred_normal' in k or 'gt_normal' in k:
                img = np.uint8(np.clip(img, 0, 1) * 255)
                if use_mask and ('tgt' in k):
                    img = np.uint8(img * mask)
                name = info['src_name'] if 'src' in k else info['tgt_name']
                cv2.imwrite(os.path.join(path, f'{name}-{iteration:05d}-{k}.png'), img[..., ::-1])
                continue

            if 'pred_sphere' in k:
                img = np.vectorize(color.lin2srgb)(img)
                img = np.uint8(np.clip(img, 0, 1) * 255)
                name = info['src_name'] if 'src' in k else info['tgt_name']
                cv2.imwrite(os.path.join(path, f'{name}-{iteration:05d}-{k}.png'), img[..., ::-1])
                continue

            img = np.vectorize(color.lin2srgb)(img) 
            img = np.uint8(np.clip(img, 0, 1) * 255)
            if use_mask and ('tgt' in k):
                img = np.uint8(img * mask)
            name = info['src_name'] if 'src' in k else info['tgt_name']
            cv2.imwrite(os.path.join(path, f'{name}-{iteration:05d}-{k}.png'), img[..., ::-1])

class Trainer_SR(Trainer):
    def __init__(self, cfg, cfg_da):
        self.cfg_da = cfg_da
        super(Trainer_SR, self).__init__(cfg)
        
    def _init_loss(self):
        self.loss_mask_mse = MaskMSELoss()
    
    def _init_model(self):
        self.model = MultiHeadGenerator(
            encoder_cfg=self.cfg.model.encoder,
            decoder_cfg_list=[
                self.cfg.model.decoder.spec_rem
            ]
        )
        self.da_invnet = DAInverseNet(
            encoder_cfg=self.cfg_da.model.encoder,
            decoder_cfg_list=[
                self.cfg_da.model.decoder.albedo, 
                self.cfg_da.model.decoder.shading, 
                self.cfg_da.model.decoder.normal, 
                self.cfg_da.model.decoder.lighting, 
            ],
            sep_decoder_cfg_list=[
                self.cfg_da.model.decoder.specular
            ]
        )
        self.load_weight_da(self.cfg.train.da_path)
        
    def load_weight_da(self, path):
        checkpoint_dict = torch.load(path)['model']
        if list(checkpoint_dict.keys())[0].startswith('module.'):
            checkpoint_dict = {k[7:]: v for k, v in checkpoint_dict.items()}

        self.da_invnet.load_state_dict(checkpoint_dict)
    
    def _get_loss(
            self,
            t_pred_albedo, t_gt_albedo,
            t_mask_skin_loss, t_mask_specular_loss
        ):
        loss_dict = {}
        loss_dict['single'] = {
            'loss_mse_skin': self.loss_mask_mse(t_pred_albedo * t_mask_skin_loss, t_gt_albedo * t_mask_skin_loss, t_mask_skin_loss),
            'loss_mse_specular': self.loss_mask_mse(t_pred_albedo * t_mask_specular_loss, t_gt_albedo * t_mask_specular_loss, t_mask_specular_loss)
        }

        loss_dict['combined'] = {
            'spe_rem': 
                self.cfg.loss.skin * loss_dict['single']['loss_mse_skin'] + \
                self.cfg.loss.specular * loss_dict['single']['loss_mse_specular'],
        }

        loss_dict['total'] = sum(loss_dict['combined'].values())
        return loss_dict

    def update(self, inputs):
        with torch.no_grad():
            info_dainvnet = self.infer_test_dainvnet(inputs)['output_dict']
            info_dainvnet_transferred = self.infer_test_dainvnet(inputs, spec_aug=True)['output_dict']

        t_gt_peripheral = Variable(inputs['peripheral']).cuda()
        t_gt_skin = Variable(inputs['skin']).cuda()

        t_specular_mask_src = Variable(inputs['specular-mask-src']).cuda()

        self.optim.zero_grad()

        t_input = info_dainvnet_transferred['pred_albedo']
        t_pred = self.model(t_input)
        t_pred_albedo = t_pred[0]

        t_gt_albedo = info_dainvnet['pred_albedo']
        threshold = 0.1 * t_specular_mask_src.max()

        t_spec_tgt = t_gt_skin * t_gt_peripheral * (info_dainvnet['pred_specular'] > threshold)
        t_spec_tgt = utils.mask_expand(t_spec_tgt, step=2)
        t_mask_skin_loss = t_gt_skin - t_spec_tgt

        t_mask_specular_loss = t_gt_peripheral * (t_specular_mask_src > threshold)
        t_mask_specular_loss = torch.clip(t_mask_specular_loss - t_spec_tgt, 0, 1)
        t_mask_specular_loss = utils.mask_expand(t_mask_specular_loss, step=2)


        loss_dict = self._get_loss(
            t_pred_albedo, t_gt_albedo,
            t_mask_skin_loss, t_mask_specular_loss
        )
        loss_total = loss_dict['total']


        loss_total.backward()
        if self.cfg.train.grad_clip is not None:
            nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.cfg.train.grad_clip)

        self.optim.step()

        output_dict = {
            'input': t_input,

            'gt_albedo': t_gt_albedo,
            'pred_albedo': t_pred_albedo,

            'mask': t_gt_skin,
            'mask_skin': t_mask_skin_loss,
            'mask_specular': t_mask_specular_loss,
            
            'name': 'train_' + os.path.basename(inputs['filename'][0]).split('.')[0],
        }

        return {
            'loss_dict': loss_dict,
            'output_dict': output_dict
        }


    def infer(self, inputs, mode='eval'):
        output_dict = {}

        with torch.no_grad():
            info_dainvnet = self.infer_test_dainvnet(inputs)['output_dict']
            info_dainvnet_transferred = self.infer_test_dainvnet(inputs, spec_aug=True)['output_dict']

        t_gt_peripheral = Variable(inputs['peripheral']).cuda()
        t_gt_skin = Variable(inputs['skin']).cuda()

        t_specular_mask_src = Variable(inputs['specular-mask-src']).cuda()

        t_input = info_dainvnet_transferred['pred_albedo']
        t_pred = self.model(t_input)
        t_pred_albedo = t_pred[0]

        t_gt_albedo = info_dainvnet['pred_albedo']

        threshold = 0.1 * t_specular_mask_src.max()

        t_spec_tgt = t_gt_skin * t_gt_peripheral * (info_dainvnet['pred_specular'] > threshold)
        t_spec_tgt = utils.mask_expand(t_spec_tgt, step=2)
        t_mask_skin_loss = t_gt_skin - t_spec_tgt

        t_mask_specular_loss = t_gt_peripheral * (t_specular_mask_src > threshold)
        t_mask_specular_loss = torch.clip(t_mask_specular_loss - t_spec_tgt, 0, 1)
        t_mask_specular_loss = utils.mask_expand(t_mask_specular_loss, step=2)

        output_dict.update({
            'input': t_input,

            'gt_albedo': t_gt_albedo,
            'pred_albedo': t_pred_albedo,

            'mask': t_gt_skin,
            'mask_skin': t_mask_skin_loss,
            'mask_specular': t_mask_specular_loss,
            
            'name': f'{mode}_' + os.path.basename(inputs['filename'][0]).split('.')[0],
        })

        ret = {
            'output_dict': output_dict
        }

        if mode == 'eval':
            loss_dict = self._get_loss(
                t_pred_albedo, t_gt_albedo,
                t_mask_skin_loss, t_mask_specular_loss
            )
            ret.update({
                'loss_dict': loss_dict,
            })
        
        return ret


    def infer_test_dainvnet(self, inputs, spec_aug=False):

        t_gt_skin = Variable(inputs['skin']).cuda()
        if not spec_aug:
            t_in = Variable(inputs['composite']).cuda()
        else:
            t_in = Variable(inputs['transferred']).cuda()

        t_pred = self.da_invnet(t_in)
        t_pred_albedo = t_pred[0]
        t_pred_shading = t_pred[1]
        t_pred_specular = t_pred[4]

        output_dict = {
            'input': t_in,

            'pred_albedo': t_pred_albedo,
            'pred_shading': t_pred_shading,
            'pred_composite_sf': torch.clamp(t_pred_albedo * t_pred_shading, 0, 1), 
            'pred_composite': torch.clamp(t_pred_albedo * t_pred_shading + t_pred_specular, 0, 1), 
            'pred_specular': t_pred_specular,

            'mask': t_gt_skin,

            'name': 'test_da_' + os.path.basename(inputs['filename'][0]).split('.')[0]
        }

        return {
            'output_dict': output_dict
        }
    
    def infer_test_srnet(self, t_input):
        
        t_pred = self.model(t_input)
        t_pred_albedo = t_pred[0]

        output_dict = {
            'pred_albedo': t_pred_albedo,            
        }

        return {
            'output_dict': output_dict
        }       

    def save(self, iteration):
        torch.save({'model': self.model.state_dict()}, os.path.join(self.log_dir, f'model_sr_{iteration:05d}.pt'))

    def save_image(self, info, iteration, mode, use_mask=False):
        if use_mask:
            if len(info['mask'].shape) > 3:
                mask = np.transpose(info['mask'].detach().cpu().numpy()[0], axes=[1, 2, 0])
            else:
                mask = np.transpose(info['mask'].detach().cpu().numpy(), axes=[1, 2, 0])
        for k in info.keys():
            if ('name' in k) or ('loss' in k):
                continue
            if 'lighting' in k:
                pass
            elif len(info[k].shape) > 3: 
                img = np.transpose(info[k].detach().cpu().numpy()[0], axes=[1, 2, 0])
            else:
                img = np.transpose(info[k].detach().cpu().numpy(), axes=[1, 2, 0])

            path = {
                'train': self.log_img_train,
                'eval': self.log_img_eval,
                'test': self.log_img_test,
            }[mode]
            
            path = os.path.join(path, str(iteration))
            os.makedirs(path, exist_ok=True)
            
            if 'lighting' in k:
                name = info['name']
                np.save(os.path.join(path, f'{name}-{iteration:05d}-{k}.npy'), info[k].detach().cpu().numpy()[0].reshape([9, 3]))
                continue 

            if 'pred_normal' in k or 'gt_normal' in k:
                img = np.uint8(np.clip(img, 0, 1) * 255)
                if use_mask:
                    img = np.uint8(img * mask)
                name = info['name']
                cv2.imwrite(os.path.join(path, f'{name}-{iteration:05d}-{k}.png'), img[..., ::-1])
                continue

            if 'pred_sphere' in k:
                img = np.vectorize(color.lin2srgb)(img) 
                img = np.uint8(np.clip(img, 0, 1) * 255)
                name = info['name']
                cv2.imwrite(os.path.join(path, f'{name}-{iteration:05d}-{k}.png'), img[..., ::-1])
                continue

            img = np.vectorize(color.lin2srgb)(img)
            img = np.uint8(np.clip(img, 0, 1) * 255)
            if use_mask:
                img = np.uint8(img * mask)
            name = info['name']

            cv2.imwrite(os.path.join(path, f'{name}-{iteration:05d}-{k}.png'), img[..., ::-1])