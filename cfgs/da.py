from yacs.config import CfgNode as CN
import argparse
import yaml
import os

cfg = CN()

cfg.project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cfg.expr = 'expr'
cfg.cfg_path = os.path.abspath(__file__)

cfg.device_id = '0'


# ---------------------------------------------------------------------------- #
# Train
# ---------------------------------------------------------------------------- #
cfg.train = CN()
cfg.train.syn_asnl_path = os.path.join(cfg.project_dir, 'logs', 'asnl', 'expr', 'model_asnl_00059.pt')
cfg.train.syn_c_path = os.path.join(cfg.project_dir, 'logs', 'c', 'expr', 'model_c_00059.pt')

cfg.train.lr = 1e-4
cfg.train.betas = (0.9, 0.999)
cfg.train.iters_total = 30000
cfg.train.iters_asnl = 1300
cfg.train.iters_c = 700

cfg.train.grad_clip = 2


# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
cfg.test = CN()
cfg.test.num_workers = 4
cfg.test.model_path = os.path.join(cfg.project_dir, 'pretrained', 'model_da.pt')

# ---------------------------------------------------------------------------- #
# Log
# ---------------------------------------------------------------------------- #
cfg.log = CN()
cfg.log.root = os.path.join(cfg.project_dir, 'logs', 'da', cfg.expr)
cfg.log.img = os.path.join(cfg.log.root, 'images')
cfg.log.img_train = os.path.join(cfg.log.img, 'train')
cfg.log.img_eval = os.path.join(cfg.log.img, 'eval')
cfg.log.img_test = os.path.join(cfg.log.img, 'test')
cfg.log.interval = 10

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
cfg.model = CN()

cfg.model.encoder = CN()
cfg.model.encoder.name = 'resnet18'
cfg.model.encoder.pretrained = True

cfg.model.decoder = CN()

cfg.model.decoder.albedo = CN()
cfg.model.decoder.albedo.output_dim = 3
cfg.model.decoder.albedo.intermediate_dim = 64
cfg.model.decoder.albedo.norm = 'bn'
cfg.model.decoder.albedo.activation = 'relu'
cfg.model.decoder.albedo.pad_type = 'reflect'
cfg.model.decoder.albedo.res_scale = 1
cfg.model.decoder.albedo.n_resblocks = 3
cfg.model.decoder.albedo.end_act = 'sigmoid'
cfg.model.decoder.albedo.se_reduction = True
cfg.model.decoder.albedo.pyramid = True
cfg.model.decoder.albedo.type = 'FC_sep'

cfg.model.decoder.shading = CN()
cfg.model.decoder.shading.output_dim = 1
cfg.model.decoder.shading.intermediate_dim = 64
cfg.model.decoder.shading.norm = 'bn'
cfg.model.decoder.shading.activation = 'relu'
cfg.model.decoder.shading.pad_type = 'reflect'
cfg.model.decoder.shading.res_scale = 1
cfg.model.decoder.shading.n_resblocks = 3
cfg.model.decoder.shading.end_act = 'sigmoid'
cfg.model.decoder.shading.se_reduction = True
cfg.model.decoder.shading.pyramid = True
cfg.model.decoder.shading.type = 'FC_v2_sep'
cfg.model.decoder.shading.fc_input = 64

cfg.model.decoder.normal = CN()
cfg.model.decoder.normal.output_dim = 2
cfg.model.decoder.normal.intermediate_dim = 64
cfg.model.decoder.normal.norm = 'bn'
cfg.model.decoder.normal.activation = 'relu'
cfg.model.decoder.normal.pad_type = 'reflect'
cfg.model.decoder.normal.res_scale = 1
cfg.model.decoder.normal.n_resblocks = 3
cfg.model.decoder.normal.end_act = 'sigmoid'
cfg.model.decoder.normal.se_reduction = True
cfg.model.decoder.normal.pyramid = True
cfg.model.decoder.normal.type = 'FC_sep'

cfg.model.decoder.lighting = CN()
cfg.model.decoder.lighting.output_dim = 27
cfg.model.decoder.lighting.intermediate_dim = 128
cfg.model.decoder.lighting.norm = 'bn'
cfg.model.decoder.lighting.activation = 'relu'
cfg.model.decoder.lighting.pad_type = 'reflect'
cfg.model.decoder.lighting.type = 'FullConnected'

cfg.model.decoder.specular = CN()
cfg.model.decoder.specular.output_dim = 3
cfg.model.decoder.specular.intermediate_dim = 64
cfg.model.decoder.specular.norm = 'bn'
cfg.model.decoder.specular.activation = 'relu'
cfg.model.decoder.specular.pad_type = 'reflect'
cfg.model.decoder.specular.res_scale = 1
cfg.model.decoder.specular.n_resblocks = 3
cfg.model.decoder.specular.end_act = 'sigmoid'
cfg.model.decoder.specular.se_reduction = True
cfg.model.decoder.specular.pyramid = True
cfg.model.decoder.specular.type = 'FC_sep'

# ---------------------------------------------------------------------------- #
# Data
# ---------------------------------------------------------------------------- #

cfg.data = CN()
cfg.data.image_names_src = ['composite', 'albedo', 'shading', 'specular', 'normal', 'lighting', 'mask', 'eyebrow']
cfg.data.image_names_tgt = ['composite', 'peripheral', 'eyebrow', 'skin']

cfg.data.train = CN()

cfg.data.train.batch_size = 4
cfg.data.train.num_workers = 8

cfg.data.train.src = CN()
cfg.data.train.src.data_root = os.path.join(cfg.project_dir, 'dataset', 'SEIR', 'train')
cfg.data.train.src.image_height = 224
cfg.data.train.src.image_width = 224
cfg.data.train.src.num_limit = None

cfg.data.train.tgt = CN()
cfg.data.train.tgt.data_root = os.path.join(cfg.project_dir, 'dataset', 'REIR', 'train')
cfg.data.train.tgt.image_height = 224
cfg.data.train.tgt.image_width = 224
cfg.data.train.tgt.num_limit = None



cfg.data.eval = CN()

cfg.data.eval.batch_size = 1
cfg.data.eval.num_workers = 4
cfg.data.eval.num_images = 50

cfg.data.eval.src = CN()
cfg.data.eval.src.data_root = os.path.join(cfg.project_dir, 'dataset', 'SEIR', 'test')
cfg.data.eval.src.image_height = 224
cfg.data.eval.src.image_width = 224
cfg.data.eval.src.num_limit = 50

cfg.data.eval.tgt = CN()
cfg.data.eval.tgt.data_root = os.path.join(cfg.project_dir, 'dataset', 'REIR', 'test')
cfg.data.eval.tgt.image_height = 224
cfg.data.eval.tgt.image_width = 224
cfg.data.eval.tgt.num_limit = 50


cfg.data.test = CN()
cfg.data.test.num_workers = 4


cfg.data.test.src = CN()
cfg.data.test.src.data_root = os.path.join(cfg.project_dir, 'dataset', 'SEIR', 'test')
cfg.data.test.src.image_height = 224
cfg.data.test.src.image_width = 224
cfg.data.test.src.num_limit = 50

cfg.data.test.tgt = CN()
cfg.data.test.tgt.data_root = os.path.join(cfg.project_dir, 'dataset', 'REIR', 'test')
cfg.data.test.tgt.image_height = 224
cfg.data.test.tgt.image_width = 224
cfg.data.test.tgt.num_limit = 10

# ---------------------------------------------------------------------------- #
# Loss
# ---------------------------------------------------------------------------- #

cfg.loss = CN()

# ASNL-Net
cfg.loss.src = 1
cfg.loss.src_albedo_p = 10
cfg.loss.src_albedo_l = 1
cfg.loss.src_albedo_mse = 10
cfg.loss.src_albedo_mse_eyeball = 5
cfg.loss.src_albedo_gradient = 1
cfg.loss.src_albedo_smooth = 1

cfg.loss.src_shading_p = 100
cfg.loss.src_shading_l = 1
cfg.loss.src_shading_mse = 100
cfg.loss.src_shading_mse_eyeball = 100
cfg.loss.src_shading_mse_eyebrow = 100
cfg.loss.src_shading_gradient = 2
cfg.loss.src_shading_dark_eyeball = 2
cfg.loss.src_shading_dark_eyebrow = 2

cfg.loss.src_normal = 100
cfg.loss.src_normal_eyeball = 50
cfg.loss.src_normal_eyebrow = 50
cfg.loss.src_normal_smooth_eyeball = 100
cfg.loss.src_normal_smooth_eyebrow = 100

cfg.loss.src_lighting = 250

cfg.loss.src_render = 5
cfg.loss.src_render_eyeball = 2500
cfg.loss.src_render_eyebrow = 2500

cfg.loss.src_reconstruct = 2
cfg.loss.src_feats_suppress = 5


cfg.loss.tgt = 15
cfg.loss.tgt_reconstruct = 40

cfg.loss.tgt_render = 10
cfg.loss.tgt_render_eyeball = 20
cfg.loss.tgt_render_eyebrow = 5

cfg.loss.tgt_shading_dark_eyeball = 0.05
cfg.loss.tgt_shading_dark_eyebrow = 0.1

cfg.loss.tgt_normal_smooth_eyeball = 0.1
cfg.loss.tgt_normal_smooth_eyebrow = 0.1

cfg.loss.tgt_albedo_smooth = 0.5
cfg.loss.tgt_albedo_overexpo = 5



# C-Net
cfg.loss.C_src = 10
cfg.loss.C_src_specular_p = 1
cfg.loss.C_src_specular_mse = 1
cfg.loss.C_src_reconstruct = 1
cfg.loss.C_src_feats_suppress = 1

cfg.loss.C_tgt = 1
cfg.loss.C_tgt_reconstruct = 1
cfg.loss.C_tgt_spec_chrome = 1
cfg.loss.C_tgt_spec_aware_suppress = 0.1