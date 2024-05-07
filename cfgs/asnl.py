from yacs.config import CfgNode as CN
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

cfg.train.lr = 1e-3
cfg.train.betas = (0.5, 0.999)
cfg.train.epochs_total = 60
cfg.train.epochs_save = 5

cfg.train.grad_clip = None


# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
cfg.test = CN()
cfg.test.num_workers = 4
cfg.test.model_path = os.path.join(cfg.project_dir, 'logs', 'asnl', 'expr', 'model_asnl_00059.pt')

# ---------------------------------------------------------------------------- #
# Log
# ---------------------------------------------------------------------------- #
cfg.log = CN()
cfg.log.root = os.path.join(cfg.project_dir, 'logs', 'asnl', cfg.expr)
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
cfg.model.decoder.albedo.type = 'FC'

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
cfg.model.decoder.shading.type = 'FC_v2'
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
cfg.model.decoder.normal.type = 'FC'

cfg.model.decoder.lighting = CN()
cfg.model.decoder.lighting.output_dim = 27
cfg.model.decoder.lighting.intermediate_dim = 128
cfg.model.decoder.lighting.norm = 'bn'
cfg.model.decoder.lighting.activation = 'relu'
cfg.model.decoder.lighting.pad_type = 'reflect'
cfg.model.decoder.lighting.type = 'FullConnected'


# ---------------------------------------------------------------------------- #
# Data
# ---------------------------------------------------------------------------- #

cfg.data = CN()
cfg.data.image_names = ['composite', 'albedo', 'shading', 'normal', 'lighting', 'specular']

cfg.data.train = CN()

cfg.data.train.batch_size = 16
cfg.data.train.num_workers = 8


cfg.data.train.data_root = os.path.join(cfg.project_dir, 'dataset', 'SEIR', 'train')
cfg.data.train.image_height = 224
cfg.data.train.image_width = 224
cfg.data.train.num_limit = None

cfg.data.eval = CN()

cfg.data.eval.batch_size = 1
cfg.data.eval.num_workers = 4
cfg.data.eval.num_images = 50

cfg.data.eval.data_root = os.path.join(cfg.project_dir, 'dataset', 'SEIR', 'test')
cfg.data.eval.image_height = 224
cfg.data.eval.image_width = 224
cfg.data.eval.num_limit = 50



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
cfg.data.test.tgt.num_limit = 50

# ---------------------------------------------------------------------------- #
# Loss
# ---------------------------------------------------------------------------- #

cfg.loss = CN()

cfg.loss.albedo_p = 1
cfg.loss.albedo_l = 1
cfg.loss.albedo_mse = 10
cfg.loss.albedo_gradient = 1

cfg.loss.shading_p = 1
cfg.loss.shading_l = 1
cfg.loss.shading_mse = 10
cfg.loss.shading_gradient = 1


cfg.loss.normal = 10

cfg.loss.lighting = 5

cfg.loss.render = 5

cfg.loss.reconstruct = 10

