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
cfg.test.model_path = os.path.join(cfg.project_dir, 'logs', 'c', 'expr', 'model_c_00059.pt')

# ---------------------------------------------------------------------------- #
# Log
# ---------------------------------------------------------------------------- #
cfg.log = CN()
cfg.log.root = os.path.join(cfg.project_dir, 'logs', 'c', cfg.expr)
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
cfg.model.decoder.specular.type = 'FC'


# ---------------------------------------------------------------------------- #
# Data
# ---------------------------------------------------------------------------- #

cfg.data = CN()
cfg.data.image_names = ['composite', 'albedo', 'shading', 'specular']

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

cfg.loss.specular_p = 1
cfg.loss.specular_mse = 1

