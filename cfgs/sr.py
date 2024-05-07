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
cfg.train.epochs_total = 5
cfg.train.epochs_save = 1
cfg.train.da_path = os.path.join(cfg.project_dir, 'logs', 'da', 'expr', 'model_da_29999.pt')

cfg.train.grad_clip = None


# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
cfg.test = CN()
cfg.test.num_workers = 4
cfg.test.model_path = os.path.join(cfg.project_dir, 'pretrained', 'v1', 'model_sr_00003.pt')
# ---------------------------------------------------------------------------- #
# Log
# ---------------------------------------------------------------------------- #
cfg.log = CN()
cfg.log.root = os.path.join(cfg.project_dir, 'logs', 'sr', cfg.expr)
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

cfg.model.decoder.spec_rem = CN()
cfg.model.decoder.spec_rem.output_dim = 3
cfg.model.decoder.spec_rem.intermediate_dim = 64
cfg.model.decoder.spec_rem.norm = 'bn'
cfg.model.decoder.spec_rem.activation = 'relu'
cfg.model.decoder.spec_rem.pad_type = 'reflect'
cfg.model.decoder.spec_rem.res_scale = 1
cfg.model.decoder.spec_rem.n_resblocks = 4
cfg.model.decoder.spec_rem.end_act = 'sigmoid'
cfg.model.decoder.spec_rem.se_reduction = True
cfg.model.decoder.spec_rem.pyramid = True
cfg.model.decoder.spec_rem.type = 'FC_attention_heavy'


# ---------------------------------------------------------------------------- #
# Data
# ---------------------------------------------------------------------------- #

cfg.data = CN()
cfg.data.image_names = ['composite', 'peripheral', 'skin', 'specular-mask-src', 'transferred']

cfg.data.train = CN()

cfg.data.train.batch_size = 2
cfg.data.train.num_workers = 8


cfg.data.train.data_root = os.path.join(cfg.project_dir, 'dataset', 'REIR', 'train')
cfg.data.train.image_height = 224
cfg.data.train.image_width = 224
cfg.data.train.num_limit = None

cfg.data.eval = CN()

cfg.data.eval.batch_size = 1
cfg.data.eval.num_workers = 4
cfg.data.eval.num_images = 50

cfg.data.eval.data_root = os.path.join(cfg.project_dir, 'dataset', 'REIR', 'test')
cfg.data.eval.image_height = 224
cfg.data.eval.image_width = 224
cfg.data.eval.num_limit = 50



cfg.data.test = CN()
cfg.data.test.num_workers = 4


cfg.data.test.tgt = CN()
cfg.data.test.tgt.data_root = os.path.join(cfg.project_dir, 'dataset', 'REIR', 'test')
cfg.data.test.tgt.image_height = 224
cfg.data.test.tgt.image_width = 224
cfg.data.test.tgt.num_limit = 50

# ---------------------------------------------------------------------------- #
# Loss
# ---------------------------------------------------------------------------- #

cfg.loss = CN()

cfg.loss.skin = 1
cfg.loss.specular = 1

