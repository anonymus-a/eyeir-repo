import os
import sys
sys.path.append('.')
import torch
from tqdm import tqdm
import argparse

from torch.utils.data import DataLoader

from data.data import LowLevelVisionFolder

def test_src(model):
    if model == 'ASNL':
        from cfgs.asnl import cfg
        from trainer.trainer import Trainer_ASNL as Trainer
    elif model == 'C':
        from cfgs.c import cfg
        from trainer.trainer import Trainer_C as Trainer

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device_id
    torch.backends.cudnn.benchmark = True

    test_set_src = LowLevelVisionFolder(params=cfg.data.test.src, image_names=cfg.data.image_names, is_train=False)

    test_loader_src = DataLoader(test_set_src, batch_size=1, num_workers=cfg.data.test.num_workers, shuffle=False)

    trainer = Trainer(cfg).cuda()
    
    trainer.load_weight(cfg.test.model_path)

    global_iteration = int(os.path.splitext(os.path.basename(cfg.test.model_path))[0].split('_')[-1])

    ts_bar_src = tqdm(test_loader_src)
    ts_bar_src.set_description('Testing Src')
    trainer.model.eval()

    for test_data in ts_bar_src:
        with torch.no_grad():
            info = trainer.infer(inputs=test_data, mode='test')
            trainer.save_image(info['output_dict'], global_iteration, 'test')

    

def test_da():
    from cfgs.da import cfg
    from trainer.trainer import Trainer_DA as Trainer
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device_id
    torch.backends.cudnn.benchmark = True

    test_set_src = LowLevelVisionFolder(params=cfg.data.test.src, image_names=cfg.data.image_names_src, is_train=False)
    test_set_tgt = LowLevelVisionFolder(params=cfg.data.test.tgt, image_names=cfg.data.image_names_tgt, is_train=False)

    test_loader_src = DataLoader(test_set_src, batch_size=1, num_workers=cfg.data.test.num_workers, shuffle=False)
    test_loader_tgt = DataLoader(test_set_tgt, batch_size=1, num_workers=cfg.data.test.num_workers, shuffle=False)

    trainer = Trainer(cfg).cuda()
    
    trainer.load_weight(cfg.test.model_path)

    global_iteration = int(os.path.splitext(os.path.basename(cfg.test.model_path))[0].split('_')[-1])

    ts_bar_src = tqdm(test_loader_src)
    ts_bar_src.set_description('Testing Src')
    trainer.model.eval()

    for test_data in ts_bar_src:
        with torch.no_grad():
            info = trainer.infer(src_inputs=test_data, mode='test')
            trainer.save_image(info['output_dict'], global_iteration, 'test', use_mask=False)

    
    ts_bar_tgt = tqdm(test_loader_tgt)
    ts_bar_tgt.set_description('Testing Tgt')
    trainer.model.eval()

    for test_data in ts_bar_tgt:
        with torch.no_grad():
            info = trainer.infer(tgt_inputs=test_data, mode='test')
            trainer.save_image(info['output_dict'], global_iteration, 'test', use_mask=True)

def test_sr():
    from cfgs.da import cfg as cfg_da
    from cfgs.sr import cfg
    from trainer.trainer import Trainer_SR as Trainer
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device_id
    torch.backends.cudnn.benchmark = True

    test_set_tgt = LowLevelVisionFolder(params=cfg.data.test.tgt, image_names=cfg.data.image_names, is_train=False)

    test_loader_tgt = DataLoader(test_set_tgt, batch_size=1, num_workers=cfg.data.test.num_workers, shuffle=False)

    trainer = Trainer(cfg, cfg_da).cuda()
    
    trainer.load_weight(cfg.test.model_path)

    global_iteration = int(os.path.splitext(os.path.basename(cfg.test.model_path))[0].split('_')[-1])


    ts_bar_tgt = tqdm(test_loader_tgt)
    ts_bar_tgt.set_description('Testing Tgt')
    trainer.model.eval()

    for test_data in ts_bar_tgt:
        with torch.no_grad():
            info = trainer.infer(inputs=test_data, mode='test')
            trainer.save_image(info['output_dict'], global_iteration, 'test', use_mask=True)

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str)

args = parser.parse_args()

if __name__ == '__main__':

    if args.model == 'ASNL':
        test_src('ASNL')
    elif args.model == 'C':
        test_src('C')
    elif args.model == 'DA':
        test_da()
    elif args.model == 'SR':
        test_sr()
    else:
        raise NotImplementedError



