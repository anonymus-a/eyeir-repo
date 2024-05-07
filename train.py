import os
import sys
sys.path.append('.')
import torch
from tqdm import tqdm
import argparse

from torch.utils.data import DataLoader

from data.data import LowLevelVisionFolder

from torch import autograd


def train(model):
    if model == 'ASNL':
        from cfgs.asnl import cfg
        from trainer.trainer import Trainer_ASNL as Trainer
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device_id
        trainer = Trainer(cfg).cuda()
    elif model == 'C':
        from cfgs.c import cfg
        from trainer.trainer import Trainer_C as Trainer
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device_id
        trainer = Trainer(cfg).cuda()
    elif model == 'SR':
        from cfgs.da import cfg as cfg_da
        from cfgs.sr import cfg
        from trainer.trainer import Trainer_SR as Trainer
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device_id
        trainer = Trainer(cfg, cfg_da).cuda()
    else:
        raise NotImplementedError

    
    torch.backends.cudnn.benchmark = True

    train_set = LowLevelVisionFolder(params=cfg.data.train, image_names=cfg.data.image_names, is_train=True)
    eval_set  = LowLevelVisionFolder(params=cfg.data.eval,  image_names=cfg.data.image_names, is_train=False)

    train_loader = DataLoader(train_set, batch_size=cfg.data.train.batch_size, num_workers=cfg.data.train.num_workers, shuffle=True)
    eval_loader  = DataLoader(eval_set,  batch_size=cfg.data.eval.batch_size,  num_workers=cfg.data.eval.num_workers, shuffle=False)

    

    start_epoch = 0
    n_global_iter = 0

    for epoch in range(start_epoch, cfg.train.epochs_total):
        tr_bar = tqdm(train_loader)
        trainer.model.train()

        for it, data in enumerate(tr_bar):
            # with autograd.detect_anomaly():
            info = trainer.update(data)
            n_global_iter += 1

            if n_global_iter % cfg.log.interval == 0:
                trainer.save_log(info['loss_dict'], n_global_iter, key='train')
                tr_bar.set_description(f'Training epoch-{epoch:03d}/{cfg.train.epochs_total}')

        trainer.save_image(info['output_dict'], epoch, mode='train')

        ev_bar = tqdm(eval_loader)
        ev_bar.set_description('Evaluating')

        trainer.model.eval()
        for eval_data in ev_bar:
            with torch.no_grad():
                info = trainer.infer(eval_data, mode='eval')


        trainer.save_image(info['output_dict'], epoch, mode='eval')
        trainer.save_log(info['loss_dict'], epoch, key='eval')

        if (epoch + 1) % cfg.train.epochs_save == 0:
            trainer.save(epoch)

    trainer.logger.close()    



def train_da():
    from cfgs.da import cfg
    from trainer.trainer import Trainer_DA as Trainer
        
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device_id
    torch.backends.cudnn.benchmark = True
    
    train_set_src = LowLevelVisionFolder(params=cfg.data.train.src, image_names=cfg.data.image_names_src, is_train=True)
    eval_set_src = LowLevelVisionFolder(params=cfg.data.eval.src, image_names=cfg.data.image_names_src, is_train=False)

    train_set_tgt = LowLevelVisionFolder(params=cfg.data.train.tgt, image_names=cfg.data.image_names_tgt, is_train=True)
    eval_set_tgt = LowLevelVisionFolder(params=cfg.data.eval.tgt, image_names=cfg.data.image_names_tgt, is_train=False)

    train_loader_src = DataLoader(train_set_src, batch_size=cfg.data.train.batch_size, num_workers=cfg.data.train.num_workers, shuffle=True)
    eval_loader_src = DataLoader(eval_set_src, batch_size=cfg.data.eval.batch_size, num_workers=cfg.data.eval.num_workers, shuffle=False)

    train_loader_tgt = DataLoader(train_set_tgt, batch_size=cfg.data.train.batch_size, num_workers=cfg.data.train.num_workers, shuffle=True)
    eval_loader_tgt = DataLoader(eval_set_tgt, batch_size=cfg.data.eval.batch_size, num_workers=cfg.data.eval.num_workers, shuffle=False)


    
    trainer = Trainer(cfg).cuda()

    start_global_iteration = 0

    len_train_src = len(train_loader_src)
    len_train_tgt = len(train_loader_tgt)
    len_test_src = len(eval_loader_src)
    len_test_tgt = len(eval_loader_tgt)

    flag = 'ASNL'
    trainer.unfix('ASNL')

    counter = cfg.train.iters_asnl


    for n_global_iter in tqdm(range(start_global_iteration, cfg.train.iters_total)):
        trainer.model.train()

        if n_global_iter % len_train_src == 0:
            train_iterator_src = iter(train_loader_src)
        if n_global_iter % len_train_tgt == 0:
            train_iterator_tgt = iter(train_loader_tgt)

        src_data = train_iterator_src.next()
        tgt_data = train_iterator_tgt.next()


        if flag == 'ASNL':
            info = trainer.update_ASNL(src_data, tgt_data)
        else:
            info = trainer.update_C(src_data, tgt_data)
           
        if n_global_iter % cfg.log.interval == 0:
            # logging for training
            s = trainer.save_log(info['loss_dict'], n_global_iter, key='train')

        if counter == 1:
            trainer.save_image(info['output_dict'], n_global_iter, mode='train', use_mask=True)
            trainer.model.eval()

            for i in tqdm(range(0, cfg.data.eval.num_images)):
                if i % len_test_src == 0:
                    test_iterator_src = iter(eval_loader_src)
                if i % len_test_tgt == 0:
                    test_iterator_tgt = iter(eval_loader_tgt)
                src_data = test_iterator_src.next()
                tgt_data = test_iterator_tgt.next()

                with torch.no_grad():
                    info = trainer.infer(src_data, tgt_data, mode='eval')

                trainer.save_image(info['output_dict'], n_global_iter, mode='eval', use_mask=True)

            
            trainer.save_log(info['loss_dict'], n_global_iter, key='eval')  # logging evaluation loss

            trainer.save(n_global_iter)

        counter -= 1
        if n_global_iter > 0 and counter == 0:
            if flag == 'ASNL':
                trainer.unfix('C')
                flag = 'C'
                counter = cfg.train.iters_c
            else:
                trainer.unfix('ASNL')
                flag = 'ASNL'
                counter = cfg.train.iters_asnl 

    trainer.logger.close()    


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str)

args = parser.parse_args()

if __name__ == '__main__':
    if args.model == 'DA':
        train_da()
    else:
        train(args.model)

