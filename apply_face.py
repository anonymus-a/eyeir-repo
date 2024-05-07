
import os
from collections import OrderedDict
import cv2
import numpy as np
import torch

from externals.hrnet import models
from externals.hrnet.config import (
    config,
    update_config,
)
from externals.hrnet.core.evaluation import decode_preds
from externals.hrnet.utils.transforms import crop

from utils import color
from skimage.transform import resize


device = 'cuda:0'


def init_lmk_detector():
    lib_path = os.path.join(os.path.dirname(__file__), 'externals/hrnet')
    coonfig_path = os.path.join(lib_path, 'experiments/300w/face_alignment_300w_hrnet_w18.yaml')
    update_config(config, coonfig_path)
    model = models.get_face_alignment_net(config)

    pretrained_path = os.path.join(lib_path, config.MODEL.PRETRAINED)
    # load model
    state_dict = torch.load(pretrained_path, map_location=device)

    # remove `module.` prefix from the pre-trained weights
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        name = key[7:]
        new_state_dict[name] = value

    # load weights without the prefix
    model.load_state_dict(new_state_dict)
    # run model on device
    model = model.to(device)
    model = model.eval()

    # init mean and std values for the landmark model's input
    mean = config.MODEL.MEAN
    mean = np.array(mean, dtype=np.float32)
    std = config.MODEL.STD
    std = np.array(std, dtype=np.float32)

    return model, mean, std

def crop_eye(frame, model, input_size, mean, std, label=None, pos=False):

    landmarks_img = frame.copy()
    height, width = frame.shape[:2]  

    resized = crop(
        frame,
        torch.Tensor([width / 2, height / 2]),
        4,
        tuple(input_size), 
    )

    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img = resized.astype(np.float32) / 255.0

    normalized_img = (img - mean) / std 


    with torch.no_grad():
        input = torch.Tensor(normalized_img.transpose([2, 0, 1]))
        input = input.to(device)
        output = model(input.unsqueeze(0))
        score_map = output.data.cpu()

        preds = decode_preds(
            score_map,
            [torch.Tensor([width / 2, height / 2])],
            [4],
            score_map.shape[2:4],
        )

        preds = preds.squeeze(0)
        landmarks = preds.data.cpu().detach().numpy()

    factor = 1.8

    ctr_shift = [0, 0]
    small = [landmarks[42][0], landmarks[43][1], landmarks[45][0], landmarks[46][1]]
    small_h_ctr = int((small[1] + small[3]) / 2) + ctr_shift[0]
    small_h_delta = int((small[3] - small[1]) * factor / 2)
    small_w_ctr = int((small[0] + small[2]) / 2) + ctr_shift[1]
    small_w_delta = int((small[2] - small[0]) * factor / 2)
    big = [landmarks[22][0], landmarks[24][1], landmarks[26][0], landmarks[46][1]]
    big_h_ctr = int((big[1] + big[3]) / 2)
    big_h_delta = int((big[3] - big[1]) * factor / 2)
    big_w_ctr = int((big[0] + big[2]) / 2)
    big_w_delta = int((big[2] - big[0]) * factor / 2)

    pic = {
        'small': landmarks_img[
                 max(small_h_ctr - small_w_delta, 0): min(small_h_ctr + small_w_delta, landmarks_img.shape[0]),
                 max(small_w_ctr - small_w_delta, 0): min(small_w_ctr + small_w_delta, landmarks_img.shape[1])
                 ],
        'big': landmarks_img[
               max(big_h_ctr - big_w_delta, 0): min(big_h_ctr + big_w_delta, landmarks_img.shape[0]),
               max(big_w_ctr - big_w_delta, 0): min(big_w_ctr + big_w_delta, landmarks_img.shape[1])
               ]
    }

    hmin = max(small_h_ctr - small_w_delta, 0)
    hmax = min(small_h_ctr + small_w_delta, landmarks_img.shape[0])
    wmin = max(small_w_ctr - small_w_delta, 0)
    wmax = min(small_w_ctr + small_w_delta, landmarks_img.shape[1])

    if label:
        label = {
            'small': label[
                     max(small_h_ctr - small_w_delta, 0): min(small_h_ctr + small_w_delta, landmarks_img.shape[0]),
                     max(small_w_ctr - small_w_delta, 0): min(small_w_ctr + small_w_delta, landmarks_img.shape[1])
                     ],
            'big': label[
                   max(big_h_ctr - big_w_delta, 0): min(big_h_ctr + big_w_delta, landmarks_img.shape[0]),
                   max(big_w_ctr - big_w_delta, 0): min(big_w_ctr + big_w_delta, landmarks_img.shape[1])
                   ]
        }

        if not pos:
            return pic, label
        else:
            return pic, label, (hmin, hmax, wmin, wmax)
    else:
        if not pos:
            return pic
        else:
            return pic, (hmin, hmax, wmin, wmax)


def load_image(img, height, width):
    image = {}
    image['composite'] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image['composite'] = image['composite'].astype(np.float32)
    image['composite'] /= 255.

    image['composite'] = np.vectorize(color.srgb2lin)(image['composite'])

    inter_type = 1
    image['composite'] = resize(image['composite'], (height, width), order=inter_type,
                       preserve_range=True)  
    image['composite'] = torch.from_numpy(np.transpose(image['composite'], (2, 0, 1))).contiguous().float()
    image['composite'] = torch.unsqueeze(image['composite'], dim=0)


    return image


if __name__ == "__main__":
    from cfgs.da import cfg as cfg_da
    from cfgs.sr import cfg as cfg_sr
    from trainer.trainer import Trainer_DA
    from trainer.trainer import Trainer_SR

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg_da.device_id

    trainer_da = Trainer_DA(cfg_da).cuda()
    trainer_da.load_weight(cfg_da.test.model_path)
    trainer_da.model.eval()
    
    trainer_sr = Trainer_SR(cfg_sr, cfg_da).cuda()
    trainer_sr.load_weight(cfg_sr.test.model_path)
    trainer_sr.model.eval()


    img_dir = os.path.join(cfg_da.project_dir, 'example_faces')

    out_dir = os.path.join(img_dir, 'results')

    os.makedirs(out_dir, exist_ok=True)
    

    lmk_detector, mean, std = init_lmk_detector()


    for num, fileName in enumerate(os.listdir(img_dir)):
        print(fileName)
        name = os.path.splitext(fileName)[0]

        path_img = os.path.join(img_dir, fileName)
        if not path_img[-4:] in ['.png', '.jpg']:
            continue

        img = cv2.imread(path_img)

        eye_img, pos = crop_eye(img, lmk_detector, config.MODEL.IMAGE_SIZE, mean, std, pos=True)

        orig_h = pos[1] - pos[0]
        orig_w = pos[3] - pos[2]

        inputs = load_image(eye_img['small'], 224, 224)
        info = trainer_da.infer(tgt_inputs=inputs, mode='test', with_tgt_gt=False)['output_dict']
        info['tgt_pred_albedo_sr'] = trainer_sr.infer_test_srnet(info['tgt_pred_albedo'])['output_dict']['pred_albedo']

        for component in ['sphere']:
            sphere = np.transpose(info['tgt_pred_sphere'].detach().cpu().numpy()[0], axes=[1, 2, 0])
            sphere = np.vectorize(color.lin2srgb)(sphere)
            sphere = np.uint8(np.clip(sphere, 0, 1) * 255)

            cv2.imwrite(os.path.join(out_dir, '{}_{}_{}.png'.format(num, name, component)), sphere[..., ::-1])

        for component in ['tgt_input', 'tgt_pred_albedo', 'tgt_pred_albedo_sr', 'tgt_pred_shading', 'tgt_pred_normal', 'tgt_pred_specular', 'tgt_pred_shading_pseudo', 'tgt_pred_composite']:

            output = info[component]
            output = np.transpose(output.detach().cpu().numpy()[0], axes=[1, 2, 0])

            if 'normal' in component: 
                pass
            else:
                output = np.vectorize(color.lin2srgb)(output)

            tag = np.uint8(np.clip(output, 0, 1) * 255)
            cv2.imwrite(os.path.join(out_dir, '{}_{}_{}_tag.png'.format(num, name, component)), tag[..., ::-1])
            inter_type = 1
            tag = resize(tag, (orig_h, orig_w), order=inter_type,
                    preserve_range=True)
            tag = tag[..., ::-1]
            tagged_img = img.copy()
            tagged_img[pos[0]:pos[1], pos[2]:pos[3], :] = tag

            cv2.imwrite(os.path.join(out_dir, '{}_{}_{}.png'.format(num, name, component)), tagged_img)
            print('save: ', name, component)

