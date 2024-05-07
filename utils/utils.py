import torch

def mask_expand(mask, step=1):
    mask_copy = mask.clone()

    mask_copy[:, :, step:, :] += mask[:, :, :-step, :]
    mask_copy[:, :, :-step, :] += mask[:, :, step:, :]
    mask_copy[:, :, :, step:] += mask[:, :, :, :-step]
    mask_copy[:, :, :, :-step] += mask[:, :, :, step:]

    mask_copy[:, :, step:, step:] += mask[:, :, :-step, :-step]
    mask_copy[:, :, step:, :-step] += mask[:, :, :-step, step:]
    mask_copy[:, :, :-step, step:] += mask[:, :, step:, :-step]
    mask_copy[:, :, :-step, :-step] += mask[:, :, step:, step:]


    mask_copy = torch.clip(mask_copy, 0, 1)

    return mask_copy