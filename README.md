<br />
<div align="center">

  <h3 align="center"></h3>

</div>

# EyeIR: Single Eye Image Inverse Rendering In the Wild
This is the implementation of our paper **EyeIR: Single Eye Image Inverse Rendering In the Wild**

## Dataset
The complete dataset can be downloaded from the [Goole Drive](https://drive.google.com/file/d/1bhnNWvKgZlDOB9-1qMNp4pOqnqq7u8nb/view?usp=drive_link), which contains the SEIR (Synthetic Eye Inverse Rendering) dataset that we use in the paper. We also provide a small subset of our dataset for preview, which can be downloaded from [Preview](https://drive.google.com/file/d/1Xg_kbznof-XZQXcymcxXdwBcSmSfeEar/view?usp=drive_link). Note that we build REIR by cropping eye region images from [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ), using facial landmark detection model 
[HRNet](https://github.com/HRNet/HRNet-Facial-Landmark-Detection).

## Test for Faces
We provide a script to crop the input face image to eye region image, and then perform EyeIR on it.

1. Download the [pretrained models](https://drive.google.com/drive/folders/1wyu_ox7gAwKLUXZoA5RAxu5CqXBb8fKI?usp=drive_link), and move them to `pretrained/`

2. Set the model path for testing in `cfgs/da.py` and `cfgs/sr.py`
```sh
# cfgs/da.py
cfg.test.model_path = os.path.join(cfg.project_dir, 'pretrained', 'model_da.pt')
# cfgs/sr.py
cfg.test.model_path = os.path.join(cfg.project_dir, 'pretrained', 'model_sr.pt')
```

3. Place the face images in  `example_faces/`, and run:

```sh
python apply_face.py
```

The results will be shown in `example_faces/results/`

## Train
1. Unzip the downloaded dataset, move `SEIR` and `REIR` to `dataset/`. Run this command to generate the index file for training:

```sh
python dataset/gen_index.py
```

2. Train ASNL-Net and C-Net:
```sh
python train.py --model ASNL
python train.py --model C
```

3. Domain adaptation training on both SEIR and REIR. Set the ASNL-Net and C-Net pretrained models' paths in `cfgs/da.py`
```sh
cfg.train.syn_asnl_path = ASNL_PRETRAINED_PATH # choose a trained model in logs/asnl/expr, e.g., os.path.join(cfg.project_dir, 'logs', 'asnl', 'expr', 'model_asnl_00059.pt')
cfg.train.syn_c_path = C_PRETRAINED_PATH # choose a trained model in logs/c/expr, e.g., os.path.join(cfg.project_dir, 'logs', 'c', 'expr', 'model_c_00059.pt')
```
Then, run:
```sh
python train.py --model DA
```
4. Train SpecRem-Net. Set the ASNL-Net and C-Net pretrained models' paths in `cfgs/da.py`:
```sh
cfg.train.da_path = DA_PRETRAINED_PATH # choose a trained model in logs/da/expr, e.g., os.path.join(cfg.project_dir, 'logs', 'da', 'expr', 'model_da_29999.pt')
```
Then, run:
```sh
python train.py --model SR
```
