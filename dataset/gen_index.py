import os
import pickle

angle_list = [0, 120, 240]
intensity_list = [20, 60, 100]

def gen_index_s(root, category, multi, suffix='png', alias=None, with_num=False):
    index = []
    dirs = []
    if not alias:
        alias = category
    for dir in os.listdir(root):
        dir_path = os.path.join(root, dir)
        if os.path.isdir(dir_path):
            dirs.append(dir)
    for dir in sorted(dirs, key=lambda x: int(x)):
        dir_path = os.path.join(root, dir)
        if os.path.isdir(dir_path):
            if multi:
                for angle in angle_list:
                    for intensity in intensity_list:
                        file_path = os.path.join(dir_path, f'{dir}_{angle}_{intensity}_{alias}.{suffix}')
                        assert os.path.exists(file_path)
                        index.append(file_path)
                        print(f'Append: {file_path}')
            else:
                for i in range(len(angle_list) * len(intensity_list)):
                    if with_num:
                        file_path = os.path.join(dir_path, f'{dir}_{alias}-{i}.{suffix}')
                    else:
                        file_path = os.path.join(dir_path, f'{dir}_{alias}.{suffix}')
                    assert os.path.exists(file_path)
                    index.append(file_path)
                    print(f'Append: {file_path}')
    
    out_path = os.path.join(root, category)
    with open(out_path, 'w') as f:
        for line in index:
            f.write(line + '\n')

def gen_index_SEIR():
    cur_dir = os.path.dirname(__file__)
    for mode in ['train', 'test']:
        dir_path = os.path.join(cur_dir, 'SEIR', mode)

        gen_index_s(dir_path, 'composite', multi=True)
        gen_index_s(dir_path, 'albedo', multi=False)
        gen_index_s(dir_path, 'shading', multi=True)
        gen_index_s(dir_path, 'normal', multi=False)
        gen_index_s(dir_path, 'lighting', multi=True, alias='sh', suffix='npy')
        gen_index_s(dir_path, 'specular', multi=False, with_num=True)
        gen_index_s(dir_path, 'mask', multi=False)
        gen_index_s(dir_path, 'eyebrow', multi=False)

def gen_index_t(num_list, root, category, mode):

    img_paths = []

    if mode == 'train':
        num_list = num_list[:10000]
    elif mode == 'test':
        num_list = num_list[10000:]
    else:
        raise NotImplementedError
    
    for num in num_list:
        if category == 'composite':
            img_path = os.path.join(root, '{}.png'.format(num))
        elif category == 'specular-mask-src':
            img_path = os.path.join(root, '{}_spec_mask.png'.format(num))
        elif category == 'transferred':
            img_path = os.path.join(root, '{}_transferred.png'.format(num))
        else:
            img_path = os.path.join(root, '{}_{}.png'.format(num, category))
        assert os.path.exists(img_path)
        img_paths.append(img_path)
        print(f'Append: {img_path}')

    return img_paths


def gen_index_REIR():
    cur_dir = os.path.dirname(__file__)
    for mode in ['train', 'test']:
        in_dir = os.path.join(cur_dir, 'REIR', 'images')
        out_dir = os.path.join(cur_dir, 'REIR', mode)

        os.makedirs(out_dir, exist_ok=True)

        cats = ['composite', 'peripheral', 'skin', 'specular-mask-src', 'transferred', 'eyebrow']

        path_nums = os.path.join(in_dir, 'nums-1w.pk')
        with open(path_nums, 'rb') as f:
            nums = pickle.load(f)

        for cat in cats:
            out_path = os.path.join(out_dir, cat)
            paths = gen_index_t(nums, in_dir, cat, mode)
            with open(out_path, 'w') as f:
                for line in paths:
                    f.write(line + '\n')

if __name__ == "__main__":
    gen_index_SEIR()
    gen_index_REIR()


