import os
import re
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

root = os.getcwd()
src_path = root + '/data/CelebA'
dst_path = root + '/data/cropped_CelebA'
train_num = 162770

class ourDataset(Dataset):
    def __init__(self, category, dst_path='/data/cropped_CelebA', training=True):
        fn = open(src_path + '/Anno/' + category, 'r')
        fh2 = open(src_path + '/Eval/list_eval_partition.txt', 'r')
        imgs = []
        lbls = []
        ln = 0
        regex = re.compile('\s+')
        for line in fn:
            ln += 1
            if ln <= 2:
                continue
            if (ln - 2 <= train_num and training) or\
                (ln - 2 > train_num and not training):
                line = line.rstrip('\n')
                line_value = regex.split(line)
                imgs.append(line_value[0])
                lbls.append(list(int(i) if int(i) > 0 else 0 for i in line_value[1:]))
        self.imgs = imgs
        self.lbls = lbls
        self.is_train = training
        self.dst_path = root + dst_path
        if "best" in self.dst_path:
            num = len(self.imgs) // 3 * 3
            self.imgs = self.imgs[:num]
            self.lbls = self.lbls[:num]
        self.transform = transforms.Compose([
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                            ])

    def __getitem__(self, idx):
        fn = self.imgs[idx]
        lbls = self.lbls[idx]
        if self.is_train:
            imgs = Image.open(self.dst_path + '/train/' + fn)
        else:
            imgs = Image.open(self.dst_path + '/test/' + fn)
        imgs = self.transform(imgs)
        lbls = torch.Tensor(lbls)
        return [imgs, lbls]

    def __len__(self):
        return len(self.imgs)


class ourFeatureset(Dataset):
    def __init__(self, category, dst_path='/data/cropped_CelebA', training=True):
        fn = open(src_path + '/Anno/' + category, 'r')
        fh2 = open(src_path + '/Eval/list_eval_partition.txt', 'r')
        imgs = []
        lbls = []
        ln = 0
        regex = re.compile('\s+')
        for line in fn:
            ln += 1
            if ln <= 2:
                continue
            if (ln - 2 <= train_num and training) or\
                (ln - 2 > train_num and not training):
                line = line.rstrip('\n')
                line_value = regex.split(line)
                imgs.append(line_value[0])
                lbls.append(list(int(i) if int(i) > 0 else 0 for i in line_value[1:]))
        self.dst_path = root + dst_path
        self.imgs = imgs
        self.lbls = lbls
        if "best" in self.dst_path:
            num = len(self.imgs) // 3 * 3
            self.imgs = self.imgs[:num]
            self.lbls = self.lbls[:num]
        self.is_train = training

    def __getitem__(self, idx):
        fn = self.imgs[idx]
        fn = fn[:6] + '.bin'
        lbls = self.lbls[idx]
        if self.is_train:
            imgs = torch.load(self.dst_path + '/train/' + fn)
        else:
            imgs = torch.load(self.dst_path + '/test/' + fn)
        lbls = torch.Tensor(lbls)
        return [imgs, lbls]

    def __len__(self):
        return len(self.imgs)


def prepare_CelebA():

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
        os.makedirs(dst_path+'/train')
        os.makedirs(dst_path+'/test')
        for class_dir in os.listdir(src_path+'/Img/'):
            os.makedirs(dst_path+'/train/'+class_dir)
            os.makedirs(dst_path+'/test/'+class_dir)

    # with open(src_path+'Anno/images.txt') as images_f:
    #     image_path = images_f.readlines()
    #     image_path = list(map(lambda s: s.rstrip('\n').split(' ')[1], image_path))

    with open(src_path+'/Eval/list_eval_partition.txt') as train_test_split_f:
        is_test = train_test_split_f.readlines()
        image_path = list(map(lambda s: s.rstrip('\n').split(' ')[0],  is_test))
        is_test = list(map(lambda s: bool(int(s.rstrip('\n').split(' ')[1])), is_test))

    with open(src_path+'/Anno/list_bbox_celeba.txt') as bounding_boxes_f:
        bounding_boxes = bounding_boxes_f.readlines()
        regex = re.compile('\s+')
        bounding_boxes = list(map(lambda s: [float(ss) for ss in regex.split(s.rstrip('\n'))[1:]], bounding_boxes[2:]))

    for i in range(len(image_path)):
        print('%d / %d' % (i + 1, len(image_path)))
        with Image.open(src_path+'/Img/img_align_celeba/'+image_path[i]) as im:

            # crop and resize
            im = transforms.functional.resize(im, [224,224])
            # TODO: resize will change h:w ratio
            # im.show()

            if is_test[i] == 0:
                im.save(dst_path+'/train/'+image_path[i])
            else:
                im.save(dst_path+'/test/'+image_path[i])

if __name__ == "__main__":
    prepare_CelebA()
