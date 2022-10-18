import os
import sys
import torch
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import numpy as np
from torchvision.datasets.cifar import CIFAR10

from PIL import Image
import numpy as np
from torchvision.datasets import DatasetFolder


class Coarse_Cifar100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

        This is a subclass of the `CIFAR10` Dataset.
        """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'coarse_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):

        super(Coarse_Cifar100, self).__init__(root,train=train, transform=transform, target_transform=target_transform, download=download) # revised by rqh! add more argument besides root
        self.transform = transform
        self.target_transform = target_transform

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'coarse_labels' in entry:
                    self.targets.extend(entry['coarse_labels'])
                elif 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()


# ---------------------------------------------------------------------------------
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
# --------------------------------------------------------------------------------------------


class ourFeatureSet(DatasetFolder):
    def __init__(self, r):
        super(ourFeatureSet, self).__init__(r, default_loader, extensions=[".bin"])

    def __getitem__(self, index):
        path, target = self.samples[index]
        feature = torch.load(path)
        return feature, target


if __name__ == "__main__":
    rt = os.getcwd()
    d = ourFeatureSet(rt + "/data/noloss_feature/train")

    print()
