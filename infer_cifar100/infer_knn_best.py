#import scipy.io as scio
import argparse
import os
root = os.getcwd()
import sys
sys.path.append(root)
from tools.lib import *
import matplotlib
matplotlib.use('agg')

from infer_cifar100.cifar100_dataset import ourFeatureSet

class knn:
    def __init__(self,args):
        self.path = {}
        self.path["root"] = os.getcwd()
        self.bs = args.batch_size
        if args.device == "cpu":
            self.device = args.device
        else:
            self.device = int(args.device)
        self.cifar100_mean = [0.507, 0.487, 0.441]
        self.cifar100_std = [0.267, 0.256, 0.276]
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop((32, 32), 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cifar100_mean, std=self.cifar100_std)
        ])
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cifar100_mean, std=self.cifar100_std)
        ])
        self.train_id = args.k_nearest
        self.num_classes = args.num_classes
        self.train_shuffle = True
        self.val_shuffle = False
        self.load_epoch = args.load_epoch
        self.k_nearest = args.k_nearest

        self.path["data_path"] = os.path.join(self.path["root"], args.data_path)
        self.path["result_path"] = os.path.join(self.path["root"], args.result_path)
        if not os.path.exists(self.path["result_path"]):
            os.mkdir(self.path["result_path"])

    def prepare_dataset(self):
        self.train_set = ourFeatureSet(self.path["data_path"] + "/train")
        self.val_set = ourFeatureSet(self.path["data_path"] + "/test")
        self.val_loader = torch.utils.data.DataLoader(self.val_set,batch_size=self.bs, shuffle=self.val_shuffle)
        self.val_feature_set = [0 for i in range(len(self.val_set))]
        self.train_feature_set = [0 for i in range(len(self.train_set))]

    def calculate(self):
        err = 0
        for i,(img,lbl) in enumerate(self.val_loader):
            val_feature = img.to(self.device)
            batch_use = img.size(0)
            nearest = torch.Tensor([[99999 for k in range(batch_use)] for j in range(self.k_nearest)]).to(self.device)
            near_label = torch.Tensor([[-1 for k in range(batch_use)] for j in range(self.k_nearest)]).to(self.device)

            for j in range(len(self.train_set)):
                train_fn = self.train_set.samples[j][0]
                train_feature = torch.load(train_fn).to(self.device)
                batch_distance = (val_feature - train_feature) * (val_feature - train_feature)
                batch_distance = torch.sum(batch_distance, dim=(3, 2, 1))
                batch_distance = batch_distance.sqrt()
                batch_label = torch.FloatTensor([j for k in range(batch_use)]).to(self.device)

                for k in range(self.k_nearest):
                    tmp_dist = torch.min(nearest[k], batch_distance)
                    tmp_lbl = batch_label.data
                    lbl1 = (nearest[k] == tmp_dist).float()
                    lbl2 = 1 - lbl1
                    batch_label = batch_label * lbl1 + near_label[k] * lbl2
                    near_label[k] = near_label[k] * lbl1 + tmp_lbl * lbl2
                    batch_distance = torch.max(nearest[k], batch_distance)
                    nearest[k].copy_(tmp_dist)

            for j in range(batch_use):
                score = torch.Tensor([0 for i in range(self.num_classes)]).to(self.device)
                near_idx = torch.Tensor([0 for i in range(self.k_nearest)]).to(self.device)
                for k in range(self.k_nearest):
                    near_idx[k] = self.train_set.samples[int(near_label[k][j])][1]
                    score[int(near_idx[k])] += (1 / nearest[k][j])

                max_score = torch.max(score)
                # max_num = torch.sum((max_score == score).float())
                if abs(score[lbl[j]].item() - max_score) < 1e-3:
                    er = 0
                else:
                    er = 1
                err += er
            print("batch: %d, avg_err: %.5f" %(i, err/((i+1)*self.bs)))
            result = err/((i+1)*self.bs)
        return result


    def work(self):
        result = self.calculate()
        f = open(self.path["result_path"] + 'result_%d.txt' %self.train_id, 'w')
        f.write("error: %.5f" % result)
        f.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="1", type=str)
    parser.add_argument("--batch-size", default= 1000, type= int)
    parser.add_argument("--num-classes", default=100, type=int)
    parser.add_argument("--load-epoch", default=0, type=int)
    parser.add_argument("--k-nearest", default=1, type=int)
    parser.add_argument("--dim", default=5, type=int, choices=[2,3,5])

    parser.add_argument("--result-path", default="resnet56_cifar100_infer/eval_knn_best/", type=str)
    parser.add_argument("--data-path", default="data/infer_cifar100/best_feature")

    args = parser.parse_args()
    args.result_path = "result_dim%d/" % args.dim + args.result_path
    args.data_path = args.data_path + "_dim%d" % args.dim

    seed = int(args.num_classes)
    seed_torch(seed)
    baseline = knn(args)
    baseline.prepare_dataset()
    baseline.work()

if __name__ == '__main__':
    main()



