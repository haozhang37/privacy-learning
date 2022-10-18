#import scipy.io as scio
import os
root = os.getcwd()
import sys
sys.path.append(root)
import argparse
import matplotlib
matplotlib.use('agg')

from tools.lib import *

from res_model.feature_resnet import feature_resnet56
from infer_cifar100.cifar100_dataset import ourFeatureSet

class eval_baseline_2:
    def __init__(self,args):
        self.path = {}
        self.path["root"] = os.getcwd()
        self.num_epoch = args.num_epoch
        self.bs = args.batch_size
        self.init_lr = args.lr
        self.weight_decay = args.weight_decay
        self.momentum = args.momentum
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
        self.train_id = args.train_id
        self.num_classes = args.num_classes
        self.train_shuffle = True
        self.val_shuffle = False
        self.load_epoch = args.load_epoch
        self.dim = args.dim

        self.path["data_path"] = os.path.join(self.path["root"], args.data_path)
        self.path["result_path"] = os.path.join(self.path["root"], args.result_path)
        if not os.path.exists(self.path["result_path"]):
            os.mkdir(self.path["result_path"])

    def prepare_model(self,optimizer, loss_fn):
        self.model = feature_resnet56(num_classes=self.num_classes)
        self.optimizer = optimizer([{"params":self.model.parameters(),'lr':self.init_lr,'initial_lr':self.init_lr}]
                                    ,weight_decay=self.weight_decay,momentum=self.momentum)
        self.loss_fn = loss_fn(reduction="mean").to(self.device)
        self.list = [[] for _ in range(4)] #result list [tran_loss, train_err, val_loss, val_err]

        if self.load_epoch!=0:
            self.load_latest_epoch()

    def prepare_dataset(self):
        train_path = str(self.path["data_path"]+'/train')
        test_path = str(self.path["data_path"] + '/test')
        self.train_set = ourFeatureSet(train_path)
        self.train_loader = torch.utils.data.DataLoader(self.train_set,batch_size=self.bs, shuffle=self.train_shuffle)

        self.val_set = ourFeatureSet(test_path)
        self.val_loader = torch.utils.data.DataLoader(self.val_set,batch_size=self.bs, shuffle=self.val_shuffle)


    def save_latest_epoch(self,epoch):
        model_path = self.path["result_path"]+ "resnet56_" + str(epoch)+".bin"
        list_path = self.path["result_path"] + "list_" + str(epoch)+".bin"
        torch.save(self.model.state_dict(),model_path)
        torch.save(self.list,list_path)

    def load_latest_epoch(self):
        model_path = self.path["result_path"] + "resnet56_" + str(self.load_epoch)+".bin"
        list_path = self.path["result_path"] + "list_" + str(self.load_epoch)+".bin"
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.list = torch.load(list_path, map_location=torch.device("cpu"))

    def get_error(self, pre, lbl):
        acc = torch.sum(pre == lbl).item() / len(pre)
        return 1 - acc

    def train_model(self):
        self.model.train()
        self.model.to(self.device)
        train_loss, train_err = 0, 0

        for i,(img,lbl) in enumerate(self.train_loader):
            img = img.to(self.device)
            lbl = lbl.to(self.device)
            batch_use = img.size(0)
            batch_use = batch_use // self.dim * self.dim
            if batch_use == 0:
                continue
            img = img[:batch_use]
            lbl = lbl[:batch_use]

            model_output = self.model(img)

            loss = self.loss_fn(model_output, lbl)
            predict = model_output.detach().max(1)[1]
            error = self.get_error(predict,lbl)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.cpu().item()
            train_err += error

            print("[train] batch: %d, loss: %f, err: %f" %(i, train_loss/(i+1), train_err/(i+1)))
        return train_loss/(i+1), train_err/(i+1)

    def val_model(self):
        self.model.eval()
        self.model.to(self.device)
        val_loss, val_err, val_loss_noloss, val_err_noloss, val_loss_best, val_err_best = 0, 0, 0, 0, 0, 0

        with torch.no_grad():
            for i,(img,lbl) in enumerate(self.val_loader):
                img = img.to(self.device)
                lbl = lbl.to(self.device)
                batch_use = img.size(0)
                batch_use = batch_use // self.dim * self.dim
                if batch_use == 0:
                    continue
                img = img[:batch_use]
                lbl = lbl[:batch_use]

                model_output = self.model(img)

                loss = self.loss_fn(model_output, lbl)
                predict = model_output.detach().max(1)[1]
                error = self.get_error(predict,lbl)

                val_loss += loss.cpu().item()
                val_err += error
                print("[val] batch: %d, loss: %f, err: %f" % (i, val_loss / (i + 1), val_err / (i + 1)))
            val_loss = val_loss/(i+1)
            val_err = val_err/(i+1)

            return val_loss,val_err

    def draw_fig(self):
        x = np.arange(0,len(self.list[0]),1)
        y1 = np.array(self.list[0])
        y2 = np.array(self.list[1])
        y3 = np.array(self.list[2])
        y4 = np.array(self.list[3])


        plt.figure(str(self.train_id))
        plt.subplot(211)
        plt.plot(x, y1, color='blue')
        plt.plot(x, y3, color='red')
        plt.title('loss')
        plt.subplot(212)
        plt.plot(x, y2, color='blue')
        plt.plot(x, y4, color='red')
        plt.title('error')
        plt.savefig(self.path["result_path"] + 'baseline_' + str(self.train_id) + ".png")

    def adjust_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def work(self):
        for epoch in range(self.load_epoch, self.num_epoch):
            if epoch>= 150:
                self.adjust_lr(0.01 * self.init_lr)
            elif epoch>=100:
                self.adjust_lr(0.1*self.init_lr)
            train_loss, train_err = self.train_model()
            self.list[0].append(train_loss)
            self.list[1].append(train_err)

            val_loss,val_err = self.val_model()
            self.list[2].append(val_loss)
            self.list[3].append(val_err)

            if epoch % 10 == 9 or epoch == self.num_epoch-1:
                self.save_latest_epoch(epoch)
            self.draw_fig()
        # self.val_model()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="1", type=str)
    # parser.add_argument("--batch-size", default = 100, type= int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--num-epoch", default=200, type=int)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--momentum", default=0.9,type=float)
    parser.add_argument("--train-id",default=3,type=int)
    parser.add_argument("--num-classes", default=100, type=int)
    parser.add_argument("--load-epoch", default=0, type=int)

    parser.add_argument("--result-path", default="resnet56_cifar100_infer/eval_feature_best_sample/", type=str)
    parser.add_argument("--data-path", default="data/infer_cifar100/best_feature")
    parser.add_argument("--dim", default=5, type=int, choices=[2,3,5])

    args = parser.parse_args()
    args.batch_size = 100 // (2*args.dim) * (2*args.dim) # set batch size close to 100
    args.result_path = "result_dim%d/" % args.dim + args.result_path
    args.data_path = args.data_path + "_dim%d" % args.dim

    seed = int(args.num_classes + args.num_epoch)
    seed_torch(seed)
    baseline = eval_baseline_2(args)
    baseline.prepare_model(optimizer=torch.optim.SGD,loss_fn=torch.nn.CrossEntropyLoss)
    baseline.prepare_dataset()
    baseline.work()

if __name__ == '__main__':
    main()


