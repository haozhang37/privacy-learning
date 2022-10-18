import torch
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dim", default=5, type=int, choices=[2,3,5])
parser.add_argument("--result-path", default="resnet56_cifar100_infer", type=str,
                    choices=["resnet56_cifar100_infer", "Alexnet_CelebA_infer"])

parser.add_argument("--type", default="inf_atk1", type=str,
                    choices=["inf_atk1","inf_atk2","inf_atk3"])
args = parser.parse_args()

result_path = "result_dim%d/" % args.dim + args.result_path
# print(args.result_path)

if args.result_path == "resnet56_cifar100_infer":
    if args.type == "inf_atk1":
        file_name = os.path.join(result_path, "eval_baseline_1/list_199.bin")
    elif args.type == "inf_atk2":
        file_name = os.path.join(result_path, "eval_best_sample/list_199.bin")
    elif args.type == "inf_atk3":
        file_name = os.path.join(result_path, "eval_feature_best_sample/list_199.bin")
elif args.result_path == "Alexnet_CelebA_infer":
    if args.type == "inf_atk1":
        file_name = os.path.join(result_path, "eval_baseline_1/3/list_9.bin")
    elif args.type == "inf_atk2":
        file_name = os.path.join(result_path, "eval_best_sample/3/list_9.bin")
    elif args.type == "inf_atk3":
        file_name = os.path.join(result_path, "eval_feature_best_sample/3/list_9.bin")

info = torch.load(file_name, map_location="cpu")
# print(len(info))
print("result_path: ", args.result_path)
print("inference attak %d" % int(args.type[-1]))
print("reconstruction error: ", info[-1][-1])


