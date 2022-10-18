This is code of paper Complex-Valued Neural Network [1], Quaternion Neural Network [2], and their further general extension, RENN [3].

[1] **Interpretable Complex-Valued Neural Networks for Privacy Protection**. Liyao Xiang\*, Hao Zhang\*, Haotian Ma, Yifan Zhang, Jie Ren, and Quanshi Zhang, in ICLR 2020.

[2] **Deep Quaternion Features for Privacy Protection**. Hao Zhang, Yiting Chen, Liyao Xiang, Haotian Ma, Jie Shi, and Quanshi Zhang. In arXiv:2003.08365, 2020.

[3] **Rotation-Equivariant Neural Networks for Privacy Protection**. Hao Zhang, Yiting Chen, Haotian Ma, Xu Cheng, Qihan Ren, Liyao Xiang, Jie Shi, and Quanshi Zhang. In arXiv:2006.13016, 2020.



## Data Preparation

```shell
mkdir data
```

Download CelebA.zip and cropped_CelebA.zip, move them to the `data` directory and unzip them. The CIFAR-10 and CIFAR-100 datasets will be downloaded automatically when running the following Python scripts, so you do not need to worry about them.



## Inversion Attack

By changing the `--dim` argument, you can specify the dimension of the multi-ary feature. Currently support --`--dim=2|3|5`. Specifically, when dim=2, RENN is equivalent to the Complex-Valued Neural Network. When dim=3, RENN is equivalent to the Quaternion Neural Network. 

To specify the GPU id, you can use the `--device` argument.

#### Lenet CIFAR-10

**Train Lenet:**
```
python privacy_GAN_trainer.py --lr=0.01 --epoch-num=50 --G-lr=1e-4 --D-lr=1e-4 --result-path=lenet_cifar10 --dim=5
```

**Train Decoder for Lenet:**  
```
python unet_decoder_trainer.py --decoder-type=3 --result-path=lenet_cifar10 --load-net-epoch=49 --dim=5

python unet_decoder_trainer.py --decoder-type=4 --result-path=lenet_cifar10 --load-net-epoch=49 --dim=5
```
**Inversion Attack Result for Lenet:**

```
python unet_test_result.py  --decoder-type=5 --result-path=lenet_cifar10 --load-net-epoch=49 --dim=5

python unet_test_result.py  --decoder-type=4 --result-path=lenet_cifar10 --load-net-epoch=49 --dim=5
```
**View Results**

- result_dim{dim}/lenet-cifar10/visual_result/best_sample (for dec(a'))
  - folder `1` : images for human annotators
  - mse_error_a.txt: reconstruction error
- result_dim{dim}/lenet-cifar10/visual_result/unet_decoder_privacy_GAN_encrypt_stage (for dec(x))
  - folder `1` : images for human annotators
  - mse_error_a.txt: reconstruction error



#### ResNet-56 CIFAR-10

**Train ResNet-56:**
```
python privacy_GAN_trainer.py --lr=0.1 --epoch-num=100 --G-lr=1e-3 --D-lr=5e-4 --result-path=resnet56_cifar10 --dim=5
```

**Train Decoder for ResNet-56:**  
```
python unet_decoder_trainer.py --decoder-type=3 --result-path=resnet56_cifar10 --load-net-epoch=99 --dim=5

python unet_decoder_trainer.py --decoder-type=4 --result-path=resnet56_cifar10 --load-net-epoch=99 --dim=5
```
**Inversion Attack Result for ResNet-56:**
```
python unet_test_result.py --decoder-type=5 --result-path=resnet56_cifar10 --load-net-epoch=99 --dim=5

python unet_test_result.py --decoder-type=4 --result-path=resnet56_cifar10 --load-net-epoch=99 --dim=5
```
**View results**

- result_dim{dim}/resnet56-cifar10/visual_result/best_sample (for dec(a'))
  - folder `1` : images for human annotators
  - mse_error_a.txt: reconstruction error
- result_dim{dim}/resnet56-cifar10/visual_result/unet_decoder_privacy_GAN_encrypt_stage (for dec(x))
  - folder `1` : images for human annotators
  - mse_error_a.txt: reconstruction error



#### AlexNet CelebA

**Train AlexNet:**
```
python privacy_GAN_trainer_CelebA.py --dim=5
```
**Train Decoder for AlexNet:**
```
python unet_CelebA_decoder_trainer.py --decoder-type=3 --load-net-epoch=9 --dim=5

python unet_CelebA_decoder_trainer.py --decoder-type=4 --load-net-epoch=9 --dim=5
```
**Inversion Attack Result for AlexNet:**
```
python unet_CelebA_test_result.py --decoder-type=5 --load-net-epoch=9 --dim=5

python unet_CelebA_test_result.py --decoder-type=4 --load-net-epoch=9 --dim=5
```
**View results**

- result_dim{dim}/Alexnet_CelebA/visual_result/best_sample2 (for dec(a'))

  - folder `3` : images for human annotators
  - mse_error_c.txt: reconstruction error

- result_dim{dim}/Alexnet_CelebA/visual_result/unet_decoder_privacy_GAN_encrypt_stage (for dec(x))

  - folder `3` : images for human annotators

  - mse_error_c.txt: reconstruction error



## Inference Attack

#### ResNet-56 CIFAR-100

**Train ResNet-56 on the Coarse 20 Categories:**
```
python infer_cifar100/infer_net_normal.py --dim=5
```
**Train Decoder for ResNet-56:**
```
python infer_cifar100/unet_decoder_trainer_infer_cifar100.py --decoder-type=3 --dim=5

python infer_cifar100/unet_decoder_trainer_infer_cifar100.py --decoder-type=4 --dim=5
```
**Generate Best-sample and Best-feature dataset:**

```
python infer_cifar100/result_dataset.py --decoder-type=5 --dim=5
```

**Inference Attack Result:**
```
python infer_cifar100/infer_baseline_1.py --dim=5

python infer_cifar100/infer_eval_best_sample.py --dim=5

python infer_cifar100/infer_feature_eval_best_sample.py --dim=5

python infer_cifar100/infer_knn_best.py --k-nearest=1 --dim=5

python infer_cifar100/infer_knn_best.py --k-nearest=3 --dim=5

python infer_cifar100/infer_knn_best.py --k-nearest=5 --dim=5
```
**View Results**

To view the reconstruction error of net($I$), net($\hat I$) and net($a'$), run

```shell
python print_inference_atk_result.py --result-path=resnet56_cifar100_infer --type=inf_atk1 --dim=5
python print_inference_atk_result.py --result-path=resnet56_cifar100_infer --type=inf_atk2 --dim=5
python print_inference_atk_result.py --result-path=resnet56_cifar100_infer --type=inf_atk3 --dim=5
```

Remember to set the `--dim` argument according to your need.

To view the reconstruction error for knn (k=1,3,5), see `result_dim{dim}/resnet56_cifar100_infer/eval_knn_best`



#### AlexNet CelebA

**Train AlexNet on the 30 Public Categories:**
```
python infer_CelebA/infer_CelebA_net_normal.py --dim=5
```
**Train Decoder for AlexNet:**
```
python infer_CelebA/unet_decoder_trainer_infer_CelebA.py --decoder-type=3 --dim=5

python infer_CelebA/unet_decoder_trainer_infer_CelebA.py --decoder-type=4 --dim=5
```
**Generate Best-sample and Best-feature dataset:**
```
python infer_CelebA/CelebA_result_dataset.py --decoder-type=5 --dim=5
```
**Inference Attack Result:**

```
python infer_CelebA/infer_CelebA_baseline1.py --dim=5

python infer_CelebA/infer_CelebA_best.py --dim=5

python infer_CelebA/infer_CelebA_feature.py --dim=5

python infer_CelebA/infer_CelebA_knn.py --k-nearest=1 --dim=5

python infer_CelebA/infer_CelebA_knn.py --k-nearest=3 --dim=5

python infer_CelebA/infer_CelebA_knn.py --k-nearest=5 --dim=5
```
**View Results**

To view the reconstruction error of net($I$), net($\hat I$) and net($a'$), run

```shell
python print_inference_atk_result.py --result-path=Alexnet_CelebA_infer --type=inf_atk1 --dim=5
python print_inference_atk_result.py --result-path=Alexnet_CelebA_infer --type=inf_atk2 --dim=5
python print_inference_atk_result.py --result-path=Alexnet_CelebA_infer --type=inf_atk3 --dim=5
```

Remember to set the `--dim` argument according to your need.

To view the reconstruction error for knn (k=1,3,5), see `result_dim{dim}/Alexnet_CelebA_infer/3/eval_knn_best`
