# deep-learning-
模型说明：
我们有两个网络，G（生成器）和 D（判别器）。生成器是用于生成图像的网络。它接收一个随机噪声 z 并从这个噪声生成图像，称为 G(z)。判别器是一个判别网络，用于判别图像是否真实。输入是x，x是图片，输出是D，x是x是真实图片的概率，如果为1，则100%真实，如果为0，则不真实。

数据集说明：
我们使用了DIV2K数据集，其中训练集有800张图片，测试集有100张图片
download dataset
$ cd data/
$ bash download_dataset.sh

测试图片：
启动test_image.py

1.测试命令：
test_image.py [-h] [--arch {rfb_esrgan}] --lr LR [--hr HR]
                     [--upscale-factor {16}] [--model-path MODEL_PATH]
                     [--pretrained] [--seed SEED] [--gpu GPU]

2.参数说明如下:
  -h, --help            show this help message and exit
  --arch {rfb_esrgan}   Model architecture: rfb_esrgan. (Default: `rfb_esrgan`)
  --lr LR               Test low resolution image name.
  --hr HR               Raw high resolution image name.
  --upscale-factor {16}
                        Low to high resolution scaling factor. Optional: [16].
                        (Default: 16)
  --model-path MODEL_PATH
                        Path to latest checkpoint for model.
  --pretrained          Use pre-trained model.
  --seed SEED           Seed for initializing training.
  --gpu GPU             GPU id to use.
  
3.Example
$ python3 test_image.py --arch rfb_esrgan --lr [path-to-lr-image] --hr [Optional, path-to-hr-image] --pretrained --gpu 0






训练：
一、启动train.py训练基于psnr驱动的模型 (e.g DIV2K)


1.测试命令： 
train.py [-h] [--arch {rfb_esrgan}] [-j WORKERS]
                [--psnr-epochs PSNR_EPOCHS]
                [--start-psnr-epoch START_PSNR_EPOCH]
                [--gan-epochs GAN_EPOCHS] [--start-gan-epoch START_GAN_EPOCH]
                [-b BATCH_SIZE] [--psnr-lr PSNR_LR] [--gan-lr GAN_LR]
                [--image-size IMAGE_SIZE] [--upscale-factor {16}]
                [--netD NETD] [--netG NETG] [--pretrained]
                [--world-size WORLD_SIZE] [--rank RANK] [--dist-url DIST_URL]
                [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU]
                [--multiprocessing-distributed]
                DIR

2.参数说明:
  DIR                   Path to dataset.
  -h, --help            show this help message and exit
  --arch {rfb_esrgan}   Model architecture: rfb_esrgan. (Default:
                        `rfb_esrgan`)
  -j WORKERS, --workers WORKERS
                        Number of data loading workers. (Default: 4)
  --psnr-epochs PSNR_EPOCHS
                        Number of total psnr epochs to run. (Default: 128)
  --start-psnr-epoch START_PSNR_EPOCH
                        Manual psnr epoch number (useful on restarts).
                        (Default: 0)
  --gan-epochs GAN_EPOCHS
                        Number of total gan epochs to run. (Default: 64)
  --start-gan-epoch START_GAN_EPOCH
                        Manual gan epoch number (useful on restarts).
                        (Default: 0)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        The batch size of the dataset. (Default: 16)
  --psnr-lr PSNR_LR     Learning rate for psnr-oral. (Default: 0.0002)
  --gan-lr GAN_LR       Learning rate for gan-oral. (Default: 0.0001)
  --image-size IMAGE_SIZE
                        Image size of high resolution image. (Default: 512)
  --upscale-factor {16}
                        Low to high resolution scaling factor. Optional: [16].
                        (Default: 16)
  --netD NETD           Path to Discriminator checkpoint.
  --netG NETG           Path to Generator checkpoint.
  --pretrained          Use pre-trained model.
  --world-size WORLD_SIZE
                        Number of nodes for distributed training.
  --rank RANK           Node rank for distributed training. (Default: -1)
  --dist-url DIST_URL   url used to set up distributed training. (Default:
                        `tcp://59.110.31.55:12345`)
  --dist-backend DIST_BACKEND
                        Distributed backend. (Default: `nccl`)
  --seed SEED           Seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs. This is the
                        fastest way to use PyTorch for either single node or
                        multi node data parallel training.
                                       
3.Example (e.g DIV2K)
$ python3 train.py --arch rfb_esrgan [image-folder with train and val folders]



4.如果您想调用之前训练过的模型继续训练，可以使用以下命令：
$ python3 train.py --arch rfb_esrgan [image-folder with train and val folders] \
                   --start-psnr-epoch 10 \
                   --netG weights/PSNR_epoch10.pth \
                   --gpu 0

二、启动gantrain.py训练基于gan驱动的模型 (e.g DIV2K)
过程和参数与一相同

