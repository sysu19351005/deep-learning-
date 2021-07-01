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
test_image.py [-h] [--arch {rfb_esrgan}] --lr LR [--hr HR]<br>
                     [--upscale-factor {16}] [--model-path MODEL_PATH]<br>
>>>>>>>> [--pretrained] [--seed SEED] [--gpu GPU]<br>  

2.参数说明如下:
  -h, --help            show this help message and exit<br>
  --arch {rfb_esrgan}   Model architecture: rfb_esrgan. (Default: `rfb_esrgan`)<br>
  --lr LR               Test low resolution image name.<br>
  --hr HR               Raw high resolution image name.<br>
  --upscale-factor {16}<br>
                        Low to high resolution scaling factor. Optional: [16].<br>
                        (Default: 16)<br>
  --model-path MODEL_PATH<br>
                        Path to latest checkpoint for model.<br>
  --pretrained          Use pre-trained model.<br>
  --seed SEED           Seed for initializing training.<br>
  --gpu GPU             GPU id to use.<br>
  
3.Example<br>
$ python3 test_image.py --arch rfb_esrgan --lr [path-to-lr-image] --hr [Optional, path-to-hr-image] --pretrained --gpu 0<br>






训练：
一、启动train.py训练基于psnr驱动的模型 (e.g DIV2K)<br>


1.测试命令： 
train.py [-h] [--arch {rfb_esrgan}] [-j WORKERS]<br>
                [--psnr-epochs PSNR_EPOCHS]<br>
                [--start-psnr-epoch START_PSNR_EPOCH]<br>
                [--gan-epochs GAN_EPOCHS] [--start-gan-epoch START_GAN_EPOCH]<br>
                [-b BATCH_SIZE] [--psnr-lr PSNR_LR] [--gan-lr GAN_LR]<br>
                [--image-size IMAGE_SIZE] [--upscale-factor {16}]<br>
                [--netD NETD] [--netG NETG] [--pretrained]<br>
                [--world-size WORLD_SIZE] [--rank RANK] [--dist-url DIST_URL]<br>
                [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU]<br>
                [--multiprocessing-distributed]<br>
                DIR<br>

2.参数说明:
  DIR                   Path to dataset.<br>
  -h, --help            show this help message and exit<br>
  --arch {rfb_esrgan}   Model architecture: rfb_esrgan. (Default:<br>
                        `rfb_esrgan`)<br>
  -j WORKERS, --workers WORKERS<br>
                        Number of data loading workers. (Default: 4)<br>
  --psnr-epochs PSNR_EPOCHS<br>
                        Number of total psnr epochs to run. (Default: 128)<br>
  --start-psnr-epoch START_PSNR_EPOCH<br>
                        Manual psnr epoch number (useful on restarts).<br>
                        (Default: 0)<br>
  --gan-epochs GAN_EPOCHS<br>
                        Number of total gan epochs to run. (Default: 64)<br>
  --start-gan-epoch START_GAN_EPOCH<br>
                        Manual gan epoch number (useful on restarts).<br>
                        (Default: 0)<br>
  -b BATCH_SIZE, --batch-size BATCH_SIZE<br>
                        The batch size of the dataset. (Default: 16)<br>
  --psnr-lr PSNR_LR     Learning rate for psnr-oral. (Default: 0.0002)<br>
  --gan-lr GAN_LR       Learning rate for gan-oral. (Default: 0.0001)<br>
  --image-size IMAGE_SIZE<br>
                        Image size of high resolution image. (Default: 512)<br>
  --upscale-factor {16}<br>
                        Low to high resolution scaling factor. Optional: [16].<br>
                        (Default: 16)<br>
  --netD NETD           Path to Discriminator checkpoint.<br>
  --netG NETG           Path to Generator checkpoint.<br>
  --pretrained          Use pre-trained model.<br>
  --world-size WORLD_SIZE<br>
                        Number of nodes for distributed training.<br>
  --rank RANK           Node rank for distributed training. (Default: -1)<br>
  --dist-url DIST_URL   url used to set up distributed training. (Default:<br>
                        `tcp://59.110.31.55:12345`)<br>
  --dist-backend DIST_BACKEND<br>
                        Distributed backend. (Default: `nccl`)<br>
  --seed SEED           Seed for initializing training.<br>
  --gpu GPU             GPU id to use.<br>
  --multiprocessing-distributed<br>
                        Use multi-processing distributed training to launch N<br>
                        processes per node, which has N GPUs. This is the<br>
                        fastest way to use PyTorch for either single node or<br>
                        multi node data parallel training.<br>
                                       
3.Example (e.g DIV2K)<br>
$ python3 train.py --arch rfb_esrgan [image-folder with train and val folders]<br>



4.如果您想调用之前训练过的模型继续训练，可以使用以下命令：<br>
$ python3 train.py --arch rfb_esrgan [image-folder with train and val folders] \<br>
                   --start-psnr-epoch 10 \<br>
                   --netG weights/PSNR_epoch10.pth \<br>
                   --gpu 0<br>

二、启动gantrain.py训练基于gan驱动的模型 (e.g DIV2K)<br>
过程和参数与一相同<br>

