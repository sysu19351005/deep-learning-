import argparse
import logging
import math
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
import random
import time
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from tensorboardX import SummaryWriter

import rfb_esrgan_pytorch.models as models
from rfb_esrgan_pytorch.dataset import BaseTestDataset
from rfb_esrgan_pytorch.dataset import BaseTrainDataset
from rfb_esrgan_pytorch.loss import ContentLoss
from rfb_esrgan_pytorch.models.discriminator import discriminator_for_vgg
from rfb_esrgan_pytorch.utils.common import AverageMeter
from rfb_esrgan_pytorch.utils.common import ProgressMeter
from rfb_esrgan_pytorch.utils.common import configure
from rfb_esrgan_pytorch.utils.common import create_folder
from rfb_esrgan_pytorch.utils.estimate import test

# Find all available models.

model_names = sorted(name for name in models.__dict__ if
                     name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

# It is a convenient method for simple scripts to configure the log package at one time.
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)
best_psnr = 0.0
best_ssim = 0.0
# Load base low-resolution image.
fixed_lr = transforms.ToTensor()(Image.open(os.path.join("assets", "butterfly.png"))).unsqueeze(0)


def main(args):
    if args.seed is not None:
        # In order to make the model repeatable, the first step is to set random seeds, and the second step is to set
        # convolution algorithm.
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        logger.warning("You have chosen to seed training. "
                       "This will turn on the CUDNN deterministic setting, "
                       "which can slow down your training considerably! "
                       "You may see unexpected behavior when restarting "
                       "from checkpoints.")
        # for the current configuration, so as to optimize the operation efficiency.
        cudnn.benchmark = True
        # Ensure that every time the same input returns the same result.
        cudnn.deterministic = True

    if args.gpu is not None:
        logger.warning("You have chosen a specific GPU. This will completely disable data parallelism.")

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size needs to be adjusted accordingly.
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the main_worker process function.
        mp.spawn(main_worker, args=(ngpus_per_node, args), nprocs=ngpus_per_node)
    else:
        # Simply call main_worker function
        main_worker(ngpus_per_node, args)


def main_worker(ngpus_per_node, args):
    global best_psnr, best_ssim, fixed_lr

    if args.gpu is not None:
        logger.info(f"Use GPU: {args.gpu} for training.")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + args.gpu
        dist.init_process_group(args.dist_backend, args.dist_url, world_size=args.world_size, rank=args.rank)
    # create model
    generator = configure(args)
    discriminator = discriminator_for_vgg(args.image_size)

    if not torch.cuda.is_available():
        logger.warning("Using CPU, this will be slow.")
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            discriminator.cuda(args.gpu)
            generator.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            discriminator = nn.parallel.DistributedDataParallel(discriminator, device_ids=[0,1])
            generator = nn.parallel.DistributedDataParallel(generator, device_ids=[0,1])
            #discriminator=torch.nn.DataParallel(discriminator, device_ids=[0,1])
            #generator=torch.nn.DataParallel(generator, device_ids=[0,1])
        else:
            discriminator.cuda()
            generator.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            discriminator = nn.parallel.DistributedDataParallel(discriminator)
            generator = nn.parallel.DistributedDataParallel(generator)
    elif args.gpu is not None:
        #torch.cuda.set_device(args.gpu)
        discriminator = discriminator.cuda(args.gpu)
        generator = generator.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            discriminator.features = torch.nn.DataParallel(discriminator.features)
            generator.features = torch.nn.DataParallel(generator.features)
            discriminator.cuda()
            generator.cuda()
        else:
            discriminator = torch.nn.DataParallel(discriminator).cuda()
            generator = torch.nn.DataParallel(generator).cuda()

    # Loss = 10 * pixel loss + content loss + 0.005 * adversarial loss
    pixel_criterion = nn.L1Loss().cuda(args.gpu)
    content_criterion = ContentLoss().cuda(args.gpu)
    adversarial_criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)

    if args.gpu is not None:
        fixed_lr = fixed_lr.cuda(args.gpu)

    # All optimizer function and scheduler function.
    psnr_optimizer = torch.optim.Adam(generator.parameters(), lr=args.psnr_lr, betas=(0.9, 0.99))
    psnr_epoch_indices = math.floor(args.psnr_epochs // 4)
    psnr_scheduler = torch.optim.lr_scheduler.StepLR(psnr_optimizer, psnr_epoch_indices, 0.5)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.gan_lr, betas=(0.9, 0.99))
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.gan_lr, betas=(0.9, 0.99))
    interval_epoch = math.ceil(args.gan_epochs // 8)
    gan_epoch_indices = [interval_epoch, interval_epoch * 2, interval_epoch * 4, interval_epoch * 6]
    discriminator_scheduler = torch.optim.lr_scheduler.MultiStepLR(discriminator_optimizer, gan_epoch_indices, 0.5)
    generator_scheduler = torch.optim.lr_scheduler.MultiStepLR(generator_optimizer, gan_epoch_indices, 0.5)

    # Selection of appropriate treatment equipment.
    train_dataset = BaseTrainDataset(os.path.join(args.data, "train"), args.image_size, args.upscale_factor)
    test_dataset = BaseTestDataset(os.path.join(args.data, "test"), args.image_size, args.upscale_factor)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=(train_sampler is None),
                                                   pin_memory=True,
                                                   sampler=train_sampler,
                                                   num_workers=args.workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=args.workers)

    # Load pre training model.
    # if args.netD != "":
    #     discriminator.load_state_dict(torch.load(args.netD))
    # if args.netG != "":
    #     generator.load_state_dict(torch.load(args.netG))

        # The mixed precision training is used in PSNR-oral.
    if args.netG != "":
        generator.load_state_dict(torch.load(args.netG))

        # The mixed precision training is used in PSNR-oral.
        scaler = amp.GradScaler()
        logger.info("Turn on mixed precision training.")

        # Create a SummaryWriter at the beginning of training.
        psnr_writer = SummaryWriter(f"runs/{args.arch}_psnr_logs")
        gan_writer = SummaryWriter(f"runs/{args.arch}_gan_logs")

        for epoch in range(args.start_psnr_epoch, args.psnr_epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)

            # Train for one epoch for PSNR-oral.
            train_psnr(train_dataloader, generator, pixel_criterion, psnr_optimizer, epoch, scaler, psnr_writer, args)
            # Update GAN-oral optimizer learning rate.
            psnr_scheduler.step()

            # Evaluate on test dataset.
            psnr, ssim, gmsd = test(test_dataloader, generator, args.gpu)
            psnr_writer.add_scalar("PSNR_Test/PSNR", psnr, epoch)
            psnr_writer.add_scalar("PSNR_Test/SSIM", ssim, epoch)
            psnr_writer.add_scalar("PSNR_Test/GMSD", gmsd, epoch)
            f=open("result.txt","a")
            f.write('{} {} {} {} \n'.format(epoch, psnr, ssim, gmsd))
            # Check whether the evaluation index of the current model is the highest.
            is_best = psnr > best_psnr
            best_psnr = max(psnr, best_psnr)
            # Save model weights for every epoch.
            if not args.multiprocessing_distributed or (
                    args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                torch.save(generator.state_dict(), os.path.join("weights", f"PSNR_epoch{epoch}.pth"))
                if is_best:
                    torch.save(generator.state_dict(), os.path.join("weights", f"PSNR-best.pth"))

            # Save the last training model parameters.
        torch.save(generator.state_dict(), os.path.join("weights", f"PSNR-last.pth"))

    


def train_psnr(dataloader, model, criterion, optimizer, epoch, scaler, writer, args):
    batch_time = AverageMeter("Time", ":6.6f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(num_batches=len(dataloader),
                             meters=[batch_time, losses],
                             prefix=f"Epoch: [{epoch}]")

    # Switch to train mode.
    model.train()

    end = time.time()
    for i, (lr, hr) in enumerate(dataloader):
        # Move data to special device.
        if args.gpu is not None:
            lr = lr.cuda(args.gpu, non_blocking=True)
            hr = hr.cuda(args.gpu, non_blocking=True)

        # Start mixed precision training.
        optimizer.zero_grad()

        with amp.autocast():
            sr = model(lr)
            loss = criterion(sr, hr)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Measure elapsed time.
        batch_time.update(time.time() - end)
        end = time.time()
        # measure accuracy and record loss
        losses.update(loss.item(), lr.size(0))

        # Add scalar data to summary.
        writer.add_scalar("PSNR_Train/Loss", loss.item(), i + epoch * len(dataloader) + 1)

        # Output results every 100 batches.
        if i % 100 == 0:
            progress.display(i)

    # Each one epoch create a sr image.
    with torch.no_grad():
        sr = model(fixed_lr)
        vutils.save_image(sr.detach(), os.path.join("runs", f"PSNR_epoch_{epoch}.png"), normalize=True)


def train_gan(dataloader, discriminator, discriminator_optimizer, generator, generator_optimizer,
              pixel_criterion, content_criterion, adversarial_criterion, epoch, scaler, writer, args):
    batch_time = AverageMeter("Time", ":6.4f")
    d_losses = AverageMeter("D Loss", ":6.6f")
    g_losses = AverageMeter("G Loss", ":6.6f")
    pixel_losses = AverageMeter("Pixel Loss", ":6.6f")
    content_losses = AverageMeter("Content Loss", ":6.6f")
    adversarial_losses = AverageMeter("Adversarial Loss", ":6.6f")
    progress = ProgressMeter(num_batches=len(dataloader),
                             meters=[batch_time, d_losses, g_losses, pixel_losses, content_losses, adversarial_losses],
                             prefix=f"Epoch: [{epoch}]")

    # switch to train mode
    discriminator.train()
    generator.train()

    end = time.time()
    for i, (lr, hr) in enumerate(tqdm(dataloader)):
        # Move data to special device.
        if args.gpu is not None:
            lr = lr.cuda(args.gpu, non_blocking=True)
            hr = hr.cuda(args.gpu, non_blocking=True)
        batch_size = lr.size(0)

        # The real sample label is 1, and the generated sample label is 0.
        real_label = torch.full((batch_size, 1), 1, dtype=lr.dtype).cuda(args.gpu, non_blocking=True)
        fake_label = torch.full((batch_size, 1), 0, dtype=lr.dtype).cuda(args.gpu, non_blocking=True)

        ##############################################
        # (1) Update D network: E(hr)[fake(C(D(hr) - E(sr)C(sr)))] + E(sr)[fake(C(fake) - E(real)C(real))]
        ##############################################
        discriminator_optimizer.zero_grad()

        with amp.autocast():
            sr = generator(lr)
            # It makes the discriminator distinguish between real sample and fake sample.
            real_output = discriminator(hr)
            fake_output = discriminator(sr.detach())

            # Adversarial loss for real and fake images (relativistic average GAN)
            d_loss_real = adversarial_criterion(real_output - torch.mean(fake_output), real_label)
            d_loss_fake = adversarial_criterion(fake_output - torch.mean(real_output), fake_label)

            # Count all discriminator losses.
            d_loss = d_loss_real + d_loss_fake

        scaler.scale(d_loss).backward()
        scaler.step(discriminator_optimizer)
        scaler.update()

        ##############################################
        # (2) Update G network: E(hr)[sr(C(D(hr) - E(sr)C(sr)))] + E(sr)[sr(C(fake) - E(real)C(real))]
        ##############################################
        generator_optimizer.zero_grad()

        with amp.autocast():
            sr = generator(lr)
            # It makes the discriminator unable to distinguish the real samples and fake samples.
            real_output = discriminator(hr.detach())
            fake_output = discriminator(sr)

            # Calculate the absolute value of pixels with L1 loss.
            pixel_loss = pixel_criterion(sr, hr.detach())
            # The 35th layer in VGG19 is used as the feature extractor by default.
            content_loss = content_criterion(sr, hr.detach())
            # Adversarial loss for real and fake images (relativistic average GAN)
            adversarial_loss = adversarial_criterion(fake_output - torch.mean(real_output), real_label)

            # Count all generator losses.
            g_loss = 10 * pixel_loss + 1 * content_loss + 0.005 * adversarial_loss

        scaler.scale(g_loss).backward()
        scaler.step(generator_optimizer)
        scaler.update()

        # Set generator gradients to zero.
        generator.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # measure accuracy and record loss
        d_losses.update(d_loss.item(), lr.size(0))
        g_losses.update(g_loss.item(), lr.size(0))
        pixel_losses.update(pixel_loss.item(), lr.size(0))
        content_losses.update(content_loss.item(), lr.size(0))
        adversarial_losses.update(adversarial_loss.item(), lr.size(0))

        iters = i + epoch * len(dataloader) + 1
        writer.add_scalar("GAN_Train/D_Loss", d_loss.item(), iters)
        writer.add_scalar("GAN_Train/G_Loss", g_loss.item(), iters)
        writer.add_scalar("GAN_Train/Pixel_Loss", pixel_loss.item(), iters)
        writer.add_scalar("GAN_Train/Content_Loss", content_loss.item(), iters)
        writer.add_scalar("GAN_Train/Adversarial_Loss", adversarial_loss.item(), iters)

        # Output results every 100 batches.
        if i % 100 == 0:
            progress.display(i)

    # Each one epoch create a sr image.
    with torch.no_grad():
        sr = generator(fixed_lr)
        vutils.save_image(sr.detach(), os.path.join("runs", f"GAN_epoch_{epoch}.png"), normalize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", metavar="DIR",default="./data/",
                        help="Path to dataset.")
    parser.add_argument("--arch", default="rfb_esrgan", type=str, choices=model_names,
                        help="Model architecture: " +
                             " | ".join(model_names) +
                             ". (Default: `rfb_esrgan`)")
    parser.add_argument("-j", "--workers", default=4, type=int,
                        help="Number of data loading workers. (Default: 4)")
    parser.add_argument("--psnr-epochs", default=600, type=int,
                        help="Number of total psnr epochs to run. (Default: 128)")
    parser.add_argument("--start-psnr-epoch", default=0, type=int,
                        help="Manual psnr epoch number (useful on restarts). (Default: 0)")
    parser.add_argument("--gan-epochs", default=64, type=int,
                        help="Number of total gan epochs to run. (Default: 64)")
    parser.add_argument("--start-gan-epoch", default=0, type=int,
                        help="Manual gan epoch number (useful on restarts). (Default: 0)")
    parser.add_argument("-b", "--batch-size", default=8, type=int,
                        help="The batch size of the dataset. (Default: 16)")
    parser.add_argument("--psnr-lr", default=0.0002, type=float,
                        help="Learning rate for psnr-oral. (Default: 0.0002)")
    parser.add_argument("--gan-lr", default=0.0001, type=float,
                        help="Learning rate for gan-oral. (Default: 0.0001)")
    parser.add_argument("--image-size", default=512, type=int,
                        help="Image size of high resolution image. (Default: 512)")
    parser.add_argument("--upscale-factor", default=16, type=int, choices=[16],
                        help="Low to high resolution scaling factor. Optional: [16]. (Default: 16)")
    parser.add_argument("--netD", default="", type=str,
                        help="Path to Discriminator checkpoint.")
    parser.add_argument("--netG", default="", type=str,
                        help="Path to Generator checkpoint.")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                        help="Use pre-trained model.")
    parser.add_argument("--world-size", default=-1, type=int,
                        help="Number of nodes for distributed training.")
    parser.add_argument("--rank", default=-1, type=int,
                        help="Node rank for distributed training. (Default: -1)")
    parser.add_argument("--dist-url", default="tcp://59.110.31.55:12345", type=str,
                        help="url used to set up distributed training. (Default: `tcp://59.110.31.55:12345`)")
    parser.add_argument("--dist-backend", default="nccl", type=str,
                        help="Distributed backend. (Default: `nccl`)")
    parser.add_argument("--seed", default=None, type=int,
                        help="Seed for initializing training.")
    parser.add_argument("--gpu", default=0, type=int,
                        help="GPU id to use.")
    parser.add_argument("--multiprocessing-distributed", action="store_true",
                        help="Use multi-processing distributed training to launch "
                             "N processes per node, which has N GPUs. This is the "
                             "fastest way to use PyTorch for either single node or "
                             "multi node data parallel training.")
    args = parser.parse_args()

    create_folder("runs")
    create_folder("weights")

    logger.info("TrainEngine:")
    logger.info("\tAPI version .......... 0.3.0")


    main(args)
  

    logger.info("All training has been completed successfully.\n")
