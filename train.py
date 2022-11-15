import os
import torch
import numpy as np
import logging
import time
import torch.nn as nn
import random
import calculate_psnr_ssim as util

from torchvision import transforms
from collections import OrderedDict
from config import Config
from torch.utils.data import DataLoader
from data import ISPData

from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm



os.environ['CUDA_VISIBLE_DEVICES'] = '7, 0'


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logging(config):
    if not os.path.exists(config.log_path): 
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )


if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)

    # config file
    config = Config({
        # dataset path
        "dataset_path": "/mnt/data/wth22/ISP_dataset/train/",

        # optimization
        "batch_size": 4,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "n_epoch": 300,
        "val_freq": 1,
        "T_max": 50,
        "eta_min": 0,
        "num_workers": 8,

        # model


        # load & save checkpoint
        "model_name": "ISP-temp",
        "output_path": "./output",
        "snap_path": "./output/models/",               # directory for saving checkpoint
        "log_path": "./output/log/ISP/",
        "log_file": ".txt",
        "tensorboard_path": "./output/tensorboard/"
    })

    if not os.path.exists(config.output_path):
        os.mkdir(config.output_path)

    if not os.path.exists(config.snap_path):
        os.mkdir(config.snap_path)

    if not os.path.exists(config.tensorboard_path):
        os.mkdir(config.tensorboard_path)

    config.snap_path += config.model_name
    config.log_file = config.model_name + config.log_file
    config.tensorboard_path += config.model_name

    set_logging(config)
    logging.info(config)

    writer = SummaryWriter(config.tensorboard_path)
    TrainDataset = ISPData(config.dataset_path, type="train")
    ValDataset = ISPData(config.dataset_path, type="val")

    train_loader = DataLoader(
        dataset=TrainDataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=ValDataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=True,
        shuffle=False
    )

    # net = creat_model(config=config, model_weight_path=config.pretrained_weight_path, pretrained=config.pretrained)
    # net = nn.DataParallel(net).cuda()
    logging.info('{} : {} [M]'.format('#Params', sum(map(lambda x: x.numel(), net.parameters())) / 10 ** 6))

    # loss function
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

    # # make directory for saving weights
    if not os.path.exists(config.snap_path):
        os.mkdir(config.snap_path)

    train_results = OrderedDict()
    train_results['psnr'] = []
    train_results['ssim'] = []

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    best_psnr = 0
    bset_ssim = 0
    main_score = 0

    for epoch in range(0, config.n_epoch):
        start_time = time.time()
        logging.info('Running training epoch {}'.format(epoch + 1))

        for data in tqdm(train_loader):
            rgb_image, raw_image = data
            rgb_image = rgb_image.cuda()
            raw_image = raw_image.cuda()

            pred_image = net(d, table)

            optimizer.zero_grad()
            loss = criterion(pred_image, raw_image)

            loss.backward()
            optimizer.step()
            scheduler.step()

            pred_image = (pred_image * 255.0).round().astype(np.uint8)  # float32 to uint8
            raw_image = (raw_image * 255.0).round().astype(np.uint8)  # float32 to uint8
            pred_image = np.transpose(pred_image, (1, 2, 0))
            raw_image = np.transpose(raw_image, (1, 2, 0))
            psnr = util.calculate_psnr(pred_image, raw_image, crop_border=0)
            ssim = util.calculate_ssim(pred_image, raw_image, crop_border=0)
            train_results['psnr'].append(psnr)
            train_results['ssim'].append(ssim)

        ave_psnr = sum(train_results['psnr']) / len(train_results['psnr'])
        ave_ssim = sum(train_results['ssim']) / len(train_results['ssim'])
        writer.add_scalar("PSNR", ave_psnr, epoch)
        writer.add_scalar("SSIM", ave_ssim, epoch)
        logging.info('--Training Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}'.format(ave_psnr, ave_ssim))

        if (epoch + 1) % config.val_freq == 0:
            logging.info('Starting eval...')
            logging.info('Running val {} in epoch {}'.format(config.dataset_name, epoch + 1))
            
            for data in tqdm(val_loader):
                rgb_image, raw_image = data
                rgb_image = rgb_image.cuda()
                raw_image = raw_image.cuda()

                pred_image = net(d, table)

                pred_image = (pred_image * 255.0).round().astype(np.uint8)  # float32 to uint8
                raw_image = (raw_image * 255.0).round().astype(np.uint8)  # float32 to uint8
                pred_image = np.transpose(pred_image, (1, 2, 0))
                raw_image = np.transpose(raw_image, (1, 2, 0))
                psnr = util.calculate_psnr(pred_image, raw_image, crop_border=0)
                ssim = util.calculate_ssim(pred_image, raw_image, crop_border=0)
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)
            
            ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
            ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
            writer.add_scalar("PSNR", ave_psnr, epoch)
            writer.add_scalar("SSIM", ave_ssim, epoch)
            logging.info('--Testing Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}'.format(ave_psnr, ave_ssim))
            logging.info('Eval done...')

            if ave_psnr + ave_ssim > main_score:
                main_score = ave_psnr + ave_ssim
                logging.info('======================================================================================')
                logging.info('============================== best main score is {} ================================='.format(main_score))
                logging.info('======================================================================================')
        
                best_psnr = ave_psnr
                best_ssim = ave_ssim

                # save weights
                model_name = "epoch{}.pt".format(epoch + 1)
                model_save_path = os.path.join(config.snap_path, model_name)
                torch.save(net.module.state_dict(), model_save_path)
                logging.info('Saving weights and model of epoch{}, PSNR:{}, SSIM:{}'.format(epoch + 1, best_psnr, best_ssim))
        
        logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))