import torch
from osgeo import gdal
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np



def arr2raster(arr, raster_file, prj=None, trans=None):
    """
    将数组转成栅格文件写入硬盘
    :param arr: 输入的mask数组 ReadAsArray()
    :param raster_file: 输出的栅格文件路径
    :param prj: gdal读取的投影信息 GetProjection()，默认为空
    :param trans: gdal读取的几何信息 GetGeoTransform()，默认为空
    :return:
    """

    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(raster_file, arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32)

    if prj:
        dst_ds.SetProjection(prj)
    if trans:
        dst_ds.SetGeoTransform(trans)

    # 将数组的各通道写入图片
    dst_ds.GetRasterBand(1).WriteArray(arr)

    dst_ds.FlushCache()
    dst_ds = None
    print("successfully convert array to raster")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase,val=False)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val=True
            val_set = Data.create_dataset(dataset_opt, phase,val)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    i=1
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                del train_data['date']
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    idx = 0

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')

                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        date=val_data['date']
                        del val_data['date']
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        print(visuals['SR'].size())
                        rain=visuals['SR'][0]


                        result=np.array(rain)
                        raster_file = ('./test/rain/' +'rain_forcast'+str(i)+str(date[0])+ '.tif')  # 输出的栅格文件路径
                        src_ras_file = r'E:\WFDDM\dataset\weather\temp\19900101_avg.tif'  # 提供地理坐标信息和几何信息的栅格底图
                        dataset1 = gdal.Open(src_ras_file)
                        projection1 = dataset1.GetProjection()
                        transform1 = dataset1.GetGeoTransform()
                        arr2raster(result, raster_file, prj=projection1, trans=transform1)



                        i=i+1

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]

            else:
               pass






