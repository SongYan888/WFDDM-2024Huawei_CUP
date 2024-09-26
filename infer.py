import torch
from sklearn import preprocessing

import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
from osgeo import gdal
import datetime
from dateutil.relativedelta import relativedelta

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
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_64_512.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    
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
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase,val=True)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _,  val_data in enumerate(val_loader):
        idx += 1
        date = val_data['date']
        del val_data['date']
        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals(need_LR=False)
        Rain=val_data['Rain']
        Rain1=val_data['Rain1']



        visual=visuals['SR'].view([2,144,256])
        rain = visual[0]

        flag=np.array(val_data['Rain'].view([144,256]).cpu())
        result = np.array(rain)
        raster_file = ('./test/rain/' + 'rain_forcast' + str(date[0]) + '.tif')  # 输出的栅格文件路径
        src_ras_file = r'E:\WFDDM\dataset\weather\temp\19900101_avg.tif'  # 提供地理坐标信息和几何信息的栅格底图
        dataset1 = gdal.Open(src_ras_file)
        projection1 = dataset1.GetProjection()
        transform1 = dataset1.GetGeoTransform()
        arr2raster(result, raster_file, prj=projection1, trans=transform1)


        rain = visuals['SR'].view([2,144,256])[0]

        index=1
        date=date[0]
        for i in range(7300):

            d = datetime.datetime.strptime(date, '%Y-%m-%d')
            date=(d + relativedelta(days=1)).strftime('%Y-%m-%d')
            le = preprocessing.LabelEncoder()
            rain=np.array(rain.view(144,256).cpu())
            temp = np.array(temp.view(144, 256).cpu())
            for i in range(144):
                for j in range(256):
                    if flag[i][j] == -99:
                        rain[i][j] = -99

            rain=torch.tensor(rain).view([1,1,144,256]).to(device='cuda')

            target = le.fit_transform([date])
            a_targets = torch.as_tensor(target).expand(1,1, 144, 256)
            diffusion.feed_data({'label':rain.view([1,1,144,256])
                                    ,'Rain':Rain1.view([1,1,144,256]),
                                 'Rain1':rain.view([1,1,144,256]),

                                 'Index': torch.tensor(index),'date':torch.tensor(1),'emb':a_targets})
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals(need_LR=False)
            rain1= np.array(visuals['SR'].view([2,144,256])[0].cpu())

            for i in range(144):
                for j in range(256):
                    if flag[i][j]==-99:
                        rain1[i][j]=-99


            result = rain1
            raster_file = ('./test/rain/' + 'rain_forcast' + str(date) + '.tif')  # 输出的栅格文件路径
            src_ras_file = r'E:\WFDDM\dataset\weather\rain\precipitation_1990-01-01.tiff'  # 提供地理坐标信息和几何信息的栅格底图
            dataset1 = gdal.Open(src_ras_file)
            projection1 = dataset1.GetProjection()
            transform1 = dataset1.GetGeoTransform()
            arr2raster(result, raster_file, prj=projection1, trans=transform1)


            Rain1 = rain


            rain=torch.tensor(rain1).view([1,1,144,256]).to(device='cuda')
