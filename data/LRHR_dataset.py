import os
from io import BytesIO
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import tifffile as tf
import glob
import torch
import re
import datetime
from sklearn import preprocessing
from dateutil.relativedelta import relativedelta

def extract_date_from_filename(filename):
    pattern=r'\d{4}-\d{2}-\d{2}'#匹路YYY-MM-DD糌式的日闕
    match =re.search(pattern,filename)
    if match:
        a=match.group()
    else:
        pass
    d = datetime.datetime.strptime(a, '%Y-%m-%d')
    return((d + relativedelta(days=1)).strftime('%Y-%m-%d'))


class WeatherDataset(Dataset):
    def __init__(self,  split='train', data_len=-1, need_LR=False, val=None,window_size=1):
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        self.val=val
        self.t=None
        self.window_size = window_size
        if self.val==False:
            self.t='\weather'
        else:
            self.t='\weather1'
        self.datatype = 'tif'

        self.labelr_path ='.\dataset'+self.t+'\\rainla'
        self.labelr_path=glob.glob(self.labelr_path+'\*.tiff')

        self.labelt_path = '.\dataset' + self.t + '\\templa'
        self.labelt_path = glob.glob(self.labelt_path + '\*.tif')

        self.rain_path1 = '.\dataset'+self.t+'\\rain'
        self.rain_path = glob.glob(self.rain_path1 + '\*.tiff')


        self.date=None
        self.dataset_len=len(self.rain_path)
        self.data_len=self.dataset_len
        # if self.data_len <= 0:
        #     self.data_len = self.dataset_len
        # else:
        #     self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        if index<=1:
            index=1
        else:
            index=index
        labelr = tf.imread(self.labelr_path[index-2])

        Rain=tf.imread(self.rain_path[index])

        Rain1=tf.imread(self.rain_path[index-1])



        date=extract_date_from_filename(self.rain_path[index])
        le=preprocessing.LabelEncoder()

        target=le.fit_transform([date])
        a_targets = torch.as_tensor(target).expand(1,144,256)

        [labelr,Rain,Rain1,a_targets] = Util.transform_augment([labelr,Rain,Rain1,np.array(a_targets)], split=self.split, min_max=(-1, 1))
        return {'label':labelr,'Rain':Rain,'Rain1':Rain1,
            'Index': index,'date':date,"emb":a_targets}

