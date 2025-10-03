# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:24:49 2020

@author: ssk
"""
import argparse
import sys
from os import path as osp
cur_path = osp.abspath(
osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
# print(cur_path)
sys.path.append(cur_path+'/ATT-CR-main')

from basicsr.models.archs.ATT_CR_Model import ATT_CR
import torch as t
from data.sen12ms_cr_test_dataset import SEN12MSCRTestDataset
import cv2
import numpy as np
from utils.sen_utils import GetQuadrupletsImg
from utils.sen_utils import SaveImg
import os
# from utils.config import config

import utils.np_metric as img_met
import utils.metrics_glf_cr as metrics_glf_cr
import numpy as np
from sewar.full_ref import mse,rmse,psnr,ssim,sam

import torch
import math as m
import yaml
"""
步骤
1 取得网络输出
2 乘以scale
3 uint16 to uint8
4 np.squeeze()压缩维度 最后得到 13*256*256图片
5 取RGB图片输出
6 将无云图片 预测图片 网络输入（有云） 合成一张图片 保存输出
7 计算图像SSIM PSNR等参数
"""

import logging
import datetime
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

logging.basicConfig(level = logging.CRITICAL,
    format = '%(asctime)s [%(levelname)s] at %(filename)s,%(lineno)d: %(message)s',
    datefmt = '%Y-%m-%d(%a)%H:%M:%S',
    filename = 'test_DSen2_CR_log_'+now+'.txt',
    filemode = 'w')

#将大于或等于INFO级别的日志信息输出到StreamHandler(默认为标准错误)
console = logging.StreamHandler()
console.setLevel(logging.CRITICAL)
formatter = logging.Formatter('[%(levelname)-8s] %(message)s') #屏显实时查看，无需时间
console.setFormatter(formatter)
logging.getLogger().addHandler(console)
    
def predict(config):
    #网络定义

    # parameters = {'inp_channels':15, 'out_channels':13, 'dim':48, 'num_blocks':[1, 2, 8, 2, 1], 'num_refinement_blocks': 2, 'heads':[2,2,8,2,2], 'scales':  [[3,5],[3,5], [1,3], [3,5], [3,5]], 'ffn_expansion_factor':2.66, 'bias':False, 'LayerNorm_type':'WithBias'}

    # net = ATT_CR(**parameters) #

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    yaml_file= config.opt_yml
    x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

    s = x['network_g'].pop('type')

    net = ATT_CR(**x['network_g'])
    net = net.eval()

    # 数据集
    dataset = SEN12MSCRTestDataset(config.predict_dataset_dir)
    dataloader = t.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    print("数据集初始化完毕，数据集大小为：{}".format(len(dataset)))
    logging.critical("test len {} ".format(len(dataset)))
    logging.critical("test batch-epoch {} ".format(len(dataloader)))

    #如果有初始化的网络路径 则初始化
    if config.net_init is not None:
        param = t.load(config.net_init) #checkpoint['params']
        net.load_state_dict(param['params'])
        print("载入{}作为网络模型".format(config.net_init))
        logging.critical("load params {}".format(config.net_init))
    else:
        print("您没有输入网络模型路径，请在Config.py中的net_init行后面加上网络路径")
        return

    #将数据装入gpu（在有gpu且使用GPU进行训练的情况下）
    cloud_img=t.FloatTensor(1, 15 ,256, 256)
    ground_truth=t.FloatTensor(1, 15 ,256, 256)
    csm_img=t.FloatTensor(1, 1, 256, 256)

    
    #如果使用GPU 则把这些放进显存里面去
    #虽然不需要CSM 但是还是把它输出一下吧
    # if config.use_gpu:
    net = net.cuda()
    cloud_img=cloud_img.cuda()
    ground_truth=ground_truth.cuda()
    csm_img=csm_img.cuda()

    MAE_vs = []
    MSE_vs = []
    RMSE_vs = []
    BRMSE_vs = []
    ssim_vs = []
    psnr_vs = []
    sam_vs = []
    logging.critical("start testing...")

    with t.no_grad():
        for iteration,batch in enumerate(dataloader,1):
            img_cld,img_csm,img_truth,patch_path_out=batch
            img_cld = cloud_img.resize_(img_cld.shape).copy_(img_cld)
            img_csm = csm_img.resize_(img_csm.shape).copy_(img_csm)
            img_truth = ground_truth.resize_(img_truth.shape).copy_(img_truth)
            img_fake = net(img_cld)


            output,img_cld_RGB,img_fake_RGB,img_truth_RGB,img_csm_RGB,img_sar_RGB,img_cld_nobright_RGB = GetQuadrupletsImg(img_cld, img_fake, img_truth, img_csm)

            if not os.path.exists(config.output_dir + '/test_img'):
                os.makedirs(config.output_dir + '/test_img')

            outfilename = patch_path_out[0].split('/')[-1].split('.')[0]
            logging.critical(outfilename)
            SaveImg(output, os.path.join(config.output_dir + '/test_img',"{}_{}_out.jpg".format(outfilename, iteration)))
            SaveImg(img_cld_RGB, os.path.join(config.output_dir + '/test_img', "{}_{}_incld_bright.jpg".format(outfilename, iteration)))
            SaveImg(img_fake_RGB, os.path.join(config.output_dir + '/test_img', "{}_{}_outfake.jpg".format(outfilename, iteration)))
            SaveImg(img_truth_RGB, os.path.join(config.output_dir + '/test_img', "{}_{}_truth.jpg".format(outfilename, iteration)))
            SaveImg(img_csm_RGB, os.path.join(config.output_dir + '/test_img', "{}_{}_csm.jpg".format(outfilename, iteration)))
            SaveImg(img_sar_RGB, os.path.join(config.output_dir + '/test_img', "{}_{}_sar.jpg".format(outfilename, iteration)))
            SaveImg(img_cld_nobright_RGB, os.path.join(config.output_dir + '/test_img', "{}_{}_incld.jpg".format(outfilename, iteration)))

           
            s2img1 = img_truth.clone()
            fake_img1 = img_fake.clone()/5 # convert values from 0-5 to 0-1
            s2img1 = s2img1/5
            MAE_v = img_met.cloud_mean_absolute_error(s2img1, fake_img1)
            MSE_v = img_met.cloud_mean_squared_error(s2img1, fake_img1)
            RMSE_v = img_met.cloud_root_mean_squared_error(s2img1, fake_img1)
            BRMSE_v = img_met.cloud_bandwise_root_mean_squared_error(s2img1, fake_img1)
           
            psnr_v = metrics_glf_cr.PSNR(s2img1 , fake_img1 )
            ssim_v = metrics_glf_cr.SSIM(s2img1 , fake_img1 )

            MAE_vs.append(np.asarray(MAE_v.cpu()))
            MSE_vs.append(np.asarray(MSE_v.cpu()))
            RMSE_vs.append(np.asarray(RMSE_v.cpu()))
            BRMSE_vs.append(np.asarray(BRMSE_v.cpu()))
            ssim_vs.append(np.asarray(ssim_v.cpu().detach().numpy()))
            psnr_vs.append(np.asarray(psnr_v))

          

	    # spectral angle mapper
            mat = s2img1 * fake_img1
            mat = torch.sum(mat, 1)
            mat = torch.div(mat, torch.sqrt(torch.sum(s2img1 * s2img1, 1)))
            mat = torch.div(mat, torch.sqrt(torch.sum(fake_img1 * fake_img1, 1)))
            sam_v = torch.mean(torch.acos(torch.clamp(mat, -1, 1)) * torch.tensor(180) / m.pi)
            sam_vs.append(np.asarray(sam_v.cpu().detach().numpy()))


            logging.critical(
                "MAE_v:{:.6f},MSE_v:{:.6f},RMSE_v:{:.6f},BRMSE_v:{:.6f},psnr_v:{:.6f},ssim_v:{:.6f},sam_v:{:.6f}".format(
                    MAE_v.cpu(), MSE_v.cpu(),
                    RMSE_v.cpu(), BRMSE_v.cpu(),
                    psnr_v, ssim_v.cpu(), sam_v))
            
        MAE_v = np.mean(MAE_vs)
        MSE_v = np.mean(MSE_vs)
        RMSE_v = np.mean(RMSE_vs)
        BRMSE_v = np.mean(BRMSE_vs)
        ssim_v = np.mean(ssim_vs)
        psnr_v = np.mean(psnr_vs)
        sam_v = np.mean(sam_vs)
       
        logging.critical(
            "MAE_m:{:.6f},MSE_m:{:.6f},RMSE_m:{:.6f},BRMSE_m:{:.6f},psnr_m:{:.6f},ssim_m:{:.6f},sam_m:{:.6f}".format(
                MAE_v, MSE_v, RMSE_v, BRMSE_v,
                psnr_v, ssim_v, sam_v))
        
if __name__=="__main__":
    # myconfig= config()  # utils.config.py file save the test parameter settings
    parser = argparse.ArgumentParser(description='Test on your own images')
    parser.add_argument('--predict_dataset_dir', default='/data/wy/CV/data/cloud/SEN12MS-CR/test', type=str, help='Directory of cloudy input images')
    parser.add_argument('--output_dir', default='./output/SEN12MS', type=str, help='Directory for restored results')
    parser.add_argument('--net_init', default='./experiments/sen12ms/models/model_best.pth', type=str, help='Path to weights')
    parser.add_argument('--opt_yml', default='./option/sen12ms.yml', type=str, help='Path to model config')
    args = parser.parse_args()
    predict(args)

        
        
        
