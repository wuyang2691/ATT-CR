# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:43:07 2020

@author: ssk
"""


class config():
    #datasets config 数据集设定
    train_datset_dir = "/data/wy/CV/data/cloud/SEN12MS-CR/train" #训练数据集路径 自动分割成验证集和训练集
    predict_dataset_dir ="/data/wy/CV/data/cloud/SEN12MS-CR/test"#要预测的图片的路径
    val_dataset_dir = "/data/wy/CV/data/cloud/SEN12MS-CR/val"
    width= 256 #图片大小
    height= 256
    threads= 0 #num_of_workers 在windows环境下大于0会有BUG——>broken pipe
    
	#outputimg dir 输出图片路径
    output_dir = "./output_imgs"
    net_state_dict_save_dir = './net_state_dict/'
    
    # #train options 训练设置
    # use_gpu = True #要不要用GPU？
    # save_frequency = 5000#5000 #多少个iteration保存一次网络呢？
    
    # show_freq = 100 #50多少轮进行测试并展示成果？
    # train_size = 0.8#训练集占总数据集合的百分比
    batch_size = 1 # batchsize 越大占用显存越多
    #网络初始化pth路径
   
    net_init = '/data/wy/CV/code/Cloud_Removal/ATT-CR-main/experiments/sen12ms/models/net_g_165126.pth'

    # gpu_ids=[0] #GPU ID
    # epoch = 200    #训练轮数 paper code 8, 200 to big
    # lr= 7e-5 #学习率
    # beta1= 0.9#ADAM优化器的beta1
    # in_ch= 15#输入通道数
    # out_ch = 13#输出通道数
    # alpha =0.1#resnet aplha

    # feature_sizes=256#resnetfeature
	
