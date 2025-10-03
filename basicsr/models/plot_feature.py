import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_ubyte
import cv2
import os

n_count = 0
def plot_feature_map(feature_map):
    global n_count
    n_count += 1
    feature_map = feature_map.squeeze().permute(1,2,0)
    feature_map = feature_map.detach().cpu().numpy()
    # 假设 feature_map 是 h * w * c 的特征矩阵
    h, w, c = feature_map.shape

    # 设置图形尺寸和布局
    fig, axes = plt.subplots(10, 2, figsize=(50, 50))
    axes = axes.ravel()  # 将子图变为一维数组，方便遍历

    # 遍历每个通道并可视化
    for i in range(int(c//10)):
        channel = feature_map[:, :, i*10]

        # 归一化通道
        normalized_channel = (channel - channel.min()) / (channel.max() - channel.min())

        # 绘制通道图像
        axes[i].imshow(normalized_channel, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Channel {i*10+1}')

    # 隐藏多余的子图框架
    # for i in range(c, len(axes)):
    #     axes[i].axis('off')

    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig("/data/wy/code/Mamba/org-restormer/linear_attn_uni/fig_map_out_no_gate_128/feature_map_channels_{}.png".format(n_count), bbox_inches='tight', dpi=300)
    plt.close()  # 关闭图形以释放内存

def plot_feature_map_seperate(feature_map, stage_name, img_n):
    global n_count
    n_count += 1
    feature_map = feature_map.squeeze().permute(1,2,0)
    feature_map = feature_map.detach().cpu().numpy()
    # 假设 feature_map 是 h * w * c 的特征矩阵
    h, w, c = feature_map.shape

    # # 设置图形尺寸和布局
    # fig, axes = plt.subplots(10, 2, figsize=(50, 50))
    # axes = axes.ravel()  # 将子图变为一维数组，方便遍历
    ratio = 2
    if c == 96:
        ratio = 6
    if c == 192:
        ratio = 8
    # 遍历每个通道并可视化
    for i in range(int(c//ratio)):
        channel = feature_map[:, :, i*ratio]

        # # 归一化通道
        channel = (channel - channel.min()) / (channel.max() - channel.min())
      
        folder_path = "/data/wy/CV/code/Cloud_Removal/ATT-CR-main/RICE2/epoch_last_OUT_after_gate/model_{}/img_{}/".format(stage_name, img_n)  # 文件夹路径

        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            # 创建文件夹
            os.makedirs(folder_path)
            # print(f"文件夹 '{folder_path}' 已创建。")
      

        # 绘制通道图像
        plt.imshow(channel , cmap='gray')
        plt.axis('off')  # 隐藏坐标轴
        plt.title(f'Channel {i*ratio}')
        plt.savefig(os.path.join(folder_path,"channel_{}.png".format(i*ratio)), bbox_inches='tight', pad_inches=0)
        plt.close()
     

 


def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def plot_output_last_decoder(output_tensor, path='/data/wy/code/Mamba/org-restormer/linear_attn_uni/decoder_output/last_decoder_output'):
    global n_count
    n_count += 1

    # 假设 output_tensor 是 3 * h * w 的张量
    # 将其转换为 numpy 格式，并将通道维度移动到最后
    output_image = output_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    output_image = img_as_ubyte(output_image)

    save_img(("/data/wy/code/Mamba/org-restormer/linear_attn_uni/rice1_decoder_output/last_decoder_output_{}.png".format(n_count)), output_image)
    

    # 可视化图像
    # plt.imshow(output_image)
    # plt.savefig("/data/wy/code/Mamba/org-restormer/linear_attn_uni/fig_map_2/feature_map_channels_{}.png".format(n_count), bbox_inches='tight', dpi=300)
    # plt.close()  # 关闭图形以释放内存

pool_out = 0
out_mult_gate =0

def feature_pool_vis(feature_tensor, folder, save_name, type='out', ):
    global pool_out
    global out_mult_gate
    if type == 'gate':
        # feature_tensor 是 c * h * w 的张量 对所有通道使用平均池化
        pooled_image = feature_tensor.squeeze(0).mean(dim=0).detach().cpu().numpy()
        # pooled_image = feature_tensor.squeeze().max(dim=0)[0].cpu().numpy()
         # 归一化通道
        pooled_image = (pooled_image - pooled_image.min()) / (pooled_image.max() - pooled_image.min())

        folder_path = "/data/wy/CV/code/Cloud_Removal/ATT-CR-main/RICE2-one-dim-gate/rice2_norm/gate_mean/S_{}".format(folder)
        if not os.path.exists(folder_path):
            # 创建文件夹
            os.makedirs(folder_path)
        save_path = os.path.join(folder_path,"{}.png".format(save_name))
        # 可视化并保存
        plt.imshow(pooled_image, cmap='gray')
        plt.axis('off')  # 隐藏坐标轴
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()


    elif type == 'gate*out':
        out_mult_gate +=1
        # feature_tensor 是 c * h * w 的张量 对所有通道使用平均池化
        pooled_image = feature_tensor.squeeze(0).mean(dim=0).detach().cpu().numpy()
        # pooled_image = feature_tensor.squeeze().max(dim=0)[0].cpu().numpy()
          # 归一化通道
        pooled_image = (pooled_image - pooled_image.min()) / (pooled_image.max() - pooled_image.min())
        # 可视化并保存
        plt.imshow(pooled_image, cmap='gray')
        plt.axis('off')  # 隐藏坐标轴
        folder_path = "/data/wy/CV/code/Cloud_Removal/ATT-CR-main/RICE2-one-dim-gate/rice2_norm/out_after_gate_mean/S_{}".format(folder)
        save_path = os.path.join(folder_path,"{}.png".format(save_name))
       
        if not os.path.exists(folder_path):
            # 创建文件夹
            os.makedirs(folder_path)
            # print(f"文件夹 '{folder_path}' 已创建。")
            
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
    
    else:   
        pool_out += 1
        # feature_tensor 是 c * h * w 的张量 对所有通道使用平均池化
        pooled_image = feature_tensor.squeeze(0).mean(dim=0).detach().cpu().numpy()
      
          # 归一化通道
        pooled_image = (pooled_image - pooled_image.min()) / (pooled_image.max() - pooled_image.min())
        # 避免最顶部或底部像素过亮导致白边
        


        #   # Step 3: 可选：平滑处理（只为了可视化更平滑）
        # pooled_image = cv2.GaussianBlur(pooled_image, (3, 3), 0)

        # # Step 4: 转换为 uint8
        # pooled_uint8 = (pooled_image * 255).astype(np.uint8)

        # # Step 5: 保存路径
        # folder_path = f"/data/wy/CV/code/Cloud_Removal/ATT-CR-main/ELU_gate_vis/rice1_norm/out_mean_before_gate/S_NO_LINEAR_{folder}"
        # os.makedirs(folder_path, exist_ok=True)
        # save_path = os.path.join(folder_path, f"{save_name}.png")

        # # Step 6: 保存图像（建议用 cv2.imwrite 避免 matplotlib 压缩和插值问题）
        # cv2.imwrite(save_path, pooled_uint8)

        # Step 7: 可选——保存一份伪彩色图像（对比更清楚）
        # color_path = save_path.replace(".png", "_color.png")
        # pooled_color = cv2.applyColorMap(pooled_uint8, cv2.COLORMAP_JET)
        # cv2.imwrite(color_path, pooled_color)

        folder_path = "/data/wy/CV/code/Cloud_Removal/ATT-CR-main/RICE2-one-dim-gate/rice2_norm/out_mean_before_gate/S_{}".format(folder)
        if not os.path.exists(folder_path):
            # 创建文件夹
            os.makedirs(folder_path)
        save_path = os.path.join(folder_path,"{}.png".format(save_name))
        # 可视化并保存
        plt.imshow(pooled_image, cmap='gray')
        plt.axis('off')  # 隐藏坐标轴
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()




def feature_pool_conv_vis(feature_tensor):
    global conv_out
    conv_out += 1
    # 假设 feature_tensor 是 c * h * w 的张量
    # 对所有通道使用平均池化
    # pooled_image = feature_tensor.mean(dim=0).detach().cpu().numpy()
    pooled_image = feature_tensor.squeeze().mean(dim=0).detach().cpu().numpy()
     # 归一化通道
    # pooled_image = (pooled_image - pooled_image.min()) / (pooled_image.max() - pooled_image.min())

    # 可视化并保存
    plt.imshow(pooled_image, cmap='gray')
    plt.axis('off')  # 隐藏坐标轴
    plt.savefig("/data/wy/code/Mamba/org-restormer/MS_Linear_Uni_Attention/gate_vis/rice1/pool_gate/pooled_gate_{}.png".format(conv_out), bbox_inches='tight', pad_inches=0)
    plt.close()




import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_fsgm_channel_distribution(gate_values, num_channels=64, save_name=None ):
    """
    显示 FSGM 门控模块中每个通道的门控值分布。
    :param gate_values: FSGM门控值的张量，形状为 (batch_size, height, width, num_channels)
    :param num_channels: 显示的通道数量
    """
    # 假设 gate_values 是一个形状为 (batch_size, height, width, num_channels) 的张量
    # 将其转换为 NumPy 数组，计算每个通道的均值或其他统计值
    gate_values = gate_values.permute(0,2,3,1)
    gate_values = gate_values.cpu().detach().numpy()
    
    # 计算每个通道的均值或方差，这里以均值为例
    gate_mean = np.mean(gate_values, axis=(0, 1, 2))  # 对 batch_size, height, width 进行求均值
    
    # 可视化门控值的分布
    plt.figure(figsize=(10, 6))
    plt.hist(gate_mean, bins=30, alpha=0.7, color='b', edgecolor='black')
    plt.title(f'FSGM Channel Gate Values Distribution (mean)')
    plt.xlabel('Gate Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    
    folder_path = "/data/wy/CV/code/Cloud_Removal/ATT-CR-main/gate_vis/channel_distributio_epoch_100"  # 文件夹路径

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        # 创建文件夹
        os.makedirs(folder_path)
    save_path = os.path.join(folder_path,"{}.png".format(save_name))
    if save_path:
        plt.savefig(save_path)  # 保存图像到指定路径
        print(f"Image saved to {save_path}")
    else:
        plt.show()  # 如果没有指定保存路径，则显示图像


def plot_fsgm_channel_distribution_curve(gate_values, num_channels=64, save_name=None):
    """
    显示 FSGM 门控模块中每个通道的门控值分布曲线，并保存到指定路径。
    :param gate_values: FSGM门控值的张量，形状为 (batch_size, height, width, num_channels)
    :param num_channels: 显示的通道数量
    :param save_path: 保存图片的路径，若为 None，则不保存
    """
    # 假设 gate_values 是一个形状为 (batch_size, height, width, num_channels) 的张量
    # 将其转换为 NumPy 数组，计算每个通道的均值或其他统计值
    # gate_values =  gate_values.permute(0,2,3,1).cpu().detach().numpy()

   
    
    # 计算通道的均值，这里以均值为例
   
    gate_values =  gate_values.cpu().detach().numpy()
    gate_mean = np.mean(gate_values, axis=(1))  # 对channel进行求均值 b, h,w,
    gate_mean =  gate_mean.flatten()
    min_val = gate_mean.min()
    max_val = gate_mean.max()

    gate_mean = (gate_mean - min_val) / (max_val - min_val + 1e-8)  # 加个小常数
#  # 计算每个位置的均值，这里以均值为例
    # gate_values = np.transpose(gate_values, (0, 2, 3, 1))
#     gate_mean = np.mean(gate_values, axis=(0,3))
#   

    # gate_mean =  gate_mean.flatten()
    # 可视化门控值的分布曲线
    plt.figure(figsize=(10, 6))
    sns.kdeplot(gate_mean, shade=True, color='b', alpha=0.7)
    plt.title(f'FSGM Channel Gate Values Distribution Curve (mean)')
    plt.xlabel('Gate Value')
    plt.ylabel('Density')
    plt.grid(True)
    

    folder_path = "/data/wy/CV/code/Cloud_Removal/ATT-CR-main/gate_vis/single_img/distribution_epoch_last/new_rice1"  # 文件夹路径

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        # 创建文件夹
        os.makedirs(folder_path)
    save_path = os.path.join(folder_path,"{}.png".format(save_name))
    if save_path:
        plt.savefig(save_path)  # 保存图像到指定路径
        print(f"Image saved to {save_path}")
    else:
        plt.show()  # 如果没有指定保存路径，则显示图像



def plot_fsgm_single_channel_distribution(gate_values, channel_index=0, save_name=None):
    """
    显示 FSGM 门控模块中指定通道的门控值分布曲线，并保存到指定路径。
    :param gate_values: FSGM门控值的张量，形状为 (batch_size, height, width, num_channels)
    :param channel_index: 要分析的通道的索引（例如0表示第一个通道）
    :param save_path: 保存图片的路径，若为 None，则不保存
    """
    # 假设 gate_values 是一个形状为 (batch_size, height, width, num_channels) 的张量
    # 将其转换为 NumPy 数组，选择指定的通道
    gate_values = gate_values.cpu().detach().numpy()
    gate_values = np.transpose(gate_values, (0, 2, 3, 1))
    
    # 提取指定通道的数据
    channel_values = gate_values[:, :, :, channel_index]  # 选择特定通道的数据，形状为 (batch_size, height, width)
    
    # 计算该通道的均值（也可以选择方差、最大值等）
    # channel_mean = np.mean(channel_values, axis=(0, 1, 2))  # 对 batch_size, height, width 进行求均值
    channel_flatten = channel_values.flatten()
    # 可视化该通道的门控值分布曲线
    plt.figure(figsize=(10, 6))
    sns.kdeplot(channel_flatten, shade=True, color='b', alpha=0.7)
    plt.title(f'FSGM Channel {channel_index} Gate Value Distribution Curve (mean)')
    plt.xlabel('Gate Value')
    plt.ylabel('Density')
    plt.grid(True)


    folder_path = "/data/wy/CV/code/Cloud_Removal/ATT-CR-main/gate_vis/channel_distributio_epoch_last"  # 文件夹路径

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        # 创建文件夹
        os.makedirs(folder_path)
    save_path = os.path.join(folder_path,"{}.png".format(save_name))
    if save_path:
        plt.savefig(save_path)  # 保存图像到指定路径
        print(f"Image saved to {save_path}")
    else:
        plt.show()  # 如果没有指定保存路径，则显示图像













def gate_tensor_save(x_p, file_name):
    # 将表征保存到文件
    features_np =  x_p.cpu().numpy()  # 转换为 NumPy 数组
    # features_np = freq_emb.cpu().numpy()  # 转换为 NumPy 数组
    # np.save('denoise_50_features.npy', features_np)  # 保存表征到文件
    folder_path = '/data/wy/CV/code/Cloud_Removal/ATT-CR-main/gate_val_numpy'
    if not os.path.exists(folder_path):
        # 创建文件夹
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path,"{}".format(file_name))
    try:
        existing_data = np.load(file_path)
    # 合并现有数据和新数据
        combined_data = np.concatenate((existing_data, features_np), axis=0)
    except FileNotFoundError:
    # 如果文件不存在，直接使用新的数据
        combined_data = features_np
    np.save(file_path, combined_data)  # 追加保存表征到文件






def load_npy_files(file_paths):
    """
    读取多个 .npy 文件，并将每个文件的数据存储到对应变量。
    
    :param file_paths: list of str, 包含 .npy 文件路径的列表
    :return: dict, 文件名和其数据的字典
    """
    data_dict = {}
    
    for i, file_path in enumerate(file_paths):
        # 提取文件名（去掉路径和扩展名）
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 读取 .npy 文件
        data = np.load(file_path)
        plot_fsgm_channel_distribution_curve(data, save_name=f'stage_{i}')
        # 将数据存储到字典中，使用文件名作为键
        # data_dict[file_name] = data
        
        # # 可选：如果您想动态创建变量名，可以使用 globals()
        # globals()[file_name] = data
    
    return data_dict
if __name__ == '__main__':
    file_paths = ['/data/wy/CV/code/Cloud_Removal/ATT-CR-main/gate_val_numpy/stage_1.npy', '/data/wy/CV/code/Cloud_Removal/ATT-CR-main/gate_val_numpy/stage_2.npy', '/data/wy/CV/code/Cloud_Removal/ATT-CR-main/gate_val_numpy/stage_3.npy',
                '/data/wy/CV/code/Cloud_Removal/ATT-CR-main/gate_val_numpy/stage_4.npy', '/data/wy/CV/code/Cloud_Removal/ATT-CR-main/gate_val_numpy/stage_5.npy']
    data_dict = load_npy_files(file_paths)