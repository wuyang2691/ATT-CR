import torch as t
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
# import tensorflow as tf

def cloud_mean_absolute_error(y_true, y_pred):
    """Computes the MAE over the full image."""
    return t.mean(t.abs(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :]))


def cloud_mean_squared_error(y_true, y_pred):
    """Computes the MSE over the full image."""
    return t.mean(t.square(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :]))


def cloud_root_mean_squared_error(y_true, y_pred):
    """Computes the RMSE over the full image."""
    return t.sqrt(t.mean(t.square(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :])))


def cloud_bandwise_root_mean_squared_error(y_true, y_pred):
    return t.mean(t.sqrt(t.mean(t.square(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :]), axis=[2, 3])))




def cloud_psnr(y_true, y_pred):
    """Computes the PSNR over the full image."""
    y_true = y_true[:, 0:13, :, :].clone()
    y_pred = y_pred[:, 0:13, :, :].clone()
    y_true *= 2000
    y_pred *= 2000

    
    rmse = t.sqrt(t.mean(t.square(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :])))

    return 20.0 * (t.log(10000.0 / rmse) / t.log(t.tensor(10.0)))
 


