# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from osgeo import gdal
import joblib
import warnings
from scipy.ndimage import zoom
import math
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
import scipy.io as sio

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")


## 开始读取模型和scaler


def write_img(filename, im_proj, im_geotrans, im_data):
    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset


def read_image(image_path):
    """定义图像加载函数"""

    # 打开TIFF文件
    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    # 读取波段数据到数组
    image = dataset.ReadAsArray()

    # # 异常值处理
    # image[image < 0] = np.nan
    # image[image > 9999] = np.nan

    im_proj = dataset.GetProjection()
    im_geotrans = dataset.GetGeoTransform()

    # 关闭数据集
    dataset = None

    return image, im_proj, im_geotrans


def cz_arr(_original_array, _target_shape):

    _original_shape = (_original_array.shape[0], _original_array.shape[1])

    # 计算缩放比例
    zoom_factors = np.array(_target_shape) / np.array(_original_shape)

    # 使用scipy.ndimage.zoom进行插值
    resized_array = zoom(_original_array, zoom_factors, order=0)

    return resized_array

def raster_data_windows(i,j,input_tif,block_size):
    image = rasterio.open(input_tif)
    img = image.read(window=Window(j, i, block_size, block_size))
    # if (i==1) & (j==2):
    #     plt.imshow(img)
    return img

def process_data(x):
    yy = np.stack(x, axis=0)
    # print('y shape=',yy.shape)
    num_,nl_,ns_ = yy.shape
    feature_li = np.zeros((nl_*ns_,num_),dtype = np.float32)
    for i in range(num_):
        xi = yy[i,:,:]
        xi[xi<-100] = np.nan
        feature_li[:,i:i+1] = xi.reshape(-1,1)
    mask = feature_li[:,1].copy()
    feature_arr = np.array(feature_li)
    del feature_li
    feature_arr_s = feature_arr
    feature_arr_s[np.isnan(feature_arr_s)] = -1
    mask[feature_arr_s[:, 0] == -1] = -1
    del feature_arr
    return feature_arr_s,nl_,ns_

# if __name__ == '__main__':
def RF_predict_s(model_path,data_path,out_path,feature_name,ii):
    # model_path = r"I:/数据/大安市/DM/RF_model/"
    # data_path = r"I:/数据/大安市/data/"##栅格数据路径
    # out_path = r"I:/数据/大安市/DM/"##输出路径
    # feature_name = ['slope','SI3','SI2','SI1','SI','S5','S3','S2','NDWI','NDVI','CRSI','BI','cec','Evapotrans',
    #         'S6','S1','SI_T','temperatur','precipitat','Land_use','Elevation']##全部变量
    data_out = f'{out_path}/RF_predict_{str(ii+1).zfill(2)}.tif'
    ##
    model_n = f'model_{str(ii+1).zfill(2)}.pkl'
    scaler_n = f'scaler_{str(ii+1).zfill(2)}.pkl'


    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    # 加载模型
    scaler = joblib.load(model_path + scaler_n)
    model = joblib.load(model_path + model_n)

    block_size = 800  # 400x400 像元
    input_tif = f'{data_path}/{feature_name[0]}/{feature_name[0]}.dat' ## anyone
    ## 打开一个随机的数据，为了获取数据的尺寸和投影信息
    image = rasterio.open(input_tif)
    h = image.height
    w = image.width

    preds = rasterio.open(
        data_out,
        'w+',
        driver='GTiff',
        height=h,
        width=w,
        count=1,
        crs=image.crs,
        transform=image.transform,
        dtype=np.uint8,
    )
    output_tif = data_out

    for i in range(0, h, block_size):
        i = np.minimum(i, h - block_size)
        for j in range(0, w, block_size):
            j = np.minimum(j, w - block_size)
            # img = image.read(window=Window(j, i,block_size, block_size))
            print('i=',i,'  j=',j)
            arrays_list = []
            for nn,feature in enumerate(feature_name):
                # print(feature)
                feature_path = f'{data_path}/{feature}'
                feature_path = f'{feature_path}/{feature}.dat'
                data_ = raster_data_windows(i,j,feature_path,block_size)
                arrays_list.append(data_[0,:,:])

            feature_arr_s,nl_,ns_ = process_data(arrays_list)
            indx_ = feature_arr_s[:, 0]
            y_pred = model.predict(feature_arr_s)
            del feature_arr_s
            y_pred = y_pred.astype(np.float16)
            y_pred[indx_==-1] = np.nan
            result_ij = y_pred.reshape(nl_,ns_)
            result_ij_ = np.zeros((1,nl_,ns_),dtype=np.float32)
            result_ij_[0,:,:] = result_ij
            del result_ij
            result_ij_ = result_ij_[0,:h, :w]
            preds.write(result_ij_, window=Window(j, i, block_size, block_size), indexes=1)
    preds.close()
    print('DONE!')