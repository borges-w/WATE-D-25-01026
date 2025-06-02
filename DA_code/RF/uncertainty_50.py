import numpy as np
from scipy import stats
from RF_train import *
from RF_predict import *
import os
import rasterio
from rasterio.windows import Window


if __name__ == "__main__":
    filen = r"I:/数据/大安市/DA2023_point/2023DA.csv"
    model_path = r"I:/数据/大安市/DA_model/shap/RF/"
    out_path = r"I:/数据/大安市/DA_model/shap/RF/"##输出路径
    data_path = r"I:\数据\大安市\2023data" ##栅格数据路径

    outpath_shap_fig = r"I:/数据/大安市/DA_model/shap/RF/"   ###shap画图保存路径
    fig_indx = 0 ###第31次训练的时候 进行shap分析/画图

    cols = ['Albedo', 'TEMP_MIN', 'TEMP_MAX', 'TEMP', 'SSR', 'SRAD', 'slope', 'SI3', 'SI2', 'SI1',
            'SI_T', 'SI', 'SAVI', 'S6', 'S5', 'S3', 'S2', 'S1', 'RVI', 'Roughness', 'RND', 'PRCP',
            'POP', 'PM2_5', 'NDWI', 'NDVI', 'MAP', 'LU', 'GWP', 'FVC', 'EVI', 'ET', 'Elevation', 'DVI',
            'DMSP', 'CRSI', 'BI', 'ANNAP', 'AI']
    for ii in range(50):
        print('当前在训练:',str(ii+1).zfill(2))
        run_main_RF_train(filen,model_path,cols,ii,outpath_shap_fig,fig_indx)
    ##
    for jj in range(50):
        print('当前在预测:',str(jj+1).zfill(2))
        RF_predict_s(model_path,data_path,out_path,cols,jj)

    # 3.calculate uncertainty
    outname = f'{out_path}/uncertainty.tif'
    result_path0 = os.path.join(out_path, 'RF_predict_01.tif')
    ## set the size of windows
    block_size = 800

    image = rasterio.open(result_path0)
    h = image.height
    w = image.width
    une_out = rasterio.open(
        outname,
        'w+',
        driver='GTiff',
        height=h,
        width=w,
        count=1,
        crs=image.crs,
        transform=image.transform,
        dtype=np.uint8,
    )

    for i in range(0, h, block_size):
        i = np.minimum(i, h - block_size)
        for j in range(0, w, block_size):
            j = np.minimum(j, w - block_size)

            arrays_list = []
            for kk in range(50):
                print("不确定度", str(kk + 1).zfill(2))
                feature_path = f'{out_path}/RF_predict_{str(kk + 1).zfill(2)}.tif'
                data_ = raster_data_windows(i, j, feature_path, block_size)
                arrays_list.append(data_[0, :, :])
            tmp = np.stack(arrays_list, axis=0)
            sample_mean = np.mean(tmp, axis=0)
            sample_std = np.std(tmp, axis=0)
            sample_size = tmp.shape[0]
            se = sample_std / np.sqrt(sample_size)
            confidence_level = 0.95
            h1 = stats.t.ppf((1 + confidence_level) / 2., sample_size - 1)
            confidence_interval = (sample_mean - h1 * se, sample_mean + h1 * se)
            une = ((sample_mean + h1 * se) / (sample_mean - h1 * se)) / sample_mean
            une = une[:h, :w]
            une_out.write(une, window=Window(j, i, block_size, block_size), indexes=1)
    une_out.close()

