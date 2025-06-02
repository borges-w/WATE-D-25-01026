import numpy as np
from scipy import stats
from Xgboost_train import *
from xgboost_predict import *
import os
import rasterio
from rasterio.windows import Window


if __name__ == "__main__":
    filen = "I:/数据/大安市/DA2023_point/2023DA.csv"
    model_path = r"I:/数据/大安市/DA_model/shap/XGB/"
    out_path  = r"I:/数据/大安市/DA_model/shap/XGB/"     ##输出路径
    data_path =  r"I:\数据\大安市\2023data/"     ##栅格数据路
    outpath_shap_fig = r"I:/数据/大安市/DA_model/shap/XGB/"   ###shap画图保存路径
    fig_indx = 0 ###第31次训练的时候 进行shap分析/画图

    cols = ['Albedo', 'TEMP_MIN', 'TEMP_MAX', 'TEMP', 'SSR', 'SRAD', 'slope', 'SI3', 'SI2', 'SI1',
            'SI_T', 'SI', 'SAVI', 'S6', 'S5', 'S3', 'S2', 'S1', 'RVI', 'Roughness', 'RND', 'PRCP',
            'POP', 'PM2_5', 'NDWI', 'NDVI', 'MAP', 'LU', 'GWP', 'FVC', 'EVI', 'ET', 'Elevation', 'DVI',
            'DMSP', 'CRSI', 'BI', 'ANNAP', 'AI']
    # 1.train
    for ii in range(50):
        print('当前在训练:',str(ii+1).zfill(2))
        run_main_xgboost_train(filen,model_path,cols,ii,outpath_shap_fig, fig_indx)
        plt.close('all')
    # # 2.predict
    # for jj in range(50):
    #     print('当前在预测:',str(jj+1).zfill(2))
    #     xgboost_predict_s(model_path,data_path,out_path,cols,jj)

    # 3.calculate uncertainty
    outname = f'{out_path}/uncertainty.tif'
    result_path0 = os.path.join(out_path, 'xgboost_predict_01.tif')
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
                feature_path = f'{out_path}/xgboost_predict_{str(kk + 1).zfill(2)}.tif'
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
    


    # for ii in range(win_nl):
    #     for jj in range(win_ns):
    #         xx_indx1 = ii * 400
    #         yy_indx1 = jj * 400
    #         xx_indx2 = (ii + 1) * 400
    #         yy_indx2 = (jj + 1) * 400
    #         if xx_indx2 > (h - 1):
    #             xx_indx2 = h - 1
    #             xx_indx1 = xx_indx2 - 400
    #         if yy_indx2 > (w - 1):
    #             yy_indx2 = w - 1
    #             yy_indx1 = yy_indx2 - 400
    #         tmp = np.zeros((50, 400, 400), dtype=np.float32)
    #         for kk in range(50):
    #             feature_path = f'{out_path}/xgboost_predict_{str(kk + 1).zfill(2)}.tif'
    #             print(feature_path)
    #             image, _, _ = read_image(feature_path)
    #             print('==', xx_indx1, xx_indx2, yy_indx1, yy_indx2)
    #             image1 = image[xx_indx1:xx_indx2, yy_indx1:yy_indx2]
    #             tmp[kk, :, :] = image1
    #
    #         sample_mean = np.mean(tmp, axis=0)
    #         sample_std = np.std(tmp, axis=0)
    #         sample_size = tmp.shape[0]
    #         se = sample_std / np.sqrt(sample_size)
    #         confidence_level = 0.95
    #         h1 = stats.t.ppf((1 + confidence_level) / 2., sample_size - 1)
    #
    #         # 计算置信区间
    #         lower_bound = sample_mean - h1 * se
    #         upper_bound = sample_mean + h1 * se
    #
    #         # 检查是否有任何元素小于等于零
    #         if np.any(lower_bound <= 0):
    #             une = 0  # 或者其他处理方法
    #         else:
    #             une = (upper_bound / lower_bound) / sample_mean
    #
    #         result[xx_indx1:xx_indx2, yy_indx1:yy_indx2] = une

    # write_img(f'{out_path}/uncertainty.tif', im_proj, im_geotrans, result)

