## ************************Random Forest**********************
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from scipy.ndimage.interpolation import zoom
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFE
import numpy as np
from sklearn import preprocessing, metrics, svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import pickle
import shap
## *************
# '''↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓自动设置模型参数部分↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓'''
# #对如下6个超参数进行随机搜索
# #参数设置
def parameter_optimization(X_train, y_train,X_validation,y_validation):
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
    max_features = ['log2', 'sqrt']
    max_depth = [int(x) for x in np.linspace(start = 1, stop = 15, num = 15)]
    min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 50, num = 10)]
    min_samples_leaf = [int(x) for x in np.linspace(start = 2, stop = 50, num = 10)]
    bootstrap = [True, False]
    param_dist = {'n_estimators': n_estimators,'max_features': max_features,
            'max_depth': max_depth,'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,'bootstrap': bootstrap}#这里应该涵盖了所有常用参数，如有需要再添加
    #参数调整
    RFC = RandomForestRegressor()
    RFC.fit(X_train, y_train)#把低分辨率数据放模型里跑，得出参数最优结果
    RS = RandomizedSearchCV(RFC, param_dist, n_iter=100, cv=3, verbose=1,n_jobs=8, random_state=0)
    RS.fit(X_train, y_train)
    print(RS.best_params_) # 打印随机搜索的最佳参数
    # '''↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑自动设置模型参数部分↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑'''
    cs = RS.best_params_
    #3.3构建模型并运行
    clf = RandomForestRegressor(n_estimators=cs['n_estimators'], min_samples_split=cs['min_samples_split'], min_samples_leaf=cs['min_samples_leaf'], max_features=cs['max_features'])#设置随机森林参数，是个玄学，随便设置设置，用的这个
    clf_new = clf
    clf_new.fit(X_train, y_train)#训练集来训练模型
    y_train_pred = clf_new.predict(X_train)
    y_test_pred = clf_new.predict(X_validation)

    plt.scatter(y_test_pred, y_validation)
    plt.title('Scatter Plot Example')
    plt.xlabel('predict')
    plt.ylabel('real')

    train_mse = mean_squared_error(y_train_pred.flatten(), y_train.flatten())
    test_mse = mean_squared_error(y_test_pred.flatten(), y_validation.flatten())
    train_r2 = cal_r2(y_train_pred.flatten(),y_train.flatten())
    test_r2 = cal_r2(y_test_pred.flatten(), y_validation.flatten())
    print("(before)Training MSE:", train_mse)
    print("(before)Testing MSE:", test_mse) 
    print("(before)Training R^2:", train_r2)
    print("(before)Testing R^2:", test_r2) 
    return clf_new
## ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
###
def cal_r2(y_train_pred,y_train):
    x1 = np.sum((y_train_pred-y_train)**2)
    x2 = np.sum((y_train-np.mean(y_train))**2)
    print('x1=')
    r2 = 1-(x1/x2)
    return r2
# 读取数据
## pH	slope	SI3	SI2	SI1	SI	S6	S5	S3	S2	NDWI	NDVI	LY	JY	dem	CRSI	BI	cec
def read_traindata(filen,normal_indx,split_scale):
    data_train = pd.read_csv(filen)
    # normal_indx = 0
    cols = ['Albedo', 'TEMP_MIN', 'TEMP_MAX', 'TEMP', 'SSR', 'SRAD', 'slope', 'SI3', 'SI2', 'SI1',
            'SI_T', 'SI', 'SAVI', 'S6', 'S5', 'S3', 'S2', 'S1', 'RVI', 'Roughness', 'RND', 'PRCP',
            'POP', 'PM2_5', 'NDWI', 'NDVI', 'MAP', 'LU', 'GWP', 'FVC', 'EVI', 'ET', 'Elevation', 'DVI',
            'DMSP', 'CRSI', 'BI', 'ANNAP', 'AI']
    x = data_train[cols].values
    y = data_train['SSC'].values
    X_train, X_validation, y_train, y_validation = train_test_split(x, y, test_size=split_scale, random_state=42)
    ##
    if normal_indx == 1:
        ss_X = preprocessing.StandardScaler()# 标准化处理
        ss_Y = preprocessing.StandardScaler()
        X_train_scaled = ss_X.fit_transform(X_train)
        y_train_scaled = ss_Y.fit_transform(y_train.reshape(-1, 1))
        # print(X_train_scaled)
        X_validation_scaled = ss_X.transform(X_validation)
        y_validation_scaled = ss_Y.transform(y_validation.reshape(-1, 1))
        X_train, X_validation, y_train, y_validation = X_train_scaled, X_validation_scaled, y_train_scaled, y_validation_scaled
    else:

        print('未进行标准化处理')
    return X_train, X_validation, y_train, y_validation



# if __name__=="__main__":
def run_main_RF_train(filen,model_path,cols,ii,outpath_fig,fig_indx):

    # filen = 'I:/数据/大安市/600点/point_600.csv'
    # model_path = "I:/数据/大安市/DM/RF_model"
    # cols = ['slope','SI3','SI2','SI1','SI','S5','S3','S2','NDWI','NDVI','CRSI','BI','cec','Evapotrans',
    #         'S6','S1','SI_T','temperatur','precipitat','Land_use','Elevation']
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    normal_indx = 0       ## 是否进行标准化处理
    split_scale = 0.2     ## 数据集划分比列，test_data = 0.2
    X_train, X_validation, y_train, y_validation = read_traindata(filen,normal_indx,split_scale)
    model = parameter_optimization(X_train, y_train, X_validation, y_validation)
    ## shap
    clf = model  ##
    ## 此处加入shap分析
    ## 1.数据转成df格式
    if ii==fig_indx:
        x_train_df = pd.DataFrame(X_train, columns=cols)
        explainer = shap.Explainer(model, feature_names=cols)
        shap_values = explainer(x_train_df)
        
        plt.figure(1)
        shap.plots.bar(shap_values)
        # plt.show() 
        plt.savefig(outpath_fig+'figure1.png', dpi=300, bbox_inches='tight')  # 保存为PNG，分辨率300DPI
        plt.close()
        ##
        plt.figure(2)
        shap.summary_plot(shap_values, x_train_df, plot_type="bar")
        # plt.show() 
        plt.savefig(outpath_fig+'figure2.png', dpi=300, bbox_inches='tight')  # 保存为PNG，分辨率300DPI
        plt.close()
        ## 
        plt.figure(3)
        shap.plots.heatmap(explainer(x_train_df))
        # plt.show() 
        plt.savefig(outpath_fig+'figure3.png', dpi=300, bbox_inches='tight')  # 保存为PNG，分辨率300DPI
        plt.close()
    ## ==========================================================
    selector1 = RFE(model, n_features_to_select=20, step=1).fit(X_train, y_train)
    # n_features_to_select表示筛选最终特征数量，step表示每次排除一个特征
    selector1.support_.sum()
    #计算在 RFE 过程中被选中的特征数量，即布尔数组中值为 True 的个数，也就是最终选择的特征数量。 
    print(selector1.ranking_)
    #这个属性返回的是特征的排名，从1开始，表示每个特征在所有特征中的重要性排名，1为最重要的特征。                                            
    print(selector1.n_features_)  
    #这是RFE在执行完所有递归步骤后最终选择的特征数量。    
    y_train_pred = selector1.predict(X_train)
    y_test_pred = selector1.predict(X_validation)

    # 计算训练集和测试集的均方误差
    train_mse = mean_squared_error(y_train.flatten(), y_train_pred.flatten())
    test_mse = mean_squared_error(y_validation.flatten(), y_test_pred.flatten())
    train_r2 = cal_r2(y_train_pred.flatten(),y_train.flatten())
    test_r2 = cal_r2(y_test_pred.flatten(), y_validation.flatten())

    print("(after)Training MSE:", train_mse)
    print("(after)Testing MSE:", test_mse)
    print("(after)Training R^2:", train_r2)
    print("(after)Testing R^2:", test_r2)  
    ## 打印出选择的特征要素
    selected_features = [cols[i] for i in range(len(cols)) if selector1.support_[i]]
    print('特征递归消除后的参数：',selected_features)
    plt.scatter(y_test_pred, y_validation)
    plt.title('Scatter Plot Example')
    plt.xlabel('predict')
    plt.ylabel('real')
    scaler = preprocessing.StandardScaler()
    # 将scaler保存到文件
    model_n = f'model_{str(ii+1).zfill(2)}.pkl'
    scaler_n = f'scaler_{str(ii+1).zfill(2)}.pkl'
    with open(model_path + '\\' + scaler_n, 'wb') as f:
        pickle.dump(scaler, f)

    # 将模型保存到文件
    with open(model_path + '\\'+ model_n, 'wb') as f:
        pickle.dump(selector1, f)

    print('DONE!')