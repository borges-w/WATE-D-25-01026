## xgboost,rf,svm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn import preprocessing, metrics, svm
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
import pickle
import os
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import shap
from matplotlib import rcParams
# 设置matplotlib的字体
config = {"font.family":'SimHei', "font.size": 20, "mathtext.fontset":'stix'}
rcParams.update(config)
rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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
    X_train, X_validation, y_train, y_validation = train_test_split(x, y, test_size=split_scale)#, random_state=42)
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

def parameter_optimization(X_train, y_train):
    ## ==============================模型参数调整=================================
    ## 参数调优
    ## （1）迭代次数调优
    cv_params = {'n_estimators': [10,12,14,16,18,19,20,22,24,26,28,30]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                        'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    model = xgb.XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(X_train, y_train)
    evalute_result = optimized_GBM.cv_results_
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    ## （2） 最大深度，最小分割权重
    attrs1 = optimized_GBM.best_params_['n_estimators']
    cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
    other_params = {'learning_rate': 0.1, 'n_estimators': int(attrs1), 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                        'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    model = xgb.XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(X_train, y_train)
    evalute_result = optimized_GBM.cv_results_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    ## （3）gamma调优
    attrs2 = optimized_GBM.best_params_['max_depth']
    attrs3 = optimized_GBM.best_params_['min_child_weight']
    cv_params = {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
    other_params = {'learning_rate': 0.1, 'n_estimators': int(attrs1), 'max_depth': int(attrs2), 'min_child_weight': int(attrs3), 'seed': 0,
                        'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    model = xgb.XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(X_train, y_train)
    evalute_result = optimized_GBM.cv_results_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    ## （4）'subsample'、'colsample_bytree'调优
    attrs4 = optimized_GBM.best_params_['gamma']
    cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
    other_params = {'learning_rate': 0.1, 'n_estimators': int(attrs1), 'max_depth': int(attrs2), 'min_child_weight': int(attrs3), 'seed': 0,
                        'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': np.float64(attrs4), 'reg_alpha': 0, 'reg_lambda': 1}
    model = xgb.XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(X_train, y_train)
    evalute_result = optimized_GBM.cv_results_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    ## (5) 'reg_alpha','reg_lambda'调优
    attrs5 = optimized_GBM.best_params_['subsample']
    attrs6 = optimized_GBM.best_params_['colsample_bytree']
    cv_params = {'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]}
    other_params = {'learning_rate': 0.1, 'n_estimators': int(attrs1), 'max_depth': int(attrs2), 'min_child_weight': int(attrs3), 'seed': 0,
                        'subsample': np.float64(attrs5), 'colsample_bytree': np.float64(attrs6), 'gamma': np.float64(attrs4), 'reg_alpha': 0, 'reg_lambda': 1}
    model = xgb.XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(X_train, y_train)
    evalute_result = optimized_GBM.cv_results_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    ## （6）'reg_alpha', 'reg_lambda'调优
    attrs7 = optimized_GBM.best_params_['reg_alpha']
    attrs8 = optimized_GBM.best_params_['reg_lambda']
    cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
    other_params = {'learning_rate': 0.1, 'n_estimators': int(attrs1), 'max_depth': int(attrs2), 'min_child_weight': int(attrs3), 'seed': 0,
                        'subsample': np.float64(attrs5), 'colsample_bytree': np.float64(attrs6), 'gamma': np.float64(attrs4), 'reg_alpha': np.float64(attrs7), 'reg_lambda': np.float64(attrs8)}
    model = xgb.XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(X_train, y_train)
    evalute_result = optimized_GBM.cv_results_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

    ## 最终的模型
    attrs9 = optimized_GBM.best_params_['learning_rate']
    other_params = {'learning_rate': np.float64(attrs9), 'n_estimators': int(attrs1), 'max_depth': int(attrs2), 'min_child_weight': int(attrs3), 'seed': 0,
                        'subsample': np.float64(attrs5), 'colsample_bytree': np.float64(attrs6), 'gamma': np.float64(attrs4), 'reg_alpha': np.float64(attrs7), 'reg_lambda': np.float64(attrs8)}
    model = xgb.XGBRegressor(**other_params)
    model.fit(X_train, y_train)
    return model
def cal_model_error(model,X_train, X_validation, y_train, y_validation):
    ## 计算误差
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_validation)
    plt.scatter(y_test_pred, y_validation)
    plt.title('Scatter Plot Example')
    plt.xlabel('predict')
    plt.ylabel('real')

    train_mse = mean_squared_error(y_train_pred.flatten(), y_train.flatten())
    test_mse = mean_squared_error(y_test_pred.flatten(), y_validation.flatten())
    train_r2 = cal_r2(y_train_pred.flatten(), y_train.flatten())
    test_r2 = cal_r2(y_test_pred.flatten(), y_validation.flatten())
    print("(before)Training MSE:", train_mse)
    print("(before)Testing MSE:", test_mse)  
    print("(before)Training R^2:", train_r2)
    print("(before)Testing R^2:", test_r2)  

def run_main_xgboost_train(filen, model_path, cols, ii, outpath_fig, fig_indx):
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    normal_indx = 0      ## 是否进行标准化处理
    split_scale = 0.2     ## 数据集划分比列，test_data = 0.2
    X_train, X_validation, y_train, y_validation = read_traindata(filen,normal_indx,split_scale)
    model = parameter_optimization(X_train, y_train)
    # cal_model_error(model,X_train, X_validation, y_train, y_validation)
    ## 保存模型

    #递归特征消除法,model是已经调整好的模型
    clf = model  ##
    ## 此处加入shap分析
    ## 1.数据转成df格式
    if ii==fig_indx:
        x_train_df = pd.DataFrame(X_train, columns=cols)
        explainer = shap.Explainer(model, feature_names=cols)
        shap_values = explainer(x_train_df)
        
        plt.figure(1)
        shap.plots.bar(shap_values,max_display=15)
        # plt.show()
        plt.savefig(outpath_fig+'figure1.png', dpi=300, bbox_inches='tight')  # 保存为PNG，分辨率300DPI
        plt.close('all')
        ##
        plt.figure(2)
        shap.summary_plot(shap_values, x_train_df, plot_type="bar", show=True)
        ax = plt.gca()
        ax.set_xlabel(ax.get_xlabel(), fontsize=20)
        ax.set_ylabel(ax.get_ylabel(), fontsize=20)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.savefig(outpath_fig + 'figure2.png', dpi=800, bbox_inches='tight')
        plt.close('all')

        # # shap.summary_plot(shap_values, x_train_df, plot_type="bar")
        # # plt.show()
        # plt.savefig(outpath_fig+'figure2.png', dpi=800, bbox_inches='tight')  # 保存为PNG，分辨率300DPI
        # plt.close('all')
        ## 
        plt.figure(3)
        shap.plots.heatmap(explainer(x_train_df))
        # plt.show()
        plt.savefig(outpath_fig+'figure3.png', dpi=300, bbox_inches='tight')  # 保存为PNG，分辨率300DPI
        plt.close('all')






    selector1 = RFE(clf, n_features_to_select=20, step=1).fit(X_train, y_train)
    # n_features_to_select表示筛选最终特征数量，step表示每次排除一个特征
    selector1.support_.sum()
    # 计算在 RFE 过程中被选中的特征数量，即布尔数组中值为 True 的个数，也就是最终选择的特征数量。
    print(selector1.ranking_)
    # 这个属性返回的是特征的排名，从1开始，表示每个特征在所有特征中的重要性排名，1为最重要的特征。
    print(selector1.n_features_)
    # 这是RFE在执行完所有递归步骤后最终选择的特征数量。
    y_train_pred = selector1.predict(X_train)
    y_test_pred = selector1.predict(X_validation)

    # 计算训练集和测试集的均方误差
    y_train_pred = y_train_pred.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    y_test_pred = y_test_pred.reshape(-1, 1)
    y_validation = y_validation.reshape(-1, 1)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_validation, y_test_pred)
    train_r2 = cal_r2(y_train_pred.flatten(), y_train.flatten())
    test_r2 = cal_r2(y_test_pred.flatten(), y_validation.flatten())

    print("(after)Training MSE:", train_mse)
    print("(after)Testing MSE:", test_mse)
    print("(before)Training R^2:", train_r2)
    print("(before)Testing R^2:", test_r2)  
    ## 打印出选择的特征要素
    cols = ['Albedo', 'TEMP_MIN', 'TEMP_MAX', 'TEMP', 'SSR', 'SRAD', 'slope', 'SI3', 'SI2', 'SI1',
            'SI_T', 'SI', 'SAVI', 'S6', 'S5', 'S3', 'S2', 'S1', 'RVI', 'Roughness', 'RND', 'PRCP',
            'POP', 'PM2_5', 'NDWI', 'NDVI', 'MAP', 'LU', 'GWP', 'FVC', 'EVI', 'ET', 'Elevation', 'DVI',
            'DMSP', 'CRSI', 'BI', 'ANNAP', 'AI']
    selected_features = [cols[i] for i in range(len(cols)) if selector1.support_[i]]
    print('特征递归消除后的参数：',selected_features)

    # ## 绘制散点图
    # from sklearn.linear_model import LinearRegression
    # from sklearn.linear_model import LinearRegression
    # import matplotlib.pyplot as plt
    #
    # # 定义颜色
    # predicted_color = '#A0A8B3'  # 灰蓝色
    # real_color = '#C9A3D1'  # 淡紫色
    # line_color = '#E74C3C'  # 回归线红色
    # # 创建线性回归模型
    # model_l = LinearRegression()
    # # reshape
    # y_test_pred = y_test_pred.reshape(-1, 1)
    # y_validation = y_validation.reshape(-1, 1)
    # # 拟合模型
    # model_l.fit(y_test_pred, y_validation)
    # # 绘制预测点 (灰蓝色)
    # plt.scatter(y_test_pred, y_validation, color=predicted_color, label="Predicted", s=30, alpha=0.7)
    # # 绘制回归线 (红色)
    # plt.plot(y_test_pred, model_l.predict(y_test_pred), color=line_color, label="Regression Line", linewidth=2)
    # # 添加标签和标题
    # plt.xlabel('Predicted', fontsize=12)
    # plt.ylabel('Real', fontsize=12)
    # strn = 'R^2=' + str(cal_r2(y_test_pred.flatten(), y_validation.flatten()))
    # plt.title(strn, fontsize=12, weight='bold')
    # # 添加图例
    # plt.legend()
    # # 调整坐标轴字体大小
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)
    # # 显示图形
    # plt.show()

    # # 将scaler保存到文件
    model_n = f'model_{str(ii + 1).zfill(2)}.pkl'

    # scaler_n = f'scaler_{str(ii + 1).zfill(2)}.pkl'
    # with open(model_path + '\\' + scaler_n, 'wb') as f:
    #     pickle.dump(scaler, f)

    # 将模型保存到文件
    with open(model_path + '\\' + model_n, 'wb') as f:
        pickle.dump(selector1, f)
    plt.close('all')

    print('DONE!')