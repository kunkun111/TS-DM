# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:11:52 2021

@author: Kun Wang
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
import arff
from skmultiflow.drift_detection import DDM
from scipy import stats
import time





# load .arff dataset
def load_arff(path, dataset_name):
    file_path = path + dataset_name + '/'+ dataset_name + '.arff'
    dataset = arff.load(open(file_path), encode_nominal=True)
    return pd.DataFrame(dataset["data"])
        
    

def drift_detection(y_pred, y_test):
    
    n = y_pred.shape[0]
    data_stream = np.zeros(n)
    
    # output results
    for i in range(n):
        if y_pred[i] == y_test[i]:
            data_stream[i] = 0
        else:
            data_stream[i] = 1
            
    idx_warning = 0
    idx_drift = 0
    
    # drift detection
    ddm = DDM(min_num_instances = 10)
    
    # detect warning zone
    for i in range(n):
        ddm.add_element(data_stream[i])
        
        if ddm.detected_warning_zone():
            idx_warning = i
            break
        
    # detect drift zone
    for i in range(n):
        ddm.add_element(data_stream[i])
        
        if ddm.detected_change():
            idx_drift = i
            break
        
            
    return idx_warning, idx_drift
    
        
    

def CES_method(data, ini_train_size, win_size, seeds, name):
    
    np.random.seed(seeds)
    
    data = data.values
    
    x1 = data[0:ini_train_size, :-1]
    y1 = data[0:ini_train_size, -1]
    
    x2 = np.zeros((x1.shape[0], x1.shape[1]))
    y2 = np.zeros(x1.shape[0])
    
    
    # build initial model for drift detection
    model_ini = GradientBoostingRegressor(subsample = 0.8)
    model_ini.fit(x1, y1)
    
    
    # build model 1 for normal data learning
    # model1 = GradientBoostingRegressor(n_estimators = 100, subsample = 0.8)
    model1 = GradientBoostingRegressor(subsample = 0.8)
    model1.fit(x1, y1)
    
    
    # build model 2 for drift data learning
    model2 = GradientBoostingRegressor(subsample = 0.8)
    model2.fit(x1, y1)
    
    
   
    
    # k-fold
    kf = KFold(int((data.shape[0] - ini_train_size) / win_size))
    stream = data[ini_train_size:, :]
    # pred = np.zeros(stream.shape[0])
    
    batch_acc = []
    batch_f1=[]
    acc2_pre = 0
    
    test_deviance1 = np.zeros((100,), dtype = np.float64)
    test_deviance2 = np.zeros((100,), dtype = np.float64)
    test_deviance2_former = np.zeros((100,), dtype = np.float64)
    test_deviance2_latter = np.zeros((100,), dtype = np.float64)
    
    pred = np.empty(0)
    
    # model learning
    for train_index, test_index in tqdm(kf.split(stream), total = kf.get_n_splits(), desc = "#batch"):
            
        x_test = stream[test_index, :-1]
        y_test = stream[test_index, -1]
        
        
        # test initial model for drift detection
        y_pred_ini = model_ini.predict(x_test)
        y_pred_ini = (y_pred_ini >= 0.5)

        
        # drift detection
        # idx = drift_detection(y_pred_ini, y_test)
        idx_warning, idx_drift = drift_detection(y_pred_ini, y_test)
        # print(idx_warning, idx_drift)
        
        model_ini = GradientBoostingRegressor(subsample = 0.8)
        model_ini.fit(x_test, y_test)
        
        
        # test model1
        y_pred_1 = model1.predict(x_test)
        y_pred_1 = (y_pred_1 >= 0.5)
        
        
        # warning zone selection
        high_warning_idx = []
        low_warning_idx = []
        high_warning = 0
        if idx_warning > 1 and idx_warning < idx_drift:
            normal_zone_acc = metrics.accuracy_score(y_test[0:idx_warning-1], y_pred_1.T[0:idx_warning-1])
            
            warning_zone_acc = []
            for i in range (idx_warning, idx_drift-1):
                warning_zone_acc = metrics.accuracy_score(y_test[0:i], y_pred_1[0:i])
                if warning_zone_acc > normal_zone_acc:
                    high_warning = i                 
                    
        
        idx_drift = high_warning + 1

        
        # evaluation
        acc_1 = metrics.accuracy_score(y_test, y_pred_1.T)
        f1_1 = metrics.f1_score(y_test, y_pred_1.T, average='macro')
        
        # for i, y_pred1 in enumerate(model1.staged_predict(x_test)):
        #     # clf.loss_ assumes that y_test[i] in {0, 1}
        #     test_deviance1[i] = model1.loss_(y_test, y_pred1)
            
        # test model2
        y_pred_2 = model2.predict(x_test)
        y_pred_2 = (y_pred_2 >= 0.5)

        
        # evaluation
        acc_2 = metrics.accuracy_score(y_test, y_pred_2.T)
        f1_2 = metrics.f1_score(y_test, y_pred_2.T, average='macro')
        
        
        
        # for i, y_pred2 in enumerate(model2.staged_predict(x_test)):
        #     # clf.loss_ assumes that y_test[i] in {0, 1}
            # test_deviance2[i] = model2.loss_(y_test, y_pred2)
        
        # print(test_deviance1)
        # print(test_deviance2)
        
        # zz = stats.kstest(test_deviance1, test_deviance2)
        # print(zz)
        
        # fig = plt.figure(figsize=(6, 6))
        # plt.plot(np.arange(100) + 1, test_deviance1, "r-", label="Model1 Test Set Deviance")
        # plt.plot(np.arange(100) + 1, test_deviance2, "b-", label="Model2 Test Set Deviance")
        # plt.legend(loc="upper right")
        # plt.xlabel("Boosting Iterations")
        # plt.ylabel("Deviance")
        # fig.tight_layout()
        # plt.show()
        
        if acc_1 > acc_2:
                
            x1 = np.vstack((x1, x_test))
            y1 = np.hstack((y1, y_test))
            
            if x1.shape[0] > 5000:
                x1 = x1[:-5000, :]
                y1 = y1[:-5000]
                
            
            # retrain the model 1
            model1 = GradientBoostingRegressor(subsample = 0.8)
            model1.fit(x1, y1)
            
            batch_acc.append(acc_1)
            batch_f1.append(f1_1)
            
            pred = np.hstack((pred, y_pred_1))
        
        else:
            
            if idx_drift == 0:
                
                # test model2 on former drift data
                y_pred_former = model2.predict(x2)
                y_pred_former = (y_pred_former >= 0.5)
                
                
                # test model2 on latter drift data
                y_pred_latter = model2.predict(x_test)
                y_pred_latter = (y_pred_latter >= 0.5)
                
        
                
                # output former loss 
                for i, y_pred2 in enumerate(model2.staged_predict(x2)):
                    test_deviance2_former[i] = model2.loss_(y2, y_pred_former)
                    
                
                # output latter loss 
                for i, y_pred2 in enumerate(model2.staged_predict(x_test)):
                    test_deviance2_latter[i] = model2.loss_(y_test, y_pred_latter)
                    
                    
                # K-S Test and calculate p-value
                t, p_value = stats.ks_2samp(test_deviance2_former, test_deviance2_latter, alternative='greater',mode='auto')
            
            
                # combine historical data samples
                if p_value < 0.01:
                    
                    x2 = x_test
                    y2 = y_test
                
                else:
                    
                    x2 = np.vstack((x2, x_test))
                    y2 = np.hstack((y2, y_test))
                
            else:
                
                # test model2 on former drift data
                y_pred_former = model2.predict(np.vstack((x2, x_test[idx_drift:, :])))
                y_pred_former = (y_pred_former >= 0.5)
                
                
                # test model2 on latter drift data
                y_pred_latter = model2.predict(x_test[idx_drift:, :])
                y_pred_latter = (y_pred_latter >= 0.5)
                
                acc_former = metrics.accuracy_score(np.hstack((y2, y_test[idx_drift:])), y_pred_former.T)
                acc_latter = metrics.accuracy_score(y_test[idx_drift:], y_pred_latter.T)
                # print('acc2:', acc_2)
                # print('former acc:', acc_former, 'latter acc:', acc_latter)
        
                
                # output former loss 
                for i, y_pred2 in enumerate(model2.staged_predict(x_test[idx_drift:, :])):
                    test_deviance2_former[i] = model2.loss_(np.hstack((y2, y_test[idx_drift:])), y_pred_former)
                    
                
                # output latter loss 
                for i, y_pred2 in enumerate(model2.staged_predict(np.vstack((x2, x_test[idx_drift:, :])))):
                    test_deviance2_latter[i] = model2.loss_(y_test[idx_drift:], y_pred_latter)
                    
                    
                # K-S Test and calculate p-value
                t, p_value = stats.ks_2samp(test_deviance2_former, test_deviance2_latter, alternative='greater',mode='auto')
                
                
                
                    
                # combine drift historical data samples
                if acc_2 < acc2_pre * 0.7 and p_value < 0.01:
                    
                    x2 = x_test[idx_drift:, :]
                    y2 = y_test[idx_drift:]
                
                else:
                    
                    x2 = np.vstack((x2, x_test[idx_drift:, :]))
                    y2 = np.hstack((y2, y_test[idx_drift:]))   
            
            
            if x2.shape[0] > 5000:
                x2 = x2[:-5000, :]
                y2 = y2[:-5000]
                
                
                
            # retrain the model 2
            model2 = GradientBoostingRegressor(subsample = 0.8)
            model2.fit(x2, y2)    
                
            batch_acc.append(acc_2)
            batch_f1.append(f1_2)
            
            pred = np.hstack((pred, y_pred_2))
            
            acc2_pre = acc_2
            
            
    
    
    Y = data[ini_train_size:,-1]
    acc = metrics.accuracy_score(Y, pred)
    f1 = metrics.f1_score(Y, pred, average = 'macro')

    print("acc:",acc)
    print("f1:",f1)
    
    # save results
    result = np.zeros([Y.shape[0], 2])
    result[:, 0] = pred
    result[:, 1] = Y
    np.savetxt(name + str(seeds) +'.out', result, delimiter=',') 


    return acc, f1
    


    
if __name__ == '__main__':
    
    
    path = 'realworld data/'

    datasets = ['elecNorm', 'spam_corpus_x2_feature_selected', 'airline', 'EEG_eye_state']
    # datasets = ['usenet1', 'usenet2']
    # datasets = ['weather']
    
    
    for i in range (len(datasets)):
        
        data = load_arff(path, datasets[i])
        
        acc_total = []
        f1_total = []
        time_total = []
    
        for j in range(5):
            
            print(datasets[i], j)
            time_start = time.time()
            ACC, F1 = CES_method(data, ini_train_size = 100, win_size = 100, seeds = j, name = datasets[i])
            time_end = time.time()
            Time = time_end - time_start
            print('time cost:', Time, 's')
            
            acc_total.append(ACC)
            f1_total.append(F1)
            time_total.append(Time)
        
        
        print('-----------------------------------------')
        print('AVE Accuracy:', np.mean(acc_total))
        print('STD Accuracy:', np.std(acc_total))
        print('-----------------------------------------')
        print('AVE F1:', np.mean(f1_total))
        print('STD F1:', np.std(f1_total))
        print('-----------------------------------------')
        print('AVE Time:', np.mean(time_total))
        print('STD Time:', np.std(time_total))
        print('-----------------------------------------') 
    
    





        
    








