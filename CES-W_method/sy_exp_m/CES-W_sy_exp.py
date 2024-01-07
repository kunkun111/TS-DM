# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:11:52 2021

@author: Kun Wang
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
import arff
from skmultiflow.drift_detection import DDM
from scipy import stats
import time



# load .arff dataset
def load_arff(path, dataset_name, seeds):
    file_path = path + dataset_name + '/'+ dataset_name + str(seeds) + '.arff'
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
    # data = data[:500, :]
    
    x1 = data[0:ini_train_size, :-1]
    y1 = data[0:ini_train_size, -1]
    
    x2 = np.zeros((x1.shape[0], x1.shape[1]))
    y2 = np.zeros(x1.shape[0])
    
    
    # build initial model for drift detection
    model_ini = GradientBoostingClassifier(subsample = 0.8)
    model_ini.fit(x1, y1)
    
    
    # build model 1 for normal data learning
    # model1 = GradientBoostingRegressor(n_estimators = 100, subsample = 0.8)
    model1 = GradientBoostingClassifier(subsample = 0.8)
    model1.fit(x1, y1)
    
    
    # build model 2 for drift data learning
    model2 = GradientBoostingClassifier(subsample = 0.8)
    model2.fit(x1, y1)
    
    
    
    # k-fold
    kf = KFold(int((data.shape[0] - ini_train_size) / win_size))
    stream = data[ini_train_size:, :]
    # pred = np.zeros(stream.shape[0])
    
    batch_acc = []
    batch_f1=[]
    
    test_deviance1 = np.zeros((100,), dtype = np.float64)
    test_deviance2 = np.zeros((100,), dtype = np.float64)
    
    pred = np.empty(0)
    
    acc1_pre = 0
    acc2_pre = 0
    acc_df = []
    term = 0
    
    
    # model learning
    for train_index, test_index in tqdm(kf.split(stream), total = kf.get_n_splits(), desc = "#batch"):
            
        x_test = stream[test_index, :-1]
        y_test = stream[test_index, -1]
        
        
        # test initial model for drift detection
        y_pred_ini = model_ini.predict(x_test)
        # y_pred_ini = (y_pred_ini >= 0.5)

        
        # drift detection
        # idx = drift_detection(y_pred_ini, y_test)
        idx_warning, idx_drift = drift_detection(y_pred_ini, y_test)
        # print(idx_warning, idx_drift)
        
        model_ini = GradientBoostingClassifier(subsample = 0.8)
        model_ini.fit(x_test, y_test)
        
        
        # test model1
        y_pred_1 = model1.predict(x_test)
        # y_pred_1 = (y_pred_1 >= 0.5)
        
        
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

            
        # test model2 on the drift data
        y_pred_2 = model2.predict(x_test)
        # y_pred_2 = (y_pred_2 >= 0.5)

        
        # evaluation
        acc_2 = metrics.accuracy_score(y_test, y_pred_2.T)
        f1_2 = metrics.f1_score(y_test, y_pred_2.T, average='macro')
 
        
        if acc_1 > acc_2:
                
            x1 = np.vstack((x1, x_test))
            y1 = np.hstack((y1, y_test))
            
            
            if x1.shape[0] > 1000:
                
                if acc_1 >= acc1_pre:
                    
                    x1 = x1[-1000:, :]
                    y1 = y1[-1000:]
                    
                else:
                
                    x1 = x_test
                    y1 = y_test
            
            # retrain the model 1
            model1 = GradientBoostingClassifier(subsample = 0.8)
            model1.fit(x1, y1)
            
            acc1_pre = acc_1
            
            batch_acc.append(acc_1)
            batch_f1.append(f1_1)
            
            pred = np.hstack((pred, y_pred_1))
        
        else:
            
            if idx_drift == 0:
                
                # combine historical data samples
                x2 = np.vstack((x2, x_test))
                y2 = np.hstack((y2, y_test))
                
            else:
                
                # combine drift historical data samples
                x2 = np.vstack((x2, x_test[idx_drift:, :]))
                y2 = np.hstack((y2, y_test[idx_drift:]))   
            
            
            if x2.shape[0] > 1000:
                
                if acc_2 >= acc2_pre:
                    
                    x2 = x2[-1000:, :]
                    y2 = y2[-1000:]
                    
                else:
                
                    x2 = x_test
                    y2 = y_test
                
            
            # retrain the model 2
            model2 = GradientBoostingClassifier(subsample = 0.8)
            model2.fit(x2, y2)
            
            acc2_pre = acc_2    
                
            batch_acc.append(acc_2)
            batch_f1.append(f1_2)
            
            pred = np.hstack((pred, y_pred_2))
            
    
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
    
    
    # path = 'C:/Users/Administrator/Desktop/Work4/data/synthetic data/'
    path = '/home/kunwang/Data/Work4/data/synthetic data/'

    datasets = ['LEDa', 'LEDg', 'Waveform']

    
    
    for i in range (len(datasets)):
        
        acc_total = []
        f1_total = []
        time_total = []
    
        for j in range(15):
            
            data = load_arff(path, datasets[i], j)
            
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
    
    
    






        
    








