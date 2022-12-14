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
    
    idx = 0
    
    # drift detection
    ddm = DDM(min_num_instances = 10)
    
    for i in range(n):
        ddm.add_element(data_stream[i])
        
        # if ddm.detected_warning_zone():
            # print('Warning zone has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
        
        if ddm.detected_change():
            # print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
            idx = i
            
            break
        
            
    return idx
        
    

def CES_method(data, ini_train_size, win_size, seeds, name):
    
    np.random.seed(seeds)
    
    data = data.values
    
    x1 = data[0:ini_train_size, :-1]
    y1 = data[0:ini_train_size, -1]
    
    x2 = np.zeros((x1.shape[0], x1.shape[1]))
    y2 = np.zeros(x1.shape[0])
    
    
    # build initial model for drift detection
    model_ini = GradientBoostingClassifier(subsample = 0.8)
    model_ini.fit(x1, y1)
    
    
    # build model 1 for normal data learning
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
    
    # model learning
    for train_index, test_index in tqdm(kf.split(stream), total = kf.get_n_splits(), desc = "#batch"):
            
        x_test = stream[test_index, :-1]
        y_test = stream[test_index, -1]
        
        
        # test initial model for drift detection
        y_pred_ini = model_ini.predict(x_test)
        # y_pred_ini = (y_pred_ini >= 0.5)

        
        # drift detection
        idx = drift_detection(y_pred_ini, y_test)
        
        model_ini = GradientBoostingClassifier(subsample = 0.8)
        model_ini.fit(x_test, y_test)
        
        
        # test model1
        y_pred_1 = model1.predict(x_test)
        # y_pred_1 = (y_pred_1 >= 0.5)

        
        # evaluation
        acc_1 = metrics.accuracy_score(y_test, y_pred_1.T)
        f1_1 = metrics.f1_score(y_test, y_pred_1.T, average='macro')
        
        # for i, y_pred1 in enumerate(model1.staged_predict(x_test)):
        #     # clf.loss_ assumes that y_test[i] in {0, 1}
        #     test_deviance1[i] = model1.loss_(y_test, y_pred1)
            
        # test model2 on the drift data
        y_pred_2 = model2.predict(x_test)
        # y_pred_2 = (y_pred_2 >= 0.5)

        
        # evaluation
        acc_2 = metrics.accuracy_score(y_test, y_pred_2.T)
        f1_2 = metrics.f1_score(y_test, y_pred_2.T, average='macro')
        
        # for i, y_pred2 in enumerate(model2.staged_predict(x_test)):
        #     # clf.loss_ assumes that y_test[i] in {0, 1}
        #     test_deviance2[i] = model2.loss_(y_test, y_pred2)
        
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
            
            # combine historical data samples
            x1 = np.vstack((x1, x_test))
            y1 = np.hstack((y1, y_test))
            
            
            label1 = np.argmax(y1)
            label2 = np.argmin(y1)
            
            
            label_x = np.vstack((x1[label1, :], x1[label2, :]))
            label_y = np.hstack((y1[label1], y1[label2]))
            
            
            if x1.shape[0] > 5000:
                # x1 = x1[:-5000, :]
                # y1 = y1[:-5000]
                
                x1 = np.vstack((label_x, x1[:-5000, :]))
                y1 = np.hstack((label_y, y1[:-5000]))
                
            
            # retrain the model 1
            model1 = GradientBoostingClassifier(subsample = 0.8)
            model1.fit(x1, y1)
            
            batch_acc.append(acc_1)
            batch_f1.append(f1_1)
            
            pred = np.hstack((pred, y_pred_1))
        
        else:
            
            if idx == 0:
                
                # combine historical data samples
                
                # x2 = x_test
                # y2 = y_test
                
                x2 = np.vstack((x2, x_test))
                y2 = np.hstack((y2, y_test))
                
            else:
                
                # combine normal historical data samples
                # x2 = x_test[idx:, :]
                # y2 = y_test[idx:]
                
                x2 = np.vstack((x2, x_test[idx:, :]))
                y2 = np.hstack((y2, y_test[idx:]))   
            
            
            label1 = np.argmax(y2)
            label2 = np.argmin(y2)
            
            
            label_x = np.vstack((x2[label1, :], x2[label2, :]))
            label_y = np.hstack((y2[label1], y2[label2]))
            
            if x2.shape[0] > 5000:
                # x2 = x2[:-5000, :]
                # y2 = y2[:-5000]
                
                x2 = np.vstack((label_x, x2[:-5000, :]))
                y2 = np.hstack((label_y, y2[:-5000]))
                
                
            # retrain the model 2
            model2 = GradientBoostingClassifier(subsample = 0.8)
            model2.fit(x2, y2)    
                
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
    # np.savetxt(name + str(seeds) +'.out', result, delimiter=',') 


    return acc, f1
    


    
if __name__ == '__main__':
    
    
    path = 'synthetic data/'
    
    datasets = ['LEDa', 'LEDg']
    

    
    
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
    
    
    







        
    








