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
import time



# load .arff dataset
def load_arff(path, dataset_name):
    file_path = path + dataset_name + '/'+ dataset_name + '.arff'
    dataset = arff.load(open(file_path), encode_nominal=True)
    return pd.DataFrame(dataset["data"])



def CS_method(data, ini_train_size, win_size, seeds, name):
    
    np.random.seed(seeds)
    
    data = data.values
    
    x1 = data[0:ini_train_size, :-1]
    y1 = data[0:ini_train_size, -1]
    
    x2 = data[ini_train_size:ini_train_size + ini_train_size, :-1]
    y2 = data[ini_train_size:ini_train_size + ini_train_size, -1]
    
    
    # build model 1
    model1 = GradientBoostingRegressor(subsample = 0.8)
    model1.fit(x1, y1)
    
    y_pred_ini = model1.predict(x2)
    y_pred_ini = (y_pred_ini >= 0.5)
    
    # initial accuracy
    acc_ini = metrics.accuracy_score(y2, y_pred_ini.T)
    f1_ini = metrics.f1_score(y2, y_pred_ini.T, average='macro')
    
    
    # build model 2
    model2 = GradientBoostingRegressor(subsample = 0.8)
    model2.fit(x2, y2)
    
    
    # k-fold
    kf = KFold(int((data.shape[0] - (ini_train_size + ini_train_size)) / win_size))
    stream = data[ini_train_size + ini_train_size:, :]
    pred = np.zeros(stream.shape[0])
    
    batch_acc = [acc_ini]
    batch_f1 = [f1_ini]
    
    cum_acc = [acc_ini]
    cum_f1 = [f1_ini]
    
    y_pred_cum = y_pred_ini
    y_test_cum = y2
    
    
    # model learning
    for train_index, test_index in tqdm(kf.split(stream), total = kf.get_n_splits(), desc = "#batch"):
            
        x_test = stream[test_index, :-1]
        y_test = stream[test_index, -1]
        
        
        # test model 1
        y_pred_1 = model1.predict(x_test)
        y_pred_1 = (y_pred_1 >= 0.5)
        
        acc_1 = metrics.accuracy_score(y_test, y_pred_1.T)
        f1_1 = metrics.f1_score(y_test, y_pred_1.T, average='macro')
        
        
        # test model 2
        y_pred_2 = model2.predict(x_test)
        y_pred_2 = (y_pred_2 >= 0.5)
        
        acc_2 = metrics.accuracy_score(y_test, y_pred_2.T)
        f1_2 = metrics.f1_score(y_test, y_pred_2.T, average='macro')
        
        
        # compare the result
        if acc_1 > acc_2:
            
            batch_acc.append(acc_1)
            batch_f1.append(f1_1)
            
            y_pred_cum = np.hstack((y_pred_cum, y_pred_1))
            # y_test_cum = np.hstack((y_test_cum, y_test))
            
            # cum_acc.append(metrics.accuracy_score(y_test_cum, y_pred_cum.T))
            # cum_f1.append(metrics.f1_score(y_test_cum, y_pred_cum.T, average='macro'))
            
            # combine historical chunk
            x1 = np.vstack((x1, x_test))
            y1 = np.hstack((y1, y_test))

            
            if x1.shape[0] > 5000:
                x1 = x1[:-5000, :]
                y1 = y1[:-5000]
            
            
            # retrain the model 1
            model1 = GradientBoostingRegressor(subsample = 0.8)
            model1.fit(x1, y1)


        
        else:
            batch_acc.append(acc_2)
            batch_f1.append(f1_2)
            
            y_pred_cum = np.hstack((y_pred_cum, y_pred_2))
            # y_test_cum = np.hstack((y_test_cum, y_test))
            
            # cum_acc.append(metrics.accuracy_score(y_test_cum, y_pred_cum.T))
            # cum_f1.append(metrics.f1_score(y_test_cum, y_pred_cum.T, average='macro'))
            
            # combine historical chunk
            x2 = np.vstack((x2, x_test))
            y2 = np.hstack((y2, y_test))
            
            if x2.shape[0] > 5000:
                x2 = x2[:-5000, :]
                y2 = y2[:-5000]
            
            
            # retrain the model 2
            model2 = GradientBoostingRegressor(subsample = 0.8)
            model2.fit(x2, y2)
            
    Y = data[ini_train_size:,-1]
    acc = metrics.accuracy_score(Y, y_pred_cum)
    f1 = metrics.f1_score(Y, y_pred_cum, average = 'macro')

    print("acc:",acc)
    print("f1:",f1)
    
    # save results
    result = np.zeros([Y.shape[0], 2])
    result[:, 0] = y_pred_cum
    result[:, 1] = Y
    np.savetxt(name + str(seeds) +'.out', result, delimiter=',')          
    
    # print('acc', np.mean(batch_acc))

    return acc, f1
    



    
if __name__ == '__main__':
    
    
    path = 'realworld data/'
    
    # datasets = ['elecNorm','spam_corpus_x2_feature_selected','EEG_eye_state','airline']
    datasets = ['usenet1','usenet2']
    # datasets = ['weather']

    
    for i in range (len(datasets)):
        
        data = load_arff(path, datasets[i])
        
        acc_total = []
        f1_total = []
        time_total = []
    
        for j in range(5):
            
            print(datasets[i], j)
            time_start = time.time()
            ACC, F1 = CS_method(data, ini_train_size = 50, win_size = 50, seeds = j, name = datasets[i])
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
        
    
    
    
    
    
    











        
    








