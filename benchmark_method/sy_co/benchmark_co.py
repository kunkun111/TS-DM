#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 12:45:25 2020

@author: kunwang
"""

# Imports

from skmultiflow.data import SEAGenerator
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.meta import LearnPPNSEClassifier
from skmultiflow.meta import OnlineCSB2Classifier
from skmultiflow.meta import LeveragingBaggingClassifier
from skmultiflow.meta import OnlineBoostingClassifier
from skmultiflow.meta import OnlineRUSBoostClassifier
from skmultiflow.meta import OzaBaggingClassifier
from skmultiflow.meta import AdditiveExpertEnsembleClassifier
import numpy as np
import arff
import pandas as pd
from skmultiflow.data.data_stream import DataStream
from sklearn.metrics import confusion_matrix, precision_score, matthews_corrcoef, accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve
import time


# load .arff dataset
def load_arff(path, dataset_name, seeds):
    file_path = path + dataset_name + '/'+ dataset_name + str(seeds) + '.arff'
    dataset = arff.load(open(file_path), encode_nominal=True)
    return pd.DataFrame(dataset["data"])



def ARF_run (dataset_name, batch, seeds):
    np.random.seed(seeds)
    data = load_arff(path, dataset_name, seeds)

    # data transform
    stream = DataStream(data)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    
    model = AdaptiveRandomForestClassifier()
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample(batch)
        y_pred = model.predict(X)
        pred = np.hstack((pred,y_pred))
        model.partial_fit(X, y,stream.target_values)
        n_samples += batch

    # evaluate
    data = data.values
    Y = data[:,-1]
    acc = accuracy_score(Y[batch:], pred[batch:])
    f1 = f1_score(Y[batch:], pred[batch:], average = 'macro')

    print("acc:",acc)
    print("f1:",f1)
    
    # save results
    result = np.zeros([pred[batch:].shape[0], 2])
    result[:, 0] = pred[batch:]
    result[:, 1] = Y[batch:]
    np.savetxt(dataset_name +str(seeds) + '_'+'ARF.out', result, delimiter=',')
    
    return acc, f1


    
def NSE_run (dataset_name, batch, seeds):
    np.random.seed(seeds)
    data = load_arff(path, dataset_name, seeds)

    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    
    model = LearnPPNSEClassifier()
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample(batch)
        y_pred = model.predict(X)
        pred = np.hstack((pred,y_pred))
        model.partial_fit(X, y,stream.target_values)
        n_samples += batch

    # evaluate
    data = data.values
    Y = data[:,-1]
    acc = accuracy_score(Y[batch:], pred[batch:])
    f1 = f1_score(Y[batch:], pred[batch:], average = 'macro')

    print("acc:",acc)
    print("f1:",f1)
    
    # save results
    result = np.zeros([pred[batch:].shape[0], 2])
    result[:, 0] = pred[batch:]
    result[:, 1] = Y[batch:]
    np.savetxt(dataset_name +str(seeds) + '_'+'NSE.out', result, delimiter=',')
    
    return acc, f1
    
    
def LEV_run (dataset_name, batch, seeds):
    np.random.seed(seeds)
    data = load_arff(path, dataset_name, seeds)

    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    
    model = LeveragingBaggingClassifier()
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample(batch)
        y_pred = model.predict(X)
        pred = np.hstack((pred,y_pred))
        model.partial_fit(X, y,stream.target_values)
        n_samples += batch

    # evaluate
    data = data.values
    Y = data[:,-1]
    acc = accuracy_score(Y[batch:], pred[batch:])
    f1 = f1_score(Y[batch:], pred[batch:], average = 'macro')

    print("acc:",acc)
    print("f1:",f1)
    
    # save results
    result = np.zeros([pred[batch:].shape[0], 2])
    result[:, 0] = pred[batch:]
    result[:, 1] = Y[batch:]
    np.savetxt(dataset_name +str(seeds) + '_'+'LEV.out', result, delimiter=',')
    
    return acc, f1

    
def OBC_run (dataset_name, batch, seeds):
    np.random.seed(seeds)
    data = load_arff(path, dataset_name, seeds)

    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    
    model1 = OnlineBoostingClassifier()
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample(batch)
        y_pred = model1.predict(X)
        pred = np.hstack((pred,y_pred))
        model1.partial_fit(X, y,stream.target_values)
        n_samples += batch

    # evaluate
    data = data.values
    Y = data[:,-1]
    acc = accuracy_score(Y[batch:], pred[batch:])
    f1 = f1_score(Y[batch:], pred[batch:], average = 'macro')

    print("acc:",acc)
    print("f1:",f1)

    
    # save results
    result = np.zeros([pred[batch:].shape[0], 2])
    result[:, 0] = pred[batch:]
    result[:, 1] = Y[batch:]
    np.savetxt(dataset_name +str(seeds) + '_'+'OBC.out', result, delimiter=',')
   
    return acc, f1


def RUS_run (dataset_name, batch, seeds):
    np.random.seed(seeds)
    data = load_arff(path, dataset_name, seeds)

    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    
    model = OnlineRUSBoostClassifier()
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample(batch)
        y_pred = model.predict(X)
        pred = np.hstack((pred,y_pred))
        model.partial_fit(X, y,stream.target_values)
        n_samples += batch

    # evaluate
    data = data.values
    Y = data[:,-1]
    acc = accuracy_score(Y[batch:], pred[batch:])
    f1 = f1_score(Y[batch:], pred[batch:], average = 'macro')

    print("acc:",acc)
    print("f1:",f1)

    
    # save results
    result = np.zeros([pred[batch:].shape[0], 2])
    result[:, 0] = pred[batch:]
    result[:, 1] = Y[batch:]
    np.savetxt(dataset_name +str(seeds) + '_'+'RUS.out', result, delimiter=',')
    
    return acc, f1
    
    
def OZA_run (dataset_name, batch, seeds):
    np.random.seed(seeds)
    data = load_arff(path, dataset_name, seeds)

    # data transform
    stream = DataStream(data)
    #print(stream)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    
    model = OzaBaggingClassifier()
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample(batch)
        y_pred = model.predict(X)
        pred = np.hstack((pred,y_pred))
        model.partial_fit(X, y,stream.target_values)
        n_samples += batch

    # evaluate
    data = data.values
    Y = data[:,-1]
    acc = accuracy_score(Y[batch:], pred[batch:])
    f1 = f1_score(Y[batch:], pred[batch:], average = 'macro')

    print("acc:",acc)
    print("f1:",f1)
    
    # save results
    result = np.zeros([pred[batch:].shape[0], 2])
    result[:, 0] = pred[batch:]
    result[:, 1] = Y[batch:]
    np.savetxt(dataset_name +str(seeds) + '_'+'OZA.out', result, delimiter=',')
    
    return acc, f1
    

def AEC_run (dataset_name, batch, seeds):
    np.random.seed(seeds)
    data = load_arff(path, dataset_name, seeds)

    # data transform
    stream = DataStream(data)

    # Setup variables to control loop and track performance
    n_samples = 0
    max_samples = data.shape[0]

    # Train the classifier with the samples provided by the data stream
    pred = np.empty(0)
    
    model = AdditiveExpertEnsembleClassifier()
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample(batch)
        y_pred = model.predict(X)
        pred = np.hstack((pred,y_pred))
        model.partial_fit(X, y,stream.target_values)
        n_samples += batch

    # evaluate
    data = data.values
    Y = data[:,-1]
    acc = accuracy_score(Y[batch:], pred[batch:])
    f1 = f1_score(Y[batch:], pred[batch:], average = 'macro')

    print("acc:",acc)
    print("f1:",f1)
    
    # save results
    result = np.zeros([pred[batch:].shape[0], 2])
    result[:, 0] = pred[batch:]
    result[:, 1] = Y[batch:]
    np.savetxt(dataset_name +str(seeds) + '_'+'AEC.out', result, delimiter=',')
    
    return acc, f1


    
    
    
if __name__ == '__main__':
    

    path = 'synthetic data/'
    datasets = ['SEAa'] # datasets = ['SEAa', 'RTG', 'RBF', 'RBFr', 'AGRa', 'HYP', 'LEDa', 'LEDg']
    batch = [100]
    
    acc_total = []
    f1_total = []
    time_total = []
    
    for i in range (0,15):
        dataset_name = datasets[0]
        batch_size = batch[0]
        
        
        # print (dataset_name, batch_size, i , 'ARF')
        # time_start = time.time()
        # ACC, F1 = ARF_run(dataset_name, batch_size, i)
        # time_end = time.time()
        # Time = time_end - time_start  
        # print('time cost:', Time, 's')
        

        # print (dataset_name, batch_size, i, 'NSE')
        # time_start = time.time()
        # ACC, F1 = NSE_run(dataset_name, batch_size, i)
        # time_end = time.time()
        # Time = time_end - time_start
        # print('time cost:', Time, 's')
        
        
        # print (dataset_name, batch_size, i, 'LEV')
        # time_start = time.time()
        # ACC, F1 = LEV_run(dataset_name, batch_size, i)
        # time_end = time.time()
        # Time = time_end - time_start
        # print('time cost:', Time, 's')
        
        
        # print (dataset_name, batch_size, i, 'OBC')
        # time_start = time.time()
        # ACC, F1 = OBC_run(dataset_name, batch_size, i)
        # time_end = time.time()
        # Time = time_end - time_start
        # print('time cost:', Time, 's')
        
        
        # print (dataset_name, batch_size, i, 'RUS')
        # time_start = time.time() 
        # ACC, F1 = RUS_run(dataset_name, batch_size, i) 
        # time_end = time.time()
        # Time = time_end - time_start
        # print('time cost:', Time, 's')
        
        
        # print (dataset_name, batch_size, i, 'OZA')
        # time_start = time.time()
        # ACC, F1 = OZA_run(dataset_name, batch_size, i) 
        # time_end = time.time()
        # Time = time_end - time_start
        # print('time cost:', Time, 's')
        
        
        print (dataset_name, batch_size, i, 'AEC')
        time_start = time.time()
        ACC, F1 = AEC_run(dataset_name, batch_size, i) 
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
 
