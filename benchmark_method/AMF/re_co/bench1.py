#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 09:12:41 2023

@author: kunwang
"""


from river import preprocessing
# from river import datasets
from river import evaluate
from river import metrics
from river import forest
import pandas as pd
import arff
from sklearn.model_selection import KFold
from tqdm import tqdm
from river import stream
from river import preprocessing
# from sklearn import metrics
import numpy as np
import time




# load data
def load_arff(path, dataset_name):
    file_path = path + dataset_name + '/'+ dataset_name  + '.arff'
    dataset = arff.load(open(file_path), encode_nominal=True)
    return pd.DataFrame(dataset["data"])



def acc(path, name, seeds, win_size):
    
    data = load_arff(path, name)
    print(name, win_size, seeds)
    
    # get data instances and labels
    y = data.pop(data.shape[1]-1)
    
    # load the learning model
    model = forest.AMFClassifier(seed = seeds)
    
    # accuracy
    with open(str(name) + str(seeds) + '.out', 'w') as f:  
        acc = evaluate.progressive_val_score(
                                    model = model,
                                    delay = win_size,
                                    dataset = stream.iter_pandas(data, y),
                                    metric = metrics.Accuracy(),
                                    print_every = 1,
                                    file = f
                                )
    print("acc:", acc)
    
    
       
def f1(path, name, seeds, win_size):
    
    data = load_arff(path, name) 
    print(name, win_size, seeds)

    # get data instances and labels
    y = data.pop(data.shape[1]-1)
    
    # load the learning model
    model = forest.AMFClassifier(seed = seeds)
     
    # f1 score   metric = metrics.F1() / metric = metrics.MacroF1()
    f1 = evaluate.progressive_val_score(
                                model = model,
                                delay = win_size,
                                dataset = stream.iter_pandas(data, y),
                                metric = metrics.MacroF1()
                            )
    print("f1:", f1)
    
    

def mcc(path, name, seeds, win_size):
    
    data = load_arff(path, name)
    print(name, win_size, seeds)

    # get data instances and labels
    y = data.pop(data.shape[1]-1)
    
    # load the learning model
    model = forest.AMFClassifier(seed = seeds)
    
    # mcc score
    mcc = evaluate.progressive_val_score(
                                model = model,
                                delay = win_size,
                                dataset = stream.iter_pandas(data, y),
                                metric = metrics.MCC()
                            )
    print("mcc:", mcc)
    
    
    
def rocauc(path, name, seeds, win_size):
    
    data = load_arff(path, name)
    print(name, win_size, seeds)

    # get data instances and labels
    y = data.pop(data.shape[1]-1)
    
    # load the learning model
    model = forest.AMFClassifier(seed = seeds)
    
    # mcc score
    rocauc = evaluate.progressive_val_score(
                                model = model,
                                delay = win_size,
                                dataset = stream.iter_pandas(data, y),
                                metric = metrics.ROCAUC()
                            )
    print("rocauc:", rocauc)
    
    

def mp(path, name, seeds, win_size):
    
    np.random.seed(seeds)
    data = load_arff(path, name)
    print(name, win_size, seeds)

    # get data instances and labels
    y = data.pop(data.shape[1]-1)
    
    # load the learning model
    model = forest.AMFClassifier(seed = seeds)
    
    # mcc score
    mp = evaluate.progressive_val_score(
                                model = model,
                                delay = win_size,
                                dataset = stream.iter_pandas(data, y),
                                metric = metrics.MacroPrecision()
                            )
    print("mp:", mp)
    
    
    
def mr(path, name, seeds, win_size):
    
    np.random.seed(seeds)
    data = load_arff(path, name)
    print(name, win_size, seeds)

    # get data instances and labels
    y = data.pop(data.shape[1]-1)
    
    # load the learning model
    model = forest.AMFClassifier(seed = seeds)
    
    # mcc score
    mr = evaluate.progressive_val_score(
                                model = model,
                                delay = win_size,
                                dataset = stream.iter_pandas(data, y),
                                metric = metrics.MacroRecall()
                            )
    print("mr:", mr)
    
    

if __name__ == '__main__':
    
    path = '/home/kunwang/Data/Work4/data/realworld data/'

    # name = ['elecNorm', 'spam_corpus_x2_feature_selected', 'airline', 'EEG_eye_state']
    # name = ['usenet1', 'usenet2']
    # name = ['weather']
    # name = ['powersupply']
    name = ['Poker-Hand']
    # name = ['INSECTSi']
    # name = ['INSECTSa']
    # name = ['INSECTSg']
    # name = ['covtype']


    for n in range (len(name)):
        
        acc_total = []
        f1_total = []
        time_total = []
        mcc_total = []
        
        # for si in range(5):
        # for si in range(0,1):
        # for si in range(1,2):        
        # for si in range(2,3):
        for si in range(3,4):
        # for si in range(4,5):
            
            # time_start = time.time()
            # acc(path, name[n], seeds = si, win_size = 100)
            # time_end = time.time()
            # Time = time_end - time_start
            # print('time cost:', Time, 's')
            # time_total.append(Time)
            
            
            # f1(path, name[n], seeds = si, win_size = 100)

            # mcc(path, name[n], seeds = si, win_size = 100)
            
            # rocauc(path, name[n], seeds = si, win_size = 100)
            
            mp(path, name[n], seeds = si, win_size = 100)
            
            # mr(path, name[n], seeds = si, win_size = 100)
            
            
            
        # print('-----------------------------------------')
        # print('AVE Time:', np.mean(time_total))
        # print('STD Time:', np.std(time_total))
        # print('-----------------------------------------') 


    



    
    