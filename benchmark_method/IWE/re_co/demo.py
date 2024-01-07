from river import evaluate, metrics, datasets, tree
from river.ensemble import BaggingClassifier
from IWE import IWE
from IWE_M import IWE_M
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
    
    np.random.seed(seeds)
    data = load_arff(path, name)
    print(name, win_size, seeds)
    
    # get data instances and labels
    y = data.pop(data.shape[1]-1)
    
    # load the learning model
    # model = IWE(model=tree.HoeffdingTreeClassifier(), n_models=5, window=100)
    model = IWE_M(model=tree.HoeffdingTreeClassifier(), n_models=5, window=100)
    
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
    
    np.random.seed(seeds)
    data = load_arff(path, name) 
    print(name, win_size, seeds)

    # get data instances and labels
    y = data.pop(data.shape[1]-1)
    
    # load the learning model
    # model = IWE(model=tree.HoeffdingTreeClassifier(), n_models=5, window=100)
    model = IWE_M(model=tree.HoeffdingTreeClassifier(), n_models=5, window=100)
     
    # f1 score   metric = metrics.F1() / metric = metrics.MacroF1()
    f1 = evaluate.progressive_val_score(
                                model = model,
                                delay = win_size,
                                dataset = stream.iter_pandas(data, y),
                                metric = metrics.MacroF1()
                            )
    print("f1:", f1)
    
    

def mcc(path, name, seeds, win_size):
    
    np.random.seed(seeds)
    data = load_arff(path, name)
    print(name, win_size, seeds)

    # get data instances and labels
    y = data.pop(data.shape[1]-1)
    
    # load the learning model
    # model = IWE(model=tree.HoeffdingTreeClassifier(), n_models=5, window=100)
    model = IWE_M(model=tree.HoeffdingTreeClassifier(), n_models=5, window=100)
    
    # mcc score
    mcc = evaluate.progressive_val_score(
                                model = model,
                                delay = win_size,
                                dataset = stream.iter_pandas(data, y),
                                metric = metrics.MCC()
                            )
    print("mcc:", mcc)
    
    
    
def rocauc(path, name, seeds, win_size):
    
    np.random.seed(seeds)
    data = load_arff(path, name)
    print(name, win_size, seeds)

    # get data instances and labels
    y = data.pop(data.shape[1]-1)
    
    # load the learning model
    model = IWE(model=tree.HoeffdingTreeClassifier(), n_models=5, window=100)
    # model = IWE_M(model=tree.HoeffdingTreeClassifier(), n_models=5, window=100)
    
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
    # model = IWE(model=tree.HoeffdingTreeClassifier(), n_models=5, window=100)
    model = IWE_M(model=tree.HoeffdingTreeClassifier(), n_models=5, window=100)
    
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
    # model = IWE(model=tree.HoeffdingTreeClassifier(), n_models=5, window=100)
    model = IWE_M(model=tree.HoeffdingTreeClassifier(), n_models=5, window=100)
    
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
    # name = ['Poker-Hand']
    # name = ['INSECTSi']
    # name = ['INSECTSa']
    # name = ['INSECTSg']
    name = ['covtype']


    for n in range (len(name)):
        
        acc_total = []
        f1_total = []
        time_total = []
        mcc_total = []
        
        for si in range(5):
            
            
            # time_start = time.time()
            # acc(path, name[n], seeds = si, win_size = 100)
            # time_end = time.time()
            # Time = time_end - time_start
            # print('time cost:', Time, 's')
            # time_total.append(Time)
            
            
            # f1(path, name[n], seeds = si, win_size = 100)

            # mcc(path, name[n], seeds = si, win_size = 100)
            
            # rocauc(path, name[n], seeds = si, win_size = 100)
            
            
            # mp(path, name[n], seeds = si, win_size = 100)
            
            mr(path, name[n], seeds = si, win_size = 100)
            
            
            
        # print('-----------------------------------------')
        # print('AVE Time:', np.mean(time_total))
        # print('STD Time:', np.std(time_total))
        # print('-----------------------------------------') 




'''
# data path
# name = ['elecNorm', 'spam_corpus_x2_feature_selected', 'airline', 'EEG_eye_state']
# name = ['usenet1', 'usenet2']
# name = ['weather']
# name = ['powersupply']
# name = ['Poker-Hand']
# name = ['INSECTSi']
# name = ['INSECTSa']
# name = ['INSECTSg']
name = ['covtype']


path = '/home/kunwang/Data/Work4/data/realworld data/'

for n in range (len(name)):
    
    acc_total = []
    f1_total = []
    time_total = []
    mcc_total = []
    
    for si in range(5):
        
        np.random.seed(si)
        
        data = load_arff(path, name[n])
        win_size = 100
    
        # get data instances and labels
        target = data.shape[1]-1
        y = data.pop(data.shape[1]-1)
        
        
        print(name[n], win_size, si)
        
        
        # time start
        time_start = time.time()
        
        
        # load the learning model
        # model = IWE(model=tree.HoeffdingTreeClassifier(), n_models=5, window=100)
        model = IWE_M(model=tree.HoeffdingTreeClassifier(), n_models=5, window=100)
        
        
        # accuracy
        with open(str(name[n]) + str(si) + '.out', 'w') as f:  
            acc = evaluate.progressive_val_score(
                                        model = model,
                                        delay = win_size,
                                        dataset = stream.iter_pandas(data, y),
                                        metric = metrics.Accuracy(),
                                        print_every = 1,
                                        file = f
                                    )
        print("acc:", acc)
        time_end = time.time()
        Time = time_end - time_start
        print('time cost:', Time, 's') 
        
        
        # f1 score
        f1 = evaluate.progressive_val_score(
                                    model = model,
                                    delay = win_size,
                                    dataset = stream.iter_pandas(data, y),
                                    metric = metrics.F1()
                                )
        print("f1:", f1)
        
        
        # mcc score
        mcc = evaluate.progressive_val_score(
                                    model = model,
                                    delay = win_size,
                                    dataset = stream.iter_pandas(data, y),
                                    metric = metrics.MCC()
                                )
        print("mcc:", mcc)
        
        
        
        # acc_total.append(acc)
        # f1_total.append(f1)
        # mcc_total.append(mcc)
        time_total.append(Time)
        
    
   
    # print('-----------------------------------------')
    # print('AVE Accuracy:', np.mean(acc_total))
    # print('STD Accuracy:', np.std(acc_total))
    # print('-----------------------------------------')
    # print('AVE F1:', np.mean(f1_total))
    # print('STD F1:', np.std(f1_total))
    # print('-----------------------------------------')
    # print('AVE MCC:', np.mean(mcc_total))
    # print('STD MCC:', np.std(mcc_total))
    print('-----------------------------------------')
    print('AVE Time:', np.mean(time_total))
    print('STD Time:', np.std(time_total))
    print('-----------------------------------------') 
    
'''   

