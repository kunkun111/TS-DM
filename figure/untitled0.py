# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 18:17:01 2022

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
import arff
from skmultiflow.drift_detection import DDM
from scipy import stats
import time



# sudden drift

plt.figure(figsize=(6, 5))

# batch acc 
plt.subplot(211)

result_dm = pd.read_csv('C:/Users/Administrator/Desktop/Work4/code - Copy/figure/CES-DM_method/sy_exp_b/SEAa0.out', header = None)
result_dm = result_dm.values

acc_dm = []
for i in range(0, result_dm.shape[0], 100):
    Y = result_dm[i:i+100, 1]
    pred = result_dm[i:i+100, 0]
    acc_dm.append(metrics.accuracy_score(Y, pred))
    

result_sw = pd.read_csv('C:/Users/Administrator/Desktop/Work4/code - Copy/figure/Slide/sy_exp_b/SEAa0.out', header = None)
result_sw = result_sw.values

acc_sw = []
for i in range(0, result_sw.shape[0], 100):
    Y = result_sw[i:i+100, 1]
    pred = result_sw[i:i+100, 0]
    acc_sw.append(metrics.accuracy_score(Y, pred))


plt.plot(acc_sw, label = 'TS-DM(sliding window)', color = 'tab:blue',  linewidth =2)
plt.plot(acc_dm, label = 'TS-DM', color = 'tab:red',  linewidth =2)

plt.axvline(x=25,c="tab:green", linestyle='--')
plt.axvline(x=50,c="tab:green", linestyle='--')
plt.axvline(x=75,c="tab:green", linestyle='--')

# plt.xlabel('Time stamps')
plt.ylabel('Accuracy(%)')

plt.ylim(0.6, 1.1)

plt.legend(loc = 'upper right', ncol = 2)


# batch segmentation 
plt.subplot(212)

# SEAa
dm = [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

# AGRa
# dm = [0, 0, 2, 2, 0, 2, 2, 0, 0, 0, 0, 2, 0, 2, 0, 2, 2, 0, 0, 2, 2, 0, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
a = []
dm_a = []
b = []
dm_b = []
c = []
dm_c = []

for i in range(len(dm)):
    if dm[i] == 0:
        a.append(i)
        dm_a.append(dm[i])
    elif dm[i] == 1:
        b.append(i)
        dm_b.append(dm[i])
    elif dm[i] == 2:
        c.append(i)
        dm_c.append(dm[i])
        
        
plt.scatter(a, dm_a, label = 'Normal data learning', marker = 'o', color = 'tab:blue')
plt.scatter(b, dm_b, label = 'Delete old drift data', marker = '*', color = 'tab:red')
plt.scatter(c, dm_c, label = 'Keep all drift data', marker = 'v', color = 'tab:green')


plt.xlabel('Time stamps')
plt.ylabel('Frequency')

plt.ylim(-0.5, 4)

plt.legend(loc = 'upper right', ncol = 2)

plt.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.1)
plt.savefig('C:/Users/Administrator/Desktop/Work4/work4 fig/' +'SEAa_sudden.pdf')
    
plt.show()




# bar chart

# # Time
DM = [0.856565657,0.706969697,0.901818182,0.834747475,0.812121212,0.820505051,0.684545455,0.67040404]
DMs = [0.836969697,0.700505051,0.901818182,0.835353535,0.758484848,0.813434343,0.638282828,0.643939394]

labels = ['SEAa','RTG','RBF','RBFr','AGRa','HYP','LEDa','LEDg']


# DM = [0.82279041,0.790344828,0.713103448,0.78638867,0.938963573,0.666329182,0.937903226,0.15581333,0.540526053]
# DMs = [0.816575246,0.787586207,0.713103448,0.783265962,0.921834345,0.64014256,0.953293011,0.165549148,0.463359336]

# labels = ['Elec','Usenet1','Usenet2','Weather','Spam','Airline','EEG','Power','Poker']

x = np.arange(len(labels))  # the label locations
width = 0.25


plt.figure(figsize=(7, 2.5))

index_DM = x
index_DMs = x+width

plt.bar(index_DM, DM, width = 0.25, label='TS-DM')
plt.bar(index_DMs, DMs, width = 0.25, label='TS-DM (sliding window)')


# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('Accuracy (%)')
plt.xlabel('Datasets')

plt.xticks(index_DM + width/2, labels)

plt.legend(loc = 'upper left', ncol = 2)

plt.ylim(0, 1.2)

plt.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.2)
plt.savefig('C:/Users/Administrator/Desktop/Work4/work4 fig/' +'fig3.pdf')
plt.show()



# sensitivity

para = [0.6,0.7,0.8,0.9,1]

SEAa = [0.856565657,	0.856565657,	0.856565657,	0.851515152,	0.84010101]
RTG = [0.706969697,	0.706969697,	0.706969697,	0.705959596,	0.709292929]
RBF = [0.901818182,	0.901818182,	0.901818182,	0.901818182,	0.901818182]
RBFr = [0.834747475,	0.834747475,	0.834747475,	0.834747475,	0.834747475]
AGRa = [0.812121212,	0.812121212,	0.812121212,	0.799090909,	0.770505051]
HYP = [0.820505051,	0.820505051,	0.820505051,	0.809090909,	0.819292929]
LEDa = [0.684747475,	0.684747475,	0.684747475,	0.684747475,	0.684747475]
LEDg = [0.66969697,	0.66969697,	0.66969697,	0.66969697,	0.665757576]


plt.figure(figsize=(6, 4.5))

plt.plot(para, SEAa, marker = 'o', label = 'SEAa')
plt.plot(para, RTG,marker = 'o', label = 'RTG')
plt.plot(para, RBF, marker = 'o', label = 'RBF')
plt.plot(para, RBFr, marker = 'o', label = 'RBFr')
plt.plot(para, AGRa, marker = 'o', label = 'AGRa')
plt.plot(para, HYP, marker = 'o', label = 'HYP')
plt.plot(para, LEDa, marker = 'o', label = 'LEDa')
plt.plot(para, LEDg, marker = 'o', label = 'LEDg')

plt.ylim(0.6, 1)

plt.ylabel('Accuracy (%)')
plt.xlabel('Parameter setting')

plt.legend(loc = 'upper left', ncol = 4)

plt.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.1)
plt.savefig('C:/Users/Administrator/Desktop/Work4/work4 fig/' +'fig5.pdf')
plt.show()



labels = ['Elec','Usenet1','Usenet2','Weather','Spam','Airline','EEG','Power','Poker']

Elec=[0.82279041, 0.819295762,	0.818388923,	0.812350703,	0.816730072]
Usenet1=[0.790344828,	0.790344828,	0.790344828,	0.8,	0.796551724]
Usenet2=[0.713103448,	0.713103448,	0.713103448,	0.713103448,	0.713103448]
Weather=[0.78638867,	0.78638867,	0.78638867,	0.78638867,	0.786332472]
Spam=[0.938963573,	0.938963573,	0.940264527,	0.913811795,	0.905789245]
Airline=[0.666329182,	0.666423752,	0.660753259,	0.656662643,	0.643561915]
EEG=[0.95141129,	0.937903226,	0.950604839,	0.950672043,	0.950134409]
Power=[0.15860936,	0.15860936,	0.15860936,	0.15860936,	0.15860936]
Poker=[0.541467147,	0.541467147,	0.542231223,	0.542231223,	0.542125213]

plt.figure(figsize=(6, 4.5))

plt.plot(para, Elec, marker = 'o', label = 'Elec')
plt.plot(para, Usenet1, marker = 'o', label = 'Usenet1')
plt.plot(para, Usenet2, marker = 'o', label = 'Usenet2')
plt.plot(para, Weather, marker = 'o', label = 'Weather')
plt.plot(para, Spam, marker = 'o', label = 'Spam')
plt.plot(para, Airline, marker = 'o', label = 'Airline')
plt.plot(para, EEG, marker = 'o', label = 'EEG')
plt.plot(para, Power, marker = 'o', label = 'Power')
plt.plot(para, Poker, marker = 'o', label = 'Poker')

plt.ylim(0.5, 1)

plt.ylabel('Accuracy (%)')
plt.xlabel('Parameter setting')

plt.legend(loc = 'lower left', ncol = 4)

plt.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.1)
plt.savefig('C:/Users/Administrator/Desktop/Work4/work4 fig/' +'fig6.pdf')
plt.show()