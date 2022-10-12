# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 22:34:18 2021

@author: Administrator
"""


import numpy as np
import matplotlib.pyplot as plt

# data 1
mu_x11, delta_x11 = 10, 1.5
mu_x21, delta_x21 = 10, 2
x11 = mu_x11 + delta_x11 * np.random.randn(1000)
x21 = mu_x21 + delta_x21 * np.random.randn(1000)
y1 = np.zeros(1000)

y01_idx = np.where(x11 + x21 < 20)[0]
y11_idx = np.where(x11 + x21 >= 20)[0]
y1[y01_idx] = 0
y1[y11_idx] = 1 

data1 = np.vstack((x11, x21, y1))


# data 2
mu_x12, delta_x12 = 20, 1.5
mu_x22, delta_x22 = 10, 2
x12 = mu_x12 + delta_x12 * np.random.randn(1000)
x22 = mu_x22 + delta_x22 * np.random.randn(1000)
y2 = np.zeros(1000)

y02_idx = np.where(x12 + x22 < 30)[0]
y12_idx = np.where(x12 + x22 >= 30)[0]
y2[y02_idx] = 0
y2[y12_idx] = 1 

data2 = np.vstack((x12, x22, y2))


# data 3
mu_x13, delta_x13 = 30, 1.5
mu_x23, delta_x23 = 10, 2
x13 = mu_x13 + delta_x13 * np.random.randn(1000)
x23 = mu_x23 + delta_x23 * np.random.randn(1000)
y3 = np.zeros(1000)

y03_idx = np.where(x13 + x23 < 40)[0]
y13_idx = np.where(x13 + x23 >= 40)[0]
y3[y03_idx] = 0
y3[y13_idx] = 1 

data3 = np.vstack((x13, x23, y3))


# data 4
mu_x14, delta_x14 = 40, 1.5
mu_x24, delta_x24 = 10, 2
x14 = mu_x14 + delta_x14 * np.random.randn(1000)
x24 = mu_x24 + delta_x24 * np.random.randn(1000)
y4 = np.zeros(1000)

y04_idx = np.where(x14 + x24 < 50)[0]
y14_idx = np.where(x14 + x24 >= 50)[0]
y4[y04_idx] = 0
y4[y14_idx] = 1 

data4 = np.vstack((x14, x24, y4))


# data 5
mu_x15, delta_x15 = 50, 1.5
mu_x25, delta_x25 = 15, 2
x15 = mu_x15 + delta_x15 * np.random.randn(1000)
x25 = mu_x25 + delta_x25 * np.random.randn(1000)
y5 = np.zeros(1000)

y05_idx = np.where(x15  < x25+35)[0]
y15_idx = np.where(x15 >= x25+35)[0]
y5[y05_idx] = 0
y5[y15_idx] = 1 

data5 = np.vstack((x15, x25, y5))


# data 6
mu_x16, delta_x16 = 60, 1.5
mu_x26, delta_x26 = 15, 2
x16 = mu_x16 + delta_x16 * np.random.randn(1000)
x26 = mu_x26 + delta_x26 * np.random.randn(1000)
y6 = np.zeros(1000)

y06_idx = np.where(x16 < x26+45)[0]
y16_idx = np.where(x16 >= x26+45)[0]
y6[y06_idx] = 0
y6[y16_idx] = 1

data6 = np.vstack((x16, x26, y6))


# data 7
mu_x17, delta_x17 = 70, 1.5
mu_x27, delta_x27 = 25, 2
x17 = mu_x17 + delta_x17 * np.random.randn(1000)
x27 = mu_x27 + delta_x27 * np.random.randn(1000)
y7 = np.zeros(1000)

y07_idx = np.where(x17 >= x27 + 45 )[0]
y17_idx = np.where(x17 < x27 + 45)[0]
y7[y07_idx] = 0
y7[y17_idx] = 1

data7 = np.vstack((x17, x27, y7))


# data 8
mu_x18, delta_x18 = 80, 1.5
mu_x28, delta_x28 = 25, 2
x18 = mu_x18 + delta_x18 * np.random.randn(1000)
x28 = mu_x28 + delta_x28 * np.random.randn(1000)
y8 = np.zeros(1000)

y08_idx = np.where(x18 >= x28 + 55)[0]
y18_idx = np.where(x18 < x28 + 55 )[0]
y8[y08_idx] = 0
y8[y18_idx] = 1

data8 = np.vstack((x18, x28, y8))



# data 9
mu_x19, delta_x19 = 90, 1.5
mu_x29, delta_x29 = 30, 2
x19 = mu_x19 + delta_x19 * np.random.randn(1000)
x29 = mu_x29 + delta_x29 * np.random.randn(1000)
y9 = np.zeros(1000)

y09_idx = np.where(x19 + x29 >= 120)[0]
y19_idx = np.where(x19 + x29 < 120)[0]
y9[y09_idx] = 0
y9[y19_idx] = 1

data9 = np.vstack((x19, x29, y9))


# data 10
mu_x110, delta_x110 = 100, 1.5
mu_x210, delta_x210 = 30, 2
x110 = mu_x110 + delta_x110 * np.random.randn(1000)
x210 = mu_x210 + delta_x210 * np.random.randn(1000)
y10 = np.zeros(1000)

y010_idx = np.where(x110 + x210 >= 130)[0]
y110_idx = np.where(x110 + x210 < 130)[0]
y10[y010_idx] = 0
y10[y110_idx] = 1 

data10 = np.vstack((x110, x210, y10))


# data 11
mu_x111, delta_x111 = 110, 1.5
mu_x211, delta_x211 = 30, 2
x111 = mu_x111 + delta_x111 * np.random.randn(1000)
x211 = mu_x211 + delta_x211 * np.random.randn(1000)
y11 = np.zeros(1000)

y011_idx = np.where(x111 + x211 >= 140)[0]
y111_idx = np.where(x111 + x211 < 140)[0]
y11[y011_idx] = 0
y11[y111_idx] = 1

data11 = np.vstack((x111, x211, y11))


# data 12
mu_x112, delta_x112 = 120, 1.5
mu_x212, delta_x212 = 30, 2
x112 = mu_x112 + delta_x112 * np.random.randn(1000)
x212 = mu_x212 + delta_x212 * np.random.randn(1000)
y12 = np.zeros(1000)

y012_idx = np.where(x112 + x212 >= 150)[0]
y112_idx = np.where(x112 + x212 < 150)[0]
y12[y012_idx] = 0
y12[y112_idx] = 1 

data12 = np.vstack((x112, x212, y12))





data = np.hstack((data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12))
data = data.T

# np.savetxt('sud_data.txt', data, delimiter = ',')




plt.figure(figsize=(9, 3.5))


plt.plot(x11[y01_idx], x21[y01_idx], 'o', markersize = 1, color='tab:red', label = 'Label: y = 0')
plt.plot(x11[y11_idx], x21[y11_idx], 'o', markersize = 1, color='tab:blue', label = 'Label: y = 1')

plt.plot(x12[y02_idx], x22[y02_idx], 'o', markersize = 1, color='tab:red')
plt.plot(x12[y12_idx], x22[y12_idx], 'o', markersize = 1, color='tab:blue')

plt.plot(x13[y03_idx], x23[y03_idx], 'o', markersize = 1, color='tab:red')
plt.plot(x13[y13_idx], x23[y13_idx], 'o', markersize = 1, color='tab:blue')

plt.plot(x14[y04_idx], x24[y04_idx], 'o', markersize = 1, color='tab:red')
plt.plot(x14[y14_idx], x24[y14_idx], 'o', markersize = 1, color='tab:blue')

plt.plot(x15[y05_idx], x25[y05_idx], 'o', markersize = 1, color='tab:red')
plt.plot(x15[y15_idx], x25[y15_idx], 'o', markersize = 1, color='tab:blue')

plt.plot(x16[y06_idx], x26[y06_idx], 'o', markersize = 1, color='tab:red')
plt.plot(x16[y16_idx], x26[y16_idx], 'o', markersize = 1, color='tab:blue')

plt.plot(x17[y07_idx], x27[y07_idx], 'o', markersize = 1, color='tab:red')
plt.plot(x17[y17_idx], x27[y17_idx], 'o', markersize = 1, color='tab:blue')

plt.plot(x18[y08_idx], x28[y08_idx], 'o', markersize = 1, color='tab:red')
plt.plot(x18[y18_idx], x28[y18_idx], 'o', markersize = 1, color='tab:blue')

plt.plot(x19[y09_idx], x29[y09_idx], 'o', markersize = 1, color='tab:red')
plt.plot(x19[y19_idx], x29[y19_idx], 'o', markersize = 1, color='tab:blue')

plt.plot(x110[y010_idx], x210[y010_idx], 'o', markersize = 1, color='tab:red')
plt.plot(x110[y110_idx], x210[y110_idx], 'o', markersize = 1, color='tab:blue')

plt.plot(x111[y011_idx], x211[y011_idx], 'o', markersize = 1, color='tab:red')
plt.plot(x111[y111_idx], x211[y111_idx], 'o', markersize = 1, color='tab:blue')

plt.plot(x112[y012_idx], x212[y012_idx], 'o', markersize = 1, color='tab:red')
plt.plot(x112[y112_idx], x212[y112_idx], 'o', markersize = 1, color='tab:blue')


plt.axvline(x = 45, ls = "--", c = "black")
plt.axvline(x = 85, ls = "--", c = "black")

plt.xlabel('x1 (Timestamp)', fontsize = 15)
plt.ylabel('x2', fontsize = 15)

plt.ylim(0, 40)

plt.legend(loc = 'upper left', fontsize = 15)



