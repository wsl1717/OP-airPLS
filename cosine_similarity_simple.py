#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的余弦相似度计算器
快速计算CSV文件中两列数值的余弦相似度
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from natsort import natsorted
from scipy.interpolate import interp1d

def read(filename,wave_start,wave_end): 
    file = open(filename) #打开文件
    data_lines = file.readlines() #读取文件
    file.close() #关闭文件
    orign_keys = []  #储存波数
    orign_values = []  #储存峰强
    for data_line in data_lines[wave_start:wave_end]:#遍历所有行
        pair = data_line.split(' ') #分开波数和峰强,分隔符为空格
        #pair = data_line.split(',') #分开波数和峰强,分隔符为逗号
        #pair = data_line.split('\t') #分开波数和峰强,分隔符为逗号
        key = float(pair[0]) #波数
        value = float(pair[1]) #峰强
        orign_keys.append(key) #存储波数
        orign_values.append(value) #存储峰强    
    return orign_keys, orign_values  #返回所有行的波数和峰强

#按照文件名称进行读取文件
def read_simple_sorted(file_path,wave_start = 0,wave_end = 5000):#file_name是文件路径,wave_start是起始波段,wave_end是结束波段
    file_name=os.listdir(file_path)     #获取file_name下的所有文件名
    file_name=natsorted(file_name)      #按照文件名的ASCLL码排序，和文件夹中顺序保持一致
    num=0 
    n1=[] 
    m1=[] 
    name=[] 
    for i in file_name: 
        file1=os.path.join(file_path,i)     #获取单个文件的路径
        # print(file1,num) #打印文件名
        name.append(file1) 
        m,n=read(file1,wave_start,wave_end) 
        n1.append(n) 
        m1.append(m) 
        num=num+1
    return np.array(m1),np.array(n1),name #返回波数、峰强和文件名


file_path1 = r'e:\Desktop\扣背景finetune\data\2\bc'
file_path2 = r'e:\Desktop\OP-airPLS\3. data\模拟数据.csv'

wavenumber, intensity, name = read_simple_sorted(file_path1)
x = intensity[0]
f = interp1d(wavenumber[0], intensity[0], kind='linear')
keys_new=np.linspace(min(wavenumber[0]),max(wavenumber[0]),2500)
spectrum_new=f(keys_new)
df = pd.read_csv(file_path2)
y = df['1'].values
similarity = cosine_similarity(spectrum_new.reshape(1, -1), y.reshape(1, -1))[0, 0]
print(f'Cosine Similarity: {similarity}')