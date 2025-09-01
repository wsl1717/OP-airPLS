#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的余弦相似度计算器
快速计算CSV文件中两列数值的余弦相似度
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cosine_sim(file_path, col1, col2):
    """
    计算CSV文件中两列的余弦相似度
    
    Args:
        file_path (str): CSV文件路径
        col1 (str): 第一列名称
        col2 (str): 第二列名称
    
    Returns:
        float: 余弦相似度值
    """
    # 读取数据
    df = pd.read_csv(file_path)
    
    # 提取两列数据
    x = df[col1].values
    y = df[col2].values
    
    # 移除NaN值
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    # 计算余弦相似度
    similarity = cosine_similarity(x_clean.reshape(1, -1), y_clean.reshape(1, -1))[0, 0]
    
    return similarity


def manual_cosine_sim(x, y):
    """
    手动计算余弦相似度
    
    Args:
        x, y: 两个向量
    
    Returns:
        float: 余弦相似度值
    """
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    
    if norm_x == 0 or norm_y == 0:
        return 0
    
    return dot_product / (norm_x * norm_y)


# 示例使用
if __name__ == "__main__":
    # 示例1: 使用工作空间中的数据文件
    try:
        # 读取一个示例文件查看结构
        sample_file = r"e:\Desktop\OP-airPLS\2. ML Part\results\0\0_results.csv"
        df_sample = pd.read_csv(sample_file)
        
        print("示例文件结构:")
        print(f"列名: {list(df_sample.columns)}")
        print(f"数据形状: {df_sample.shape}")
        print("\n前5行数据:")
        print(df_sample.head())
        
        # 如果文件只有两列，直接计算它们的相似度
        if len(df_sample.columns) == 2:
            col1, col2 = df_sample.columns[0], df_sample.columns[1]
            similarity = cosine_sim(sample_file, col1, col2)
            print(f"\n列 '{col1}' 和 '{col2}' 的余弦相似度: {similarity:.6f}")
        
    except Exception as e:
        print(f"读取示例文件时出错: {e}")
    
    print("\n" + "="*50)
    print("使用方法:")
    print("from cosine_similarity_simple import cosine_sim")
    print("similarity = cosine_sim('file.csv', 'column1', 'column2')")
    print("="*50)
