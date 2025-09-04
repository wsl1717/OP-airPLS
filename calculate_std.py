#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算CSV文件中指定三列的总体标准差
Created on 2025-09-03
"""

import pandas as pd
import numpy as np
import os
import sys


def calculate_population_std(csv_file_path, column1, column2, column3):
    """
    计算CSV文件中指定三列的总体标准差
    
    Parameters:
    -----------
    csv_file_path : str
        CSV文件路径
    column1, column2, column3 : str or int
        要计算标准差的三列，可以是列名（字符串）或列索引（整数）
        
    Returns:
    --------
    dict : 包含各列统计信息的字典
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(csv_file_path):
            print(f"错误：文件 {csv_file_path} 不存在")
            return None
            
        # 读取CSV文件
        print(f"正在读取文件: {csv_file_path}")
        df = pd.read_csv(csv_file_path, encoding='utf-8')
        print(f"文件读取成功，共有 {df.shape[0]} 行, {df.shape[1]} 列")
        print(f"列名: {df.columns.tolist()}")
        
        # 获取指定的三列数据
        columns_to_process = [column1, column2, column3]
        selected_data = {}
        
        for i, col in enumerate(columns_to_process):
            try:
                if isinstance(col, str):
                    # 按列名获取
                    if col not in df.columns:
                        print(f"错误：找不到列名 '{col}'")
                        return None
                    selected_data[f'column_{i+1}'] = df[col]
                    print(f"列 {i+1}: '{col}' (按列名)")
                else:
                    # 按索引获取
                    if col >= df.shape[1] or col < 0:
                        print(f"错误：列索引 {col} 超出范围 (0-{df.shape[1]-1})")
                        return None
                    selected_data[f'column_{i+1}'] = df.iloc[:, col]
                    print(f"列 {i+1}: 索引 {col} ('{df.columns[col]}')")
                    
            except Exception as e:
                print(f"错误：处理列 {col} 时出错: {str(e)}")
                return None
        
        # 计算统计信息
        results = {}
        all_values = []  # 用于计算所有值的总体标准差
        
        for col_name, data in selected_data.items():
            # 检查数据类型并清理数据
            numeric_data = pd.to_numeric(data, errors='coerce')
            
            # 移除NaN值
            clean_data = numeric_data.dropna()
            
            if len(clean_data) == 0:
                print(f"警告：{col_name} 中没有有效的数值数据")
                continue
                
            # 计算各种统计量
            col_results = {
                'count': len(clean_data),
                'mean': float(clean_data.mean()),
                'population_std': float(clean_data.std(ddof=0)),  # ddof=0 表示总体标准差
                'sample_std': float(clean_data.std(ddof=1)),     # ddof=1 表示样本标准差
                'min': float(clean_data.min()),
                'max': float(clean_data.max()),
                'median': float(clean_data.median())
            }
            
            results[col_name] = col_results
            all_values.extend(clean_data.tolist())
        
        # 计算所有三列合并后的总体标准差
        if all_values:
            all_array = np.array(all_values)
            results['combined'] = {
                'count': len(all_array),
                'mean': float(np.mean(all_array)),
                'population_std': float(np.std(all_array, ddof=0)),
                'sample_std': float(np.std(all_array, ddof=1)),
                'min': float(np.min(all_array)),
                'max': float(np.max(all_array)),
                'median': float(np.median(all_array))
            }
        
        return results
        
    except Exception as e:
        print(f"错误：处理文件时出错: {str(e)}")
        return None


def print_results(results):
    """
    打印计算结果
    
    Parameters:
    -----------
    results : dict
        calculate_population_std函数返回的结果字典
    """
    if results is None:
        print("没有结果可显示")
        return
        
    print("\n" + "="*60)
    print("统计结果")
    print("="*60)
    
    # 打印各列的结果
    for col_name, stats in results.items():
        if col_name == 'combined':
            continue
            
        print(f"\n{col_name.upper()}:")
        print(f"  数据点数: {stats['count']}")
        print(f"  平均值: {stats['mean']:.6f}")
        print(f"  总体标准差: {stats['population_std']:.6f}")
        print(f"  样本标准差: {stats['sample_std']:.6f}")
        print(f"  最小值: {stats['min']:.6f}")
        print(f"  最大值: {stats['max']:.6f}")
        print(f"  中位数: {stats['median']:.6f}")
    
    # 打印合并结果
    if 'combined' in results:
        print(f"\n所有三列合并:")
        stats = results['combined']
        print(f"  总数据点数: {stats['count']}")
        print(f"  总平均值: {stats['mean']:.6f}")
        print(f"  总体标准差: {stats['population_std']:.6f}")
        print(f"  样本标准差: {stats['sample_std']:.6f}")
        print(f"  最小值: {stats['min']:.6f}")
        print(f"  最大值: {stats['max']:.6f}")
        print(f"  中位数: {stats['median']:.6f}")
    
    print("="*60)


def save_results_to_csv(results, output_path):
    """
    将结果保存到CSV文件
    
    Parameters:
    -----------
    results : dict
        计算结果
    output_path : str
        输出文件路径
    """
    if results is None:
        print("没有结果可保存")
        return
        
    # 转换为DataFrame格式
    rows = []
    for col_name, stats in results.items():
        row = {'Column': col_name}
        row.update(stats)
        rows.append(row)
    
    df_results = pd.DataFrame(rows)
    df_results.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n结果已保存到: {output_path}")


def main():
    """
    主函数 - 可以在这里修改参数进行测试
    """
    # ===== 在这里修改您的参数 =====
    
    # CSV文件路径
    csv_file = r"e:\Desktop\OP-airPLS\1.csv"
    
    # 要计算的三列，可以使用列名（字符串）或列索引（整数，从0开始）
    # 例如：使用列索引
    column1 = 0    # 第一列
    column2 = 1    # 第二列  
    column3 = 2    # 第三列
    
    # 或者使用列名（如果知道确切的列名）
    # column1 = "列名1"
    # column2 = "列名2" 
    # column3 = "列名3"
    
    # 输出文件路径（可选）
    output_file = r"e:\Desktop\OP-airPLS\std_calculation_results.csv"
    
    # ===== 参数设置结束 =====
    
    print("CSV文件标准差计算工具")
    print("="*40)
    
    # 计算标准差
    results = calculate_population_std(csv_file, column1, column2, column3)
    
    # 打印结果
    print_results(results)
    
    # 保存结果
    if results:
        save_results_to_csv(results, output_file)


if __name__ == "__main__":
    # 检查是否提供了命令行参数
    if len(sys.argv) > 1:
        print("命令行模式")
        if len(sys.argv) < 5:
            print("用法: python calculate_std.py <csv_file> <column1> <column2> <column3>")
            print("示例: python calculate_std.py data.csv 0 1 2")
            print("示例: python calculate_std.py data.csv 列名1 列名2 列名3")
            sys.exit(1)
        
        csv_file = sys.argv[1]
        
        # 尝试将参数转换为整数（列索引），如果失败则作为字符串（列名）
        columns = []
        for i in range(2, 5):
            try:
                columns.append(int(sys.argv[i]))
            except ValueError:
                columns.append(sys.argv[i])
        
        results = calculate_population_std(csv_file, columns[0], columns[1], columns[2])
        print_results(results)
        
        # 保存结果
        if results:
            output_file = csv_file.replace('.csv', '_std_results.csv')
            save_results_to_csv(results, output_file)
    else:
        # 运行默认的main函数
        main()
