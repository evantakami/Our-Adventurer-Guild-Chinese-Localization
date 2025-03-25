#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
并行翻译处理脚本
每批处理两个事件，使用多线程并行执行
"""

import os
import sys
import time
import pandas as pd
import concurrent.futures
from typing import List, Tuple
from translation_agent import GameTranslationAgent

# 全局变量用于记录进度
processed_groups = 0
total_groups = 0
start_time = None

def translate_event_group(agent: GameTranslationAgent, group: pd.DataFrame, group_index: int) -> Tuple[int, pd.DataFrame]:
    """
    翻译一个事件组
    
    Args:
        agent: 翻译Agent
        group: 事件组DataFrame
        group_index: 组索引
        
    Returns:
        组索引和翻译结果DataFrame
    """
    global processed_groups
    
    try:
        print(f"开始翻译第 {group_index+1}/{total_groups} 组事件...")
        translated_group = agent.translate_event_group(group)
        
        # 更新进度计数器
        processed_groups += 1
        
        # 计算并显示进度
        elapsed_time = time.time() - start_time
        avg_time_per_group = elapsed_time / processed_groups
        remaining_groups = total_groups - processed_groups
        estimated_remaining_time = avg_time_per_group * remaining_groups
        
        print(f"完成: {processed_groups}/{total_groups} 组 ({processed_groups/total_groups*100:.1f}%)")
        print(f"已用时间: {elapsed_time/60:.1f} 分钟")
        print(f"预计剩余时间: {estimated_remaining_time/60:.1f} 分钟")
        
        return group_index, translated_group
    except Exception as e:
        print(f"翻译组 {group_index+1} 时出错: {str(e)}")
        return group_index, group  # 出错时返回原始组

def run_parallel_translation(
    input_file: str, 
    output_file: str, 
    batch_size: int = 2,
    max_workers: int = 3,
    model_name: str = "gpt-4o-mini",
    retry_attempts: int = 2,
    max_events: int = None
):
    """
    执行并行翻译
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        batch_size: 每批处理的事件数量
        max_workers: 最大并行工作线程数
        model_name: 使用的模型名称
        retry_attempts: 重试次数
        max_events: 最大处理事件数量，None表示处理所有事件
    """
    global processed_groups, total_groups, start_time
    
    print(f"加载数据文件: {input_file}")
    try:
        df = pd.read_csv(input_file)
        print(f"成功加载 {len(df)} 行数据")
    except Exception as e:
        print(f"加载CSV文件失败: {str(e)}")
        sys.exit(1)
    
    # 创建翻译Agent
    agent = GameTranslationAgent(
        model_name=model_name,
        temperature=0.3,
        group_size=batch_size,  # 每组batch_size个事件
        retry_attempts=retry_attempts,
        api_wait_time=5
    )
    
    # 提取事件ID
    df['EventID'] = df['Key'].apply(lambda x: agent.extract_event_id(x))
    
    # 过滤掉无法识别事件ID的行
    df = df[df['EventID'].notna()]
    df['EventID'] = df['EventID'].astype(int)
    
    # 获取唯一事件ID列表并根据max_events限制数量
    unique_events = sorted(df['EventID'].unique())
    if max_events is not None:
        unique_events = unique_events[:max_events]
    
    print(f"将处理 {len(unique_events)} 个事件")
    
    # 按batch_size分组事件
    event_batches = [unique_events[i:i+batch_size] for i in range(0, len(unique_events), batch_size)]
    
    # 为每组事件创建DataFrame
    event_groups = []
    for batch in event_batches:
        group_df = df[df['EventID'].isin(batch)].copy()
        event_groups.append(group_df)
    
    # 设置全局变量
    processed_groups = 0
    total_groups = len(event_groups)
    start_time = time.time()
    
    # 初始化结果DataFrame
    result_df = df[df['EventID'].isin(unique_events)].copy()
    result_df['TranslatedString'] = None
    
    # 创建临时目录，用于保存中间结果
    os.makedirs('temp_translations', exist_ok=True)
    
    print(f"开始并行翻译，使用 {max_workers} 个工作线程...")
    
    # 使用ThreadPoolExecutor进行并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_group = {
            executor.submit(translate_event_group, agent, group, i): i 
            for i, group in enumerate(event_groups)
        }
        
        # 处理完成的任务
        for future in concurrent.futures.as_completed(future_to_group):
            group_index = future_to_group[future]
            try:
                group_index, translated_group = future.result()
                
                # 将翻译结果合并到主DataFrame
                for idx, row in translated_group.iterrows():
                    if not pd.isna(row.get('TranslatedString')):
                        result_df.loc[idx, 'TranslatedString'] = row['TranslatedString']
                
                # 保存中间结果
                temp_df = translated_group.copy()
                temp_output = f"temp_translations/group_{group_index+1}.csv"
                temp_df.to_csv(temp_output, index=False)
                print(f"已保存组 {group_index+1} 的临时结果到 {temp_output}")
                
            except Exception as e:
                print(f"处理组 {group_index+1} 的结果时出错: {str(e)}")
    
    # 准备输出数据
    output_df = pd.DataFrame({
        'ID': result_df['ID'],
        'Key': result_df['Key'],
        'String': result_df['TranslatedString'].fillna(result_df['String'])  # 使用原文作为备选
    })
    
    # 保存结果
    output_df.to_csv(output_file, index=False)
    print(f"翻译完成，结果已保存到 {output_file}")
    
    # 统计翻译完成率
    total_rows = len(result_df)
    translated_rows = result_df['TranslatedString'].notna().sum()
    completion_rate = translated_rows / total_rows * 100
    
    total_time = time.time() - start_time
    print(f"总行数: {total_rows}")
    print(f"翻译行数: {translated_rows}")
    print(f"翻译完成率: {completion_rate:.2f}%")
    print(f"总耗时: {total_time/60:.1f} 分钟")
    print(f"平均每组耗时: {total_time/total_groups:.1f} 秒")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='并行游戏文本翻译')
    parser.add_argument('--input', '-i', required=True, help='输入CSV文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出CSV文件路径')
    parser.add_argument('--batch-size', '-b', type=int, default=2, help='每批处理的事件数量')
    parser.add_argument('--workers', '-w', type=int, default=3, help='最大并行工作线程数')
    parser.add_argument('--model', '-m', default='gpt-3.5-turbo', help='LLM模型名称')
    parser.add_argument('--retry', '-r', type=int, default=2, help='失败重试次数')
    parser.add_argument('--max-events', '-e', type=int, default=None, help='最大处理事件数量')
    
    args = parser.parse_args()
    
    run_parallel_translation(
        input_file=args.input,
        output_file=args.output,
        batch_size=args.batch_size,
        max_workers=args.workers,
        model_name=args.model,
        retry_attempts=args.retry,
        max_events=args.max_events
    )

if __name__ == "__main__":
    main() 