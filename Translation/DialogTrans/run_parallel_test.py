#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
并行翻译测试脚本
翻译前6个事件，每批2个事件，使用3个工作线程并行处理
"""

from parallel_translation import run_parallel_translation

def main():
    # 配置参数
    input_file = 'CustomTranslationTemplate/WorldEvents.csv'
    output_file = 'CustomTranslation/WorldEvents_parallel_zh.csv'
    
    # 运行并行翻译
    run_parallel_translation(
        input_file=input_file,
        output_file=output_file,
        batch_size=2,      # 每批处理2个事件
        max_workers=3,     # 最大3个工作线程
        model_name='gpt-3.5-turbo',
        retry_attempts=2,
        max_events=6       # 只处理前6个事件
    )

if __name__ == "__main__":
    main() 