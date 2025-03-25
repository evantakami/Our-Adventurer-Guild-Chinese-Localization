#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
游戏文本翻译Agent
这个脚本用于将游戏世界事件的CSV文件从英文翻译成中文
保留所有特殊标记和变量，确保翻译的质量和格式一致性
"""

import os
import re
import sys
import time
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 检查API密钥是否设置
if not os.getenv("OPENAI_API_KEY"):
    print("错误：未设置OPENAI_API_KEY环境变量")
    print("请在.env文件中设置OPENAI_API_KEY=你的API密钥")
    sys.exit(1)

class GameTranslationAgent:
    """游戏翻译Agent，处理CSV文件中的游戏文本翻译"""
    
    def __init__(self, 
                model_name: str = "gpt-4o-mini", 
                temperature: float = 0.1, 
                group_size: int = 3,
                retry_attempts: int = 3,
                max_tokens: int = 4000,
                api_wait_time: int = 20):
        """
        初始化翻译Agent
        
        Args:
            model_name: 使用的LLM模型名称
            temperature: 模型温度参数，越低越精确
            group_size: 每组处理的事件数量
            retry_attempts: 失败重试次数
            max_tokens: 每次调用的最大tokens数
            api_wait_time: API请求间隔时间(秒)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.group_size = group_size
        self.retry_attempts = retry_attempts
        self.max_tokens = max_tokens
        self.api_wait_time = api_wait_time
        
        print(f"初始化翻译Agent完成，使用模型: {model_name}")
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        加载CSV文件并进行初步处理
        
        Args:
            file_path: CSV文件路径
            
        Returns:
            处理后的DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            print(f"成功加载 {len(df)} 行数据")
            return df
        except Exception as e:
            print(f"加载CSV文件失败: {str(e)}")
            sys.exit(1)
    
    def extract_event_id(self, key: str) -> Optional[int]:
        """从Key中提取事件ID"""
        match = re.search(r'WORLDEVENT(\d+)', key)
        if match:
            return int(match.group(1))
        return None
    
    def group_events(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        按照事件ID分组，每组处理group_size个事件
        
        Args:
            df: 包含所有事件的DataFrame
            
        Returns:
            分组后的DataFrame列表
        """
        # 提取事件ID
        df['EventID'] = df['Key'].apply(self.extract_event_id)
        
        # 过滤掉无法识别事件ID的行
        df = df[df['EventID'].notna()]
        df['EventID'] = df['EventID'].astype(int)
        
        # 获取唯一事件ID列表
        unique_events = sorted(df['EventID'].unique())
        
        # 按group_size分组
        event_groups = [unique_events[i:i+self.group_size] for i in range(0, len(unique_events), self.group_size)]
        
        # 为每组事件创建DataFrame
        grouped_dfs = []
        for event_group in event_groups:
            group_df = df[df['EventID'].isin(event_group)].copy()
            grouped_dfs.append(group_df)
        
        print(f"将 {len(unique_events)} 个事件分为 {len(grouped_dfs)} 组")
        return grouped_dfs
    
    def create_translation_prompt(self, event_group: pd.DataFrame) -> str:
        """
        创建翻译prompt，包含Key和翻译要求
        
        Args:
            event_group: 一组事件的DataFrame
            
        Returns:
            翻译prompt
        """
        # 提取事件组的所有文本，不包含ID
        event_texts = []
        for _, row in event_group.iterrows():
            event_texts.append(f"Key: {row['Key']}\nString: {row['String']}")
        
        event_content = "\n\n".join(event_texts)
        
        # 创建翻译prompt - 注意这里不使用f-string来避免花括号被LangChain处理
        prompt = """
你是一名专业游戏本地化翻译专家，精通英文到中文的翻译，尤其擅长游戏内容的翻译。现在你需要翻译以下游戏世界事件的文本内容。

请严格遵循以下翻译要求：
1. 保持翻译的信、达、雅：准确传达原文意思，翻译流畅自然，符合中文表达习惯。
2. 保持游戏文本的风格和语气，确保翻译后不失原文的情感和表现力。
3. 翻译时必须保留所有特殊标记，如 "<?shakeScreen>"，"<?whiteFlash>" 等，这些是游戏效果标记。
4. 必须完整保留所有变量，如 "{CHARACTER1}"，"{GENDER1:cond:=1?his|=0?her}" 等，这些是游戏中的动态变量。
5. 对于条件变量如 {GENDER1:cond:=1?his|=0?her}，应该翻译变量中的文本部分，例如翻译为 {GENDER1:cond:=1?他的|=0?她的}，但保持变量结构不变。
6. 保留原文中的所有换行符、引号和分隔符，确保格式与原文完全一致。
7. 确保同一组事件中的术语、名称翻译一致，保持上下文连贯性。
8. 只翻译String列的内容，Key列保持不变。

关于Key列的格式和类型说明：
- Key列是事件的唯一标识符，格式为"WORLDEVENT数字_类型"
- 主要类型包括：
  - PAGE: 如"WORLDEVENT1_PAGE1" - 表示事件页面内容
  - DECISION: 如"WORLDEVENT1_DECISION1" - 表示玩家可选择的决策选项
  - RESULT: 如"WORLDEVENT1_DECISION1RESULT0_PAGE1" - 表示决策的结果
  - ALTRESULT: 如"WORLDEVENT1_DECISION1ALTRESULT0_PAGE1" - 表示决策的替代结果
  - CONDRESULT: 如"WORLDEVENT7_DECISION1CONDRESULT1_PAGE1" - 表示条件判断的结果
- 类型后面的数字表示序号，如PAGE1, PAGE2等
- Key必须完全保留并在输出中包含，不要翻译或修改Key

以下是需要翻译的文本内容：

"""
        # 使用字符串连接而不是f-string
        prompt = prompt + event_content + """

请按以下格式返回翻译结果：
Key: [原始Key]
String: [中文翻译]

对于每个条目，确保格式严格匹配，便于后续处理。
"""
        return prompt
    
    def translate_event_group(self, event_group: pd.DataFrame) -> pd.DataFrame:
        """
        翻译一组事件
        
        Args:
            event_group: 一组事件的DataFrame
            
        Returns:
            包含翻译结果的DataFrame
        """
        # 创建提示
        prompt = self.create_translation_prompt(event_group)
        
        # 创建结果DataFrame的副本
        result_df = event_group.copy()
        result_df['TranslatedString'] = None
        
        # 直接使用OpenAI API，而不是LangChain
        client = OpenAI()
        
        # 尝试翻译，支持重试
        for attempt in range(self.retry_attempts):
            try:
                print("正在调用OpenAI API进行翻译...")
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "你是一名专业游戏翻译专家。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # 解析翻译结果
                response_text = response.choices[0].message.content
                
                # 直接使用解析函数处理响应
                translation_results = self.parse_translation_response_text(response_text, result_df)
                
                # 将翻译结果合并到结果DataFrame
                for idx, row in translation_results.iterrows():
                    if not pd.isna(row.get('TranslatedString')):
                        result_df.loc[idx, 'TranslatedString'] = row['TranslatedString']
                
                # 验证翻译结果
                if self.validate_group_translation(result_df):
                    return result_df
                else:
                    print(f"翻译结果验证失败，尝试重新翻译（尝试 {attempt+1}/{self.retry_attempts}）")
            except Exception as e:
                print(f"翻译过程中出错: {str(e)}")
                print(f"重试中... ({attempt+1}/{self.retry_attempts})")
            
            # 避免API限制，等待一段时间
            if attempt < self.retry_attempts - 1:
                print(f"等待 {self.api_wait_time} 秒后重试...")
                time.sleep(self.api_wait_time)
        
        # 所有重试都失败，返回原始数据
        print("警告：所有重试都失败，将使用原始文本")
        return result_df
        
    def parse_translation_response_text(self, response_text: str, original_df: pd.DataFrame) -> pd.DataFrame:
        """
        解析文本响应（直接从OpenAI响应解析）
        
        Args:
            response_text: 响应文本
            original_df: 原始DataFrame
            
        Returns:
            包含翻译结果的DataFrame
        """
        # 创建结果DataFrame的副本
        result_df = original_df.copy()
        result_df['TranslatedString'] = None
        
        # 分割响应为每个翻译条目
        entries = re.split(r'Key: ', response_text)
        if len(entries) > 0:
            entries = entries[1:]  # 跳过第一个空元素
        
        for entry in entries:
            # 提取Key和翻译后的String
            key_end = entry.find('\n')
            if key_end == -1:
                continue
                
            key_val = entry[:key_end].strip()
            
            string_match = re.search(r'String: (.*?)(?=\nKey:|$)', entry, re.DOTALL)
            
            if string_match:
                string_val = string_match.group(1).strip()
                
                # 通过Key匹配原始DataFrame中的行
                mask = result_df['Key'] == key_val
                if any(mask):
                    result_df.loc[mask, 'TranslatedString'] = string_val
        
        return result_df

    def extract_special_elements(self, text: str) -> Tuple[List[str], List[str]]:
        """
        提取文本中的特殊标记和变量
        
        Args:
            text: 需要分析的文本
            
        Returns:
            标记列表和变量列表
        """
        # 提取标记 <?...>
        tags = re.findall(r'<\?[^>]+>', text)
        
        # 提取变量 {...}
        variables = re.findall(r'\{[^}]+\}', text)
        
        return tags, variables
    
    def validate_group_translation(self, translated_df: pd.DataFrame) -> bool:
        """
        验证翻译结果的质量和完整性
        
        Args:
            translated_df: 包含翻译结果的DataFrame
            
        Returns:
            验证是否通过
        """
        issues = []
        
        for idx, row in translated_df.iterrows():
            original_text = row['String']
            translated_text = row.get('TranslatedString', '')
            
            if pd.isna(translated_text) or translated_text == '':
                issues.append(f"行 {idx}: 翻译为空")
                continue
                
            # 提取原文的特殊元素
            original_tags, original_vars = self.extract_special_elements(original_text)
            
            # 提取翻译文本的特殊元素
            translated_tags, translated_vars = self.extract_special_elements(translated_text)
            
            # 检查标记是否完整保留
            for tag in original_tags:
                if tag not in translated_text:
                    issues.append(f"行 {idx} ({row['Key']}): 缺少标记 {tag}")
            
            # 变量验证部分已移除
                    
        if issues:
            print(f"发现 {len(issues)} 个问题:")
            for issue in issues[:10]:  # 只显示前10个问题，避免输出过多
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... 以及其他 {len(issues) - 10} 个问题")
            return False
        
        return True
    
    def translate_csv(self, input_path: str, output_path: str) -> None:
        """
        翻译整个CSV文件
        
        Args:
            input_path: 输入CSV文件路径
            output_path: 输出CSV文件路径
        """
        # 记录开始时间
        start_time = time.time()
        
        # 加载数据
        df = self.load_csv(input_path)
        
        # 分组
        event_groups = self.group_events(df)
        
        # 初始化结果DataFrame
        result_df = df.copy()
        result_df['TranslatedString'] = None
        
        # 逐组翻译
        total_groups = len(event_groups)
        for i, group in enumerate(event_groups):
            print(f"正在翻译第 {i+1}/{total_groups} 组事件...")
            
            translated_group = self.translate_event_group(group)
            
            # 将翻译结果合并到主DataFrame
            for idx, row in translated_group.iterrows():
                if not pd.isna(row.get('TranslatedString')):
                    result_df.loc[idx, 'TranslatedString'] = row['TranslatedString']
            
            # 显示进度
            elapsed_time = time.time() - start_time
            avg_time_per_group = elapsed_time / (i + 1)
            remaining_groups = total_groups - (i + 1)
            estimated_remaining_time = avg_time_per_group * remaining_groups
            
            print(f"完成: {i+1}/{total_groups} 组 ({(i+1)/total_groups*100:.1f}%)")
            print(f"已用时间: {elapsed_time/60:.1f} 分钟")
            print(f"预计剩余时间: {estimated_remaining_time/60:.1f} 分钟")
            
            # 中间保存结果，避免意外中断导致全部丢失
            if (i + 1) % 5 == 0 or i == total_groups - 1:
                temp_df = pd.DataFrame({
                    'ID': result_df['ID'],
                    'Key': result_df['Key'],
                    'String': result_df['TranslatedString'].fillna(result_df['String'])
                })
                temp_output = output_path.replace('.csv', f'_temp_{i+1}.csv')
                temp_df.to_csv(temp_output, index=False)
                print(f"已保存临时结果到 {temp_output}")
        
        # 准备输出数据
        output_df = pd.DataFrame({
            'ID': result_df['ID'],
            'Key': result_df['Key'],
            'String': result_df['TranslatedString'].fillna(result_df['String'])  # 使用原文作为备选
        })
        
        # 保存结果
        output_df.to_csv(output_path, index=False)
        print(f"翻译完成，结果已保存到 {output_path}")
        
        # 统计翻译完成率
        total_rows = len(result_df)
        translated_rows = result_df['TranslatedString'].notna().sum()
        completion_rate = translated_rows / total_rows * 100
        
        print(f"总行数: {total_rows}")
        print(f"翻译行数: {translated_rows}")
        print(f"翻译完成率: {completion_rate:.2f}%")
        print(f"总耗时: {(time.time() - start_time)/60:.1f} 分钟")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='游戏文本翻译Agent')
    parser.add_argument('--input', '-i', required=True, help='输入CSV文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出CSV文件路径')
    parser.add_argument('--model', '-m', default='gpt-4', help='LLM模型名称')
    parser.add_argument('--temp', '-t', type=float, default=0.1, help='模型温度参数')
    parser.add_argument('--group-size', '-g', type=int, default=3, help='每组处理的事件数量')
    parser.add_argument('--retry', '-r', type=int, default=3, help='失败重试次数')
    parser.add_argument('--wait', '-w', type=int, default=20, help='API请求间隔时间(秒)')
    
    args = parser.parse_args()
    
    # 创建翻译Agent
    agent = GameTranslationAgent(
        model_name=args.model,
        temperature=args.temp,
        group_size=args.group_size,
        retry_attempts=args.retry,
        api_wait_time=args.wait
    )
    
    # 执行翻译
    agent.translate_csv(args.input, args.output)

if __name__ == "__main__":
    main() 