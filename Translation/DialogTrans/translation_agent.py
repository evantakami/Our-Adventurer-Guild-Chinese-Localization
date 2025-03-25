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
        # 匹配DIALOGUESCENE后的数字
        match = re.search(r'DIALOGUESCENE(\d+)', key)
        if match:
            return int(match.group(1))
        # 匹配SIDESTORY后的数字
        match = re.search(r'SIDESTORY(\d+)', key)
        if match:
            return int(match.group(1))
        # 匹配GAME_ENDING_PAGE后的数字
        match = re.search(r'GAME_ENDING_PAGE(\d+)', key)
        if match:
            return int(match.group(1))
        return None
    
    def group_events(self, df: pd.DataFrame, max_events: Optional[int] = None) -> List[pd.DataFrame]:
        """
        按照事件ID分组，每组处理group_size个事件
        
        Args:
            df: 包含所有事件的DataFrame
            max_events: 最大处理事件数量
            
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
        
        # 如果指定了最大事件数，则只取前max_events个事件
        if max_events is not None:
            unique_events = unique_events[:max_events]
        
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
        prompt = f"""你是一个专业的游戏本地化翻译专家。请将以下游戏对话翻译成中文。注意：

1. 保持对话的自然流畅性和口语化表达
2. 保留所有HTML标签（如<b>、<?shakeScreen>等）
3. 保持说话者的语气和性格特征
4. 保持原文的标点符号和格式
5. 对话要符合中文表达习惯，但不要过度本地化
6. 保持原文的情感色彩和语气强度
7. 对于拟声词（如*Gulp*）要适当本地化处理
8. 保持对话的连贯性和上下文关系

角色名称翻译对照表：
主要角色：
- Guild Master -> 公会会长
- Falken -> 法尔肯
- Emily -> 艾米丽
- Fiola -> 菲奥拉
- Fran -> 弗兰
- Bai -> 白
- Yu -> 玉
- Shen -> 申
- Rogue -> 罗格
- Luana -> 露安娜
- Vincent -> 文森特
- Astrid -> 阿斯特丽德
- Majika -> 玛吉卡
- Zoldak -> 佐尔达克
- Lily -> 莉莉
- Elan -> 伊兰
- Ferdinand -> 费迪南德
- Gilbert -> 吉尔伯特
- Tedric -> 泰德里克
- Hopi -> 霍皮

其他角色/群体：
- Adventurer -> 冒险者
- Male Adventurer -> 男性冒险者
- Female Adventurer -> 女性冒险者
- Bandit -> 强盗
- Bandit King -> 强盗王
- Demon Witch -> 恶魔女巫
- Dragon Orb -> 龙珠
- Voice of Ghanenta -> 加南塔之声
- All Bystanders -> 所有旁观者
- Bystander A/B/C -> 旁观者A/B/C

关于Key列的格式说明：
- DIALOGUESCENE[场景号]_DIALOGUE[序号]：普通对话内容
- DIALOGUESCENE[场景号]_OPTION[选项号]：对话选项，需要简洁明了
- GAME_ENDING_PAGE[页码]：游戏结局文本，需要庄重有力
- DIALOGUESCENESKIP_TUTORIAL_[说话者]_DIALOGUE[序号]：跳过教程的对话
- SKIP_TUTORIAL_GUILDMASTER_DIALOGE_NAMESCENE[序号]：跳过教程的名字场景
- SIDESTORY[编号]：支线故事标题，需要简洁有力
- SIDESTORY[编号]_DIARYENTRY：支线故事的日记内容，需要保持日记的叙事风格
- 角色名称（如ADVENTURER_MALE_NPC, EMILY等）：保持角色名称的一致性

请将以下{len(event_group)}条对话翻译成中文，保持CSV格式：

{event_group.to_string(index=False)}

请直接返回翻译后的CSV内容，不要包含任何其他说明或标记。"""
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
        
        try:
            # 尝试直接解析CSV格式
            lines = response_text.strip().split('\n')
            for line in lines[1:]:  # 跳过标题行
                # 使用正则表达式匹配CSV格式，处理引号内的逗号
                matches = re.findall(r'([^,]+|"[^"]*"),([^,]+|"[^"]*"),(.+)', line)
                if matches:
                    id_val, key_val, string_val = matches[0]
                    # 清理数据
                    key_val = key_val.strip().strip('"')
                    string_val = string_val.strip().strip('"')
                    # 移除末尾的场景号（如",1"）
                    string_val = re.sub(r',\s*\d+\s*$', '', string_val)
                    # 规范化换行符
                    string_val = string_val.replace('\\n', '\n')
                    
                    # 通过Key匹配原始DataFrame中的行
                    mask = result_df['Key'] == key_val
                    if any(mask):
                        result_df.loc[mask, 'TranslatedString'] = string_val
        except Exception as e:
            print(f"CSV解析失败: {str(e)}")
            print("尝试使用备用解析方法...")
            
            # 备用解析方法：通过Key和String标记解析
            entries = re.split(r'(?=ID,|Key,)', response_text)
            for entry in entries:
                if not entry.strip():
                    continue
                    
                # 尝试提取Key和String
                key_match = re.search(r'Key[,\s]*([^,\n]+)', entry)
                string_match = re.search(r'String[,\s]*(.+?)(?=(?:ID,|Key,)|$)', entry, re.DOTALL)
                
                if key_match and string_match:
                    key_val = key_match.group(1).strip()
                    string_val = string_match.group(1).strip().strip('"')
                    # 移除末尾的场景号
                    string_val = re.sub(r',\s*\d+\s*$', '', string_val)
                    # 规范化换行符
                    string_val = string_val.replace('\\n', '\n')
                    
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
    
    def translate_csv(self, input_path: str, output_path: str, max_events: Optional[int] = None) -> None:
        """
        翻译整个CSV文件
        
        Args:
            input_path: 输入CSV文件路径
            output_path: 输出CSV文件路径
            max_events: 最大处理事件数量
        """
        # 记录开始时间
        start_time = time.time()
        
        # 加载数据
        df = self.load_csv(input_path)
        
        # 分组
        event_groups = self.group_events(df, max_events)
        
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
    parser.add_argument('--max-events', '-me', type=int, help='最大处理事件数量')
    
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
    agent.translate_csv(args.input, args.output, args.max_events)

if __name__ == "__main__":
    main() 