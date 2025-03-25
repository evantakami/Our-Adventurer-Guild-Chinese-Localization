# 冒险者公会中文本地化 - 世界事件翻译

这个文件夹包含了用于批量翻译"我们的冒险者公会"游戏世界事件的代码和相关文件。

## 文件夹结构

- `parallel_translation.py` - 并行翻译处理主脚本
- `translation_agent.py` - 翻译代理类，负责实际翻译处理
- `run_parallel_test.py` - 用于测试并行翻译的脚本
- `requirements.txt` - 项目依赖文件
- `.env.example` - 环境变量示例文件，需要重命名为.env并填入API密钥
- `templates/` - 原始英文模板文件
- `results/` - 翻译结果输出目录
- `temp_translations/` - 临时翻译结果，用于断点续传

## 使用方法

1. 创建并激活虚拟环境（可选）:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

3. 复制`.env.example`为`.env`并填入您的OpenAI API密钥:
```bash
cp .env.example .env
# 然后编辑.env文件
```

4. 运行并行翻译:
```bash
python parallel_translation.py --input "templates/WorldEvents.csv" --output "results/WorldEvents_zh.csv" --batch-size 3 --workers 3 --model "gpt-4o-mini" --retry 2
```

参数说明:
- `--input`: 输入CSV文件路径
- `--output`: 输出CSV文件路径
- `--batch-size`: 每批处理的事件数量
- `--workers`: 最大并行工作线程数
- `--model`: 使用的LLM模型名称
- `--retry`: 失败重试次数
- `--max-events`: 最大处理事件数量(可选)

## 注意事项

- 翻译过程会消耗API额度，请确保账户有足够的额度
- 使用更多的工作线程可以加速翻译，但会增加API并发请求
- 临时结果会保存在temp_translations文件夹中，可用于断点续传 