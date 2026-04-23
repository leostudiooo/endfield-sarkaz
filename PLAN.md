# 萨卡兹语端到端翻译系统实施计划

## Context

现有 `endfield-sarkaz` 项目通过 MCP 服务器暴露多个工具（`search_exact`、`search_single_chinese_character`、`convert` 等）供 LLM 调用以翻译萨卡兹语。该方案需要多轮工具调用，每次调用都消耗大量 token，成本高且延迟大。

本计划目标：训练一个定制 tokenizer + 微调 Qwen3 模型，实现**端到端萨卡兹语翻译**，将多次 MCP 工具调用压缩为单次模型推理。同时用 Aho-Corasick 本地预处理注入专有名词线索，在零额外 token 成本下提升消歧能力。

## 推荐方案

### 第一阶段：本地验证（Tokenizer + 0.6B/1.7B 基础微调）

#### 1.1 项目结构调整

```
endfield-sarkaz/
├── corpus/                          # 语料目录
│   ├── raw/                         # 原始中文语料
│   │   ├── wiki2019zh.json          # 已有（标注为无效，需验证清洗）
│   │   ├── web_chinese/             # 后续下载的公开语料
│   │   └── endfield/                # 终末地剧情文本（预留，第二阶段用）
│   └── skz_parallel/               # 生成的平行语料
│       ├── base/                    # 通用平行语料（第一阶段）
│       └── endfield/                # 领域平行语料（第二阶段）
├── training/
│   ├── __init__.py
│   ├── tokenizer_train.py          # SentencePiece 训练 + 词表合并
│   ├── merge_tokenizer.py          # Qwen3 tokenizer + 自定义 token 合并
│   ├── data_generator.py           # 中文 → 萨卡兹语平行数据生成
│   ├── base_train.py               # 本地基础微调脚本
│   └── cloud_train.py              # 云端正式训练脚本（预留）
├── inference/
│   ├── __init__.py
│   ├── trie_builder.py             # 构建 Aho-Corasick 自动机
│   └── sarkaz_decoder.py           # 端到端解码器（AC 扫描 + 模型推理）
├── scripts/
│   ├── generate_corpus.py          # 语料下载与预处理
│   └── evaluate.py                 # CER / 专有名词命中率评估
├── models/                          # 训练产出（gitignore）
│   ├── tokenizer/                   # 定制 tokenizer
│   ├── base_model/                  # 基础微调模型
│   └── lora/                        # 领域 LoRA（第二阶段）
├── requirements.txt
└── pyproject.toml
```

#### 1.2 Tokenizer 改造

**核心思路**：Qwen3 原生使用 tiktoken byte-level BPE（非 SentencePiece），词表大小 ~151,645。本方案用 SentencePiece 在混合语料上训练新 token 的字符串表示，再通过 HuggingFace `AutoTokenizer.add_tokens()` 注入 Qwen3 tokenizer——字符串级合并不依赖底层分词算法。在**中文原文 + 萨卡兹密文混合语料**上训练一个领域 SPM，提取不在原词表中的新 token，合并后扩展词表。

**具体步骤**：

1. **语料准备**（`training/data_generator.py`）：
   - 输入：`wordlist.txt`（56k 词）、`single_char.txt`（3k 字）、wiki2019zh 字频数据（用于频率加权采样）
   - 对每条中文文本，生成对应的萨卡兹密文（复用现有 `convertCharsToSKZ` 算法）
   - 混合格式：`[中文原文]\n[萨卡兹密文]\n` 或交替段落
   - 同时保留纯中文语料，防止 tokenizer 只学会密文模式

2. **训练自定义 SPM**（`training/tokenizer_train.py`）：
   - 工具：`sentencepiece` 的 `spm.SentencePieceTrainer.train()`
   - 参数：`model_type='unigram'`，`vocab_size=16384`（依据：wordlist.txt 56k 词去重后取高频子集）
   - 输入：混合语料文件
   - 输出：`custom_sarkaz.model`、`custom_sarkaz.vocab`

3. **词表合并**（`training/merge_tokenizer.py`）：
   - 加载 Qwen3 tokenizer（`AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")`）
   - 加载自定义 SPM，提取其中不在 Qwen3 词表的 token
   - 过滤规则：排除纯拉丁字母、数字、常见标点（只保留中/萨卡兹相关 token）
   - 将新 token 加入 Qwen3 tokenizer，`resize_token_embeddings()`
   - **Embedding 初始化**：新 token 的 embedding 用其被 Qwen3 tokenizer 切分后的 subtoken embedding 的**均值**初始化（mean subtoken initialization）
   - 保存：`models/tokenizer/`

4. **专有名词注入**：
   - 读取 `endfield_words.txt`，将其及对应的萨卡兹编码作为 `AddedToken` 强制注入
   - 确保「管理员→sbt」、「佩丽卡→...」等关键映射不会被切分

**关键参考**：AraToken 论文 (arXiv:2512.18399) 验证了在 Qwen3-0.6B 上通过词表扩展 + mean subtoken 初始化，800 steps 即可将 eval loss 从 8.28 降到 2.43。注意：该结果针对阿拉伯语（低资源语言扩展），萨卡兹语场景（极高碰撞率）的效果可能不同，需以本地实验为准。

#### 1.3 第一阶段基础微调（本地验证）

**数据合成**（`training/data_generator.py`）：
- 从 `wordlist.txt` 中随机采样 2-10 个词组成短句（模拟自然语言片段）
- 对 wordlist.txt 中的句子进行过滤（长度 20-200 字），保留高质量片段
- 每条中文句子生成对应的萨卡兹密文
- **SFT 格式**：
  ```json
  {
    "messages": [
      {"role": "system", "content": "请将以下萨卡兹密文翻译为中文。"},
      {"role": "user", "content": "ytqqrvqbsbtjyrernx"},
      {"role": "assistant", "content": "你好，管理员，是来提交活动菜品的吗？"}
    ]
  }
  ```
- 数据量：本地验证 1~5 万条，云端正式 20~50 万条

**训练配置**（`training/base_train.py`）：
- 基座：`Qwen/Qwen3-0.6B-Instruct`（本地快速验证）→ `Qwen/Qwen3-1.7B-Instruct`
- 训练方式：
  - 方案 A（推荐）：**冻结原词表 embedding**（保留 Qwen3 通用知识），只训练新 token embedding + LoRA（rank=64, alpha=128）
  - 方案 B：全参数微调（如果显存允许且效果不佳时尝试）
- 使用 `transformers.Trainer` 或 `trl.SFTTrainer`
- 超参：lr=2e-4, batch_size=4, gradient_accumulation=4, epochs=3, warmup_ratio=0.1
- 输出：`models/base_model/`

**验证标准**：在本地验证集上 CER < 30% 即视为路径可行，可以上云端。

---

### 第二阶段：云端正式训练（Qwen3-4B）

#### 2.1 扩大语料规模

- 下载更大规模的公开中文语料（HPLT v4 中文子集、zhTenTen 过滤后的文本）
- 合成 20~50 万条平行语料
- 引入噪声数据（5% 的样本中随机删除/替换单个萨卡兹字母，模拟 OCR 误差）

#### 2.2 云端训练

- 基座：`Qwen/Qwen3-4B-Instruct`
- 复用第一阶段训练好的定制 tokenizer
- 训练方式：LoRA（rank=128, alpha=256）+ 新 token embedding 训练
- 输出：`models/base_model_qwen3_4b/`

---

### 第三阶段：领域 LoRA（延后执行）

当终末地剧情文本到位后：

1. **数据合成**：剧情原文 → 萨卡兹密文，构建 5k~2 万条领域平行语料
2. **训练**：在第二阶段模型基础上，训练可热插拔的 LoRA 模块
3. **输出**：`models/lora/endfield_lora/`

---

### 第四阶段：替代 MCP RAG 的推理 Pipeline

#### 4.1 Aho-Corasick 预处理（`inference/trie_builder.py`）

- 输入：`endfield_words.txt` + `favorite_words.txt` + 扩展的领域词库
- 构建 `pyahocorasick.Automaton`，key 为萨卡兹密文，value 为对应中文词列表
- 保存 automaton 到磁盘（pickle），启动时加载

#### 4.2 端到端解码器（`inference/sarkaz_decoder.py`）

**Pipeline**：
```python
def decode(skz_text: str) -> str:
    # 1. AC 自动机扫描密文，找出所有可能匹配的词
    hints = []
    for end_index, value in ac_automaton.iter(skz_text):
        hints.append((end_index, value))
    # 2. 构建 system prompt，注入候选词线索
    system_prompt = build_prompt_with_hints(hints)
    # 3. 单次调用微调模型生成翻译
    response = model.chat(system_prompt, skz_text)
    return response
```

**Prompt 设计**：
```
系统：你是萨卡兹语翻译专家。请将以下萨卡兹密文翻译为中文。
【翻译线索】密文中检测到以下可能的词汇，请结合上下文选择：
- sbt -> [管理员, 某新机制A]
- jy -> [某新角色B]

用户：ytqqrvqbsbtjyrernx
```

#### 4.3 与现有 MCP Server 的整合

保留 MCP 架构但精简为**单个工具**：
- 移除 `search_exact`、`search_single_chinese_character` 等多个工具
- 新增 `translate_sarkaz(skz_text: str) -> str`，内部直接调用 `sarkaz_decoder.decode()`
- LLM 只需一次工具调用即可获得翻译结果

---

## 关键文件与复用点

| 现有文件 | 复用方式 |
|---------|---------|
| `vendors/sarkaz_tools/HumanTools/sarkazEncoder.py` | 提取 `convertCharsToSKZ` 算法到 `training/data_generator.py` |
| `vendors/sarkaz_tools/LLMTools/MCPServer/endfield_words.txt` | Tokenizer 强制保留 token + AC 自动机构建 |
| `vendors/sarkaz_tools/LLMTools/MCPServer/favorite_words.txt` | AC 自动机动态词库 |
| `vendors/sarkaz_tools/LLMTools/MCPServer/wordlist.txt` | 平行语料生成的词源 |
| `vendors/sarkaz_tools/LLMTools/MCPServer/single_char.txt` | 平行语料生成的字源 |
| `vendors/sarkaz_tools/LLMTools/MCPServer/Sarkaz_tools.py` | 参考现有工具设计新 MCP Tool 接口 |

## 验证方案

1. **Tokenizer 验证**：
   - 检查「管理员」是否被切分为单个 token（或 `管理`+`员` 而非碎成多个无意义片段）
   - 检查「sbt」是否被切分为单个 token

2. **模型验证**：
   - 字级别准确率（CER）：目标 < 15%（云端 4B 版本）
   - 专有名词命中率：对 `endfield_words.txt` 中的词进行编码-解码测试，目标 100%
   - 人工抽查 50 条测试样例

3. **Pipeline 验证**：
   - 对比旧 MCP 方案（多工具调用）与新方案（单次调用）的 token 消耗
   - 对比翻译质量（盲测打分）

## 依赖

```
transformers>=4.45.0
torch>=2.4.0
datasets
sentencepiece
tokenizers
peft
accelerate
pyahocorasick
tiktoken
unsloth  # 可选，用于加速训练
```

## 风险与备选

- **Tokenizer 改造风险**：若扩展词表后模型无法收敛，可回退到「不改造 tokenizer，只微调」的 baseline 方案
- **语料不足风险**：wiki2019zh 如确实无效，需从 HPLT 或 C-Eval 等替代源获取
- **模型选择**：如 Qwen3-4B 效果不佳，备选 `Qwen2.5-7B-Instruct`
