# AGENTS.md — endfield-sarkaz

> 本文件面向 AI 编码助手。项目内的注释与文档主要使用中文，因此本文件亦以中文撰写。

---

## 项目概述

`endfield-sarkaz` 是一个围绕《明日方舟：终末地》中"萨卡兹语"展开的研究型仓库。萨卡兹语本质上是一种将中文字符通过 Unicode 取模后映射到 26 个英文字母上的特殊字体/编码。项目目标包括：

- [x] 收集并清洗《明日方舟》中文语料
- [x] 将语料转换为萨卡兹语平行数据集
- [x] 本地微调 Qwen3-0.6B 模型 smoke test，验证训练流程
- [ ] 微调 Qwen3-4B 左右小模型的 tokenizer 与模型，实现端到端萨卡兹语→中文翻译
- [ ] 收集和合成《明日方舟：终末地》语料，训练 LoRA 模型实现终末地专有名词的翻译

当前阶段已实现：
- 完整语料清洗与平行数据生成 pipeline
- SPM tokenizer 训练 + 中文 token 投影到密文
- LoRA 微调 Qwen3-0.6B，本地验证通过
- 基于 MCP（Model Context Protocol）协议的 LLM 工具服务器

---

## 仓库结构

```
.
├── README.md                     # 项目根说明文档
├── pyproject.toml                # uv 依赖管理与 CLI 命令入口
├── AGENTS.md                     # 本文件
├── PLAN.md                       # 端到端翻译系统实施计划
├── FINE_TUNING_PLAN.md           # 微调策略详解
├── corpus/                       # 语料目录
│   ├── raw/                      # 原始语料
│   │   └── ak/                   # 明日方舟语料
│   │       ├── arknights_stories.json    # 结构化剧情
│   │       ├── arknights_stories.txt     # 纯文本剧情
│   │       ├── arknights_sentences.txt   # 每行一句（未清洗）
│   │       └── arknights_cleaned.txt     # 清洗后的句子（333k）
│   └── skz_parallel/             # 萨卡兹语平行语料
│       └── base/
│           ├── train.jsonl       # SFT 训练数据
│           ├── valid.jsonl       # 验证数据
│           └── tokenizer_mix.txt # 中密交替文本（SPM 训练用）
├── scripts/                      # 脚本工具
│   ├── clean_corpus.py           # 语料清洗（去噪声、去重）
│   ├── extract_corpus.py         # 从 ArknightsGameData 提取剧情
│   ├── evaluate.py               # 模型评估脚本
│   ├── local_qwen3_verify.sh     # Qwen3 验证脚本
│   └── local_tiny_verify.sh      # TinyGPT2 smoke test
├── training/                     # 模型训练相关
│   ├── __init__.py
│   ├── common.py                 # 共用函数（convert_chars_to_skz）
│   ├── data_generator.py         # 平行语料生成（支持 corpus/wordlist）
│   ├── tokenizer_train.py        # SPM tokenizer 训练
│   ├── merge_tokenizer.py        # SPM + 投影 + 硬编码 → 合并词表
│   ├── base_train.py             # 本地 LoRA 训练
│   └── cloud_train.py            # 云端训练（预留）
├── inference/                    # 推理相关（预留）
│   └── sarkaz_decoder.py         # 端到端解码器
├── models/                       # 模型输出目录（gitignore）
│   ├── tokenizer/
│   │   ├── custom_sarkaz.model/.vocab  # SPM 输出
│   │   └── merged/               # 合并后的 Qwen3 tokenizer
│   └── base_model/
│       ├── resized/              # 词表扩展后的模型
│       └── qwen3_0_6b_verify/    # LoRA 训练输出
└── vendors/
    ├── ASTR-Script/              # 明日方舟剧情文本提取工具
    ├── ArknightsGameData/        # 明日方舟游戏数据（子模块）
    └── sarkaz_tools/             # 第三方工具集
        ├── HumanTools/           # 人工使用的 CLI 工具
        └── LLMTools/             # 面向大语言模型的工具
            └── MCPServer/
                ├── Sarkaz_tools.py       # MCP Server 主入口
                ├── endfield_words.txt    # 终末地专有名词
                ├── favorite_words.txt    # 用户收藏词表
                ├── single_char.txt       # 常用单字
                └── wordlist.txt          # 常用词表
```

---

## 技术栈与依赖

- **语言**：Python 3.11+
- **包管理**：uv（现代 Python 包管理器，替代 venv/pip）
- **核心依赖**（见 `pyproject.toml`）：
  - `torch>=2.4.0`
  - `transformers>=4.45.0`
  - `peft>=0.13.0`
  - `datasets>=3.0.0`
  - `sentencepiece>=0.2.0`
  - `accelerate>=0.34.2`
  - `pyahocorasick>=2.1.0`

---

## CLI 命令（uv run）

项目通过 `pyproject.toml` 注册了以下命令：

| 命令 | 功能 |
|------|------|
| `skz-generate-data` | 生成萨卡兹平行语料（支持 `--corpus` 或 wordlist） |
| `skz-train-tokenizer` | 训练 SentencePiece tokenizer |
| `skz-merge-tokenizer` | 合并 SPM+投影+硬编码 token 到 Qwen3 |
| `skz-train-base` | LoRA 微调 Qwen3 |
| `skz-build-trie` | 构建 Aho-Corasick 自动机（预留） |
| `skz-decode` | 端到端解码（预留） |
| `skz-eval` | 模型评估（预留） |

---

## 完整 Pipeline

### 1. 语料清洗

```bash
uv run python scripts/clean_corpus.py
```

输出：`corpus/raw/ak/arknights_cleaned.txt`（333k 行）

### 2. 平行语料生成

```bash
# 用清洗后的真实句子
uv run skz-generate-data --corpus corpus/raw/ak/arknights_cleaned.txt --num-samples 50000

# 或从 wordlist 随机合成
uv run skz-generate-data --num-samples 5000
```

输出：`corpus/skz_parallel/base/` 下的 `train.jsonl`、`valid.jsonl`、`tokenizer_mix.txt`

### 3. Tokenizer 构建

```bash
# 训练 SPM
uv run skz-train-tokenizer

# 合并词表（含投影）
uv run skz-merge-tokenizer --init-embeddings
```

输出：`models/tokenizer/merged/`（词表 168k，含 16k 新 token）

### 4. 模型训练

```bash
HF_ENDPOINT=https://hf-mirror.com uv run skz-train-base \
    --model-name models/base_model/resized \
    --tokenizer-path models/tokenizer/merged \
    --max-train-samples 49000 \
    --max-steps 1000
```

---

## 核心算法

### 萨卡兹编码

```python
SARKAZ_TABLE = "gkamztlbdqiyfucxbhsjoprnweygtjmevchdxsanqolkrvwiypjzquhe"
def convert_chars_to_skz(text: str) -> str:
    return "".join(SARKAZ_TABLE[ord(char) % 56] for char in text)
```

确定性编码，方向 1→1；反向解码存在哈希碰撞，依赖上下文消歧。

### Tokenizer 投影

Qwen3 tokenizer 编码中文 → 按 token 边界切分密文 → 只保留 char_len>=2 的投影 token

例如："管理员你好" → ["管理", "员", "你好"] → 密文 "sbtrvq" → ["sb", "t", "rv"]

---

## 模型与数据规模建议

| 模型 | 最少数据 | 理想数据 | 推荐步数 | 设备 |
|------|---------|---------|---------|------|
| Qwen3-0.6B | 5k-10k | 20k-50k | 500-1000 | MPS（2h） |
| Qwen3-1.7B | 10k-20k | 50k-100k | 1000-2000 | 云端 GPU |
| Qwen3-4B | 20k-50k | 100k-500k | 2000-3000 | 云端 GPU |

当前储备：33 万清洗后句子。

---

## 已知问题

1. **MPS pin_memory** — dataloader 警告，已通过 `dataloader_pin_memory=False` 抑制
2. **transformers 版本** — 新版弃用 `overwrite_output_dir`、`warmup_ratio`，已适配
3. **投影效果** — 单字/字节级 token 被过滤，可能损失部分长尾词

---

## 对 AI 助手的建议

- 修改脚本时请**保持中文注释**，代码风格简洁
- 新增词表统一放至 `vendors/sarkaz_tools/LLMTools/MCPServer/`，UTF-8 无 BOM
- 改动 MCP 服务器时注意 docstring 会作为工具描述暴露给 LLM
- 训练相关修改优先更新 `training/` 目录，而非 `scripts/`
