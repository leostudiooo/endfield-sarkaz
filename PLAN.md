# 萨卡兹语端到端翻译系统实施计划

## Context

现有 `endfield-sarkaz` 项目通过 MCP 服务器暴露多个工具（`search_exact`、`search_single_chinese_character`、`convert` 等）供 LLM 调用以翻译萨卡兹语。该方案需要多轮工具调用，每次调用都消耗大量 token，成本高且延迟大。

本计划目标：训练一个定制 tokenizer + 微调 Qwen3-4B 模型，实现**端到端萨卡兹语翻译**，将多次 MCP 工具调用压缩为单次模型推理。同时用 Aho-Corasick 本地预处理注入专有名词线索，在零额外 token 成本下提升消歧能力。

## 当前状态

- [x] 语料清洗 pipeline 完成（333k 条明日方舟句子）
- [x] 平行语料生成完成（支持 corpus/wordlist 两种模式）
- [x] Tokenizer 训练 + 合并完成（含 SPM + 投影 + 硬编码，词表 168k）
- [x] 0.6B LoRA smoke test 通过（4900 样本，80 步，loss 5.48→4.13）
- [ ] 4B 云端训练（待执行）

## 推荐方案

### 第一阶段：本地验证（Tokenizer + 0.6B smoke test）✅ 已完成

#### 1.1 项目结构调整

```
endfield-sarkaz/
├── corpus/                          # 语料目录
│   ├── raw/                         # 原始中文语料
│   │   └── ak/                      # 明日方舟语料
│   │       ├── arknights_stories.json
│   │       ├── arknights_sentences.txt
│   │       └── arknights_cleaned.txt   # 清洗后 333k 行
│   └── skz_parallel/               # 生成的平行语料
│       └── base/                    # 通用平行语料
│           ├── train.jsonl
│           ├── valid.jsonl
│           └── tokenizer_mix.txt
├── training/                        # 训练脚本
│   ├── data_generator.py            # 支持从 corpus 生成真实句子
│   ├── tokenizer_train.py           # SPM 训练
│   ├── merge_tokenizer.py           # SPM + 投影 + 硬编码合并
│   └── base_train.py                # LoRA 训练
├── models/                          # 训练产出
│   ├── tokenizer/                   # custom_sarkaz.model/.vocab + merged/
│   └── base_model/                  # resized/ + qwen3_0_6b_verify/
└── scripts/                         # 工具脚本
    └── clean_corpus.py              # 语料清洗
```

#### 1.2 Tokenizer 改造 ✅ 已完成

**核心思路**：Qwen3 原生使用 tiktoken byte-level BPE，词表 ~151k。用 SentencePiece 在混合语料上训练新 token，再通过 `AutoTokenizer.add_tokens()` 注入 Qwen3。在**中文原文 + 萨卡兹密文混合语料**上训练 SPM，提取新 token 合并后扩展词表。

**实现细节**：

1. **语料准备**（`training/data_generator.py`）：
   - 支持从清洗后的 corpus 读取真实中文句子（`--corpus` 参数）
   - 每条中文生成对应的萨卡兹密文（`convert_chars_to_skz`）
   - 混合格式：中文↔密文交替，用于 SPM 训练

2. **SPM 训练**（`training/tokenizer_train.py`）：
   - 工具：`sentencepiece.SentencePieceTrainer.train()`
   - 参数：`model_type='unigram'`, `vocab_size=16384`
   - 输出：`custom_sarkaz.model`、`custom_sarkaz.vocab`

3. **词表合并**（`training/merge_tokenizer.py`）：
   - 加载 Qwen3 tokenizer + 自定义 SPM
   - **新增**：中文 token 投影 — Qwen3 编码中文 → 按边界切分密文 → 只保留 char_len>=2
   - 硬编码注入：`endfield_words.txt`（中文+密文对）
   - Embedding 初始化：mean subtoken initialization
   - 结果：词表 168k，新增 16,487 token（其中 2,108 投影 token）

4. **专有名词注入**：
   - `endfield_words.txt`（27 行终末地专有名词）作为 `AddedToken` 强制注入
   - 确保「管理员→sbt」等关键映射不被切分

#### 1.3 0.6B smoke test ✅ 已完成

**数据**：从 `arknights_cleaned.txt` 随机采样 4900 条

**训练配置**：
- 基座：`Qwen/Qwen3-0.6B-Instruct`
- 词表：扩展后的 `models/tokenizer/merged`
- 训练：LoRA rank=16, alpha=32, lr=2e-4
- 数据：max_train_samples=4900, max_steps=80
- 设备：MPS（约 14 分钟）

**结果**：
- eval_loss：5.485 → 4.133（-1.35）
- LoRA 模块：q/k/v/o_proj + gate/up/down_proj
- 结论：训练流程跑通，loss 下降，方案可行

---

### 第二阶段：云端正式训练（Qwen3-4B）⏳ 待执行

#### 2.1 扩大语料规模

当前储备 33 万清洗后句子，建议：

- **最小可行**：5 万条，500-1000 步
- **推荐规模**：10 万-20 万条，1000-2000 步
- **理想规模**：50 万条，2000-3000 步

```bash
# 生成更大规模语料
uv run skz-generate-data \
    --corpus corpus/raw/ak/arknights_cleaned.txt \
    --num-samples 100000 \
    --valid-ratio 0.02
```

#### 2.2 云端训练配置

**基座模型**：`Qwen/Qwen3-4B-Instruct`

**Tokenizer**：复用第一阶段训练好的 `models/tokenizer/merged`

**训练方式**：
- LoRA（rank=128, alpha=256）+ 新 token embedding 训练
- lr=2e-4, batch_size=4-8（根据 GPU 调整）, gradient_accumulation=4-8
- warmup_steps=100-200
- max_steps=1000-2000

**输出**：`models/base_model/qwen3_4b/`

**预期效果**：
- CER < 15%
- 专有名词命中率 > 95%

---

### 第三阶段：领域 LoRA（延后执行）

需要终末地剧情文本到位后执行：

1. **数据收集**：终末地剧情、道具描述、世界观设定
2. **数据合成**：原文 → 萨卡兹密文，构建 5k-2 万条领域平行语料
3. **训练**：在第二阶段模型基础上，训练可热插拔的 LoRA 模块
4. **输出**：`models/lora/endfield_lora/`

---

### 第四阶段：推理 Pipeline（替代 MCP 多工具调用）

#### 4.1 Aho-Corasick 预处理

- 输入：`endfield_words.txt` + `favorite_words.txt` + 扩展词库
- 构建 `pyahocorasick.Automaton`，key 为萨卡兹密文，value 为中文词列表
- 保存到磁盘，启动时加载

#### 4.2 端到端解码器

**Pipeline**：
```python
def decode(skz_text: str) -> str:
    # 1. AC 扫描密文，找出候选词
    hints = ac_automaton.scan(skz_text)
    # 2. 构建 prompt 注入线索
    system_prompt = build_prompt_with_hints(hints)
    # 3. 单次调用微调模型
    response = model.generate(system_prompt, skz_text)
    return response
```

**Prompt 设计**：
```
系统：你是萨卡兹语翻译专家。请将以下萨卡兹密文翻译为中文。
【翻译线索】密文中检测到以下可能的词汇：
- sbt -> [管理员]
- jy -> [某角色]

用户：ytqqrvqbsbtjyrernx
```

#### 4.3 MCP 整合

精简为**单个工具**：
- 移除 `search_exact`、`search_single_chinese_character` 等多工具
- 新增 `translate_sarkaz(skz_text: str) -> str`
- LLM 单次调用获得翻译结果

---

## 关键文件与复用点

| 文件 | 作用 |
|------|------|
| `training/common.py` | `convert_chars_to_skz` 核心算法 |
| `training/data_generator.py` | 支持 `--corpus` 从真实句子生成平行语料 |
| `training/merge_tokenizer.py` | SPM + 投影 + 硬编码三路合并 |
| `scripts/clean_corpus.py` | 语料清洗（去噪声、去重、剥离说话人） |
| `vendors/.../MCPServer/endfield_words.txt` | 硬编码词表 + AC 自动机词源 |

---

## 验证方案

1. **Tokenizer 验证**：
   - 检查「管理员」是否被切分为 1-2 个 token
   - 检查「sbt」是否被切分为单个 token

2. **模型验证**：
   - CER（字错误率）：目标 < 15%
   - 专有名词命中率：目标 > 95%
   - 人工抽查 50 条

3. **Pipeline 验证**：
   - 对比旧 MCP 方案（多工具）vs 新方案（单次调用）
   - Token 消耗对比
   - 翻译质量盲测

---

## 依赖

```
transformers>=4.45.0
torch>=2.4.0
datasets>=3.0.0
sentencepiece>=0.2.0
peft>=0.13.0
accelerate>=0.34.2
pyahocorasick>=2.1.0
```

---

## 风险与备选

- **Tokenizer 改造风险**：若扩展词表后不收敛，回退到不改造 tokenizer 的 baseline
- **语料不足风险**：当前 33 万句子储备足够 4B 训练，但终末地领域语料仍需收集
- **模型选择**：如 Qwen3-4B 效果不佳，备选 `Qwen2.5-7B-Instruct`
