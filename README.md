# endfield-sarkaz

vendors/sarkaz_tools: https://github.com/ZhangHB-321688/sarkaz_tools
https://www.bilibili.com/video/BV1PtdyBMEED/ 我们破解了终末地的萨卡兹语！

萨卡兹语实际上是一种特殊的英文字体。在终末地的前瞻视频中，鹰角通过将中文以特殊规则转换为字母，并展示为萨卡兹语，以隐藏具体的信息，如新版本的内容。 [@比力币](https://space.bilibili.com/488486599) 和 [@肖冰嗼嗼Tam](https://space.bilibili.com/754455) 在上述视频中提到了转换规则，实际上就是将字符的 Unicode 编码通过取模和一个映射表转换到 26 个英文字母上。

```python
# 打印任意输入文本映射到萨卡兹语
while True:
    for _ in input("\n输入要编码的文本："):
        print("gkamztlbdqiyfucxbhsjoprnweygtjmevchdxsanqolkrvwiypjzquhe"[ord(_)%56],end="")
```

本 repo 想要达到的目标是：
- [x] 收集和清洗《明日方舟》中文语料
- [x] 将语料转换为萨卡兹语平行数据集
- [x] 本地微调 Qwen3-0.6B smoke test，验证训练流程
- [ ] 微调 Qwen3-4B 模型的 tokenizer 与模型，实现端到端萨卡兹语→中文翻译
- [ ] 收集《明日方舟：终末地》语料，训练领域 LoRA

为什么考虑微调小模型：从 Unicode 直接映射到 26 个字母会丢失几乎所有语义和语法信息。因此，需要首先调整 tokenizer，将连续的一大段文本先拆分为 token。举一个转换的例子（逐字对应，没有任何空格）：
```
你好，管理员，是来提交活动菜品的吗？
ytqqrvqbsbtjyrernx
```
在这个例子中，原始文本被转换为萨卡兹语后，完全失去了可读性和结构。通过微调 tokenizer，我们可以将连续的字符序列拆分成更小的 token，这些 token 可以捕捉到一些语义信息，从而使模型能够更好地理解和处理这些输入。除此之外，还需要利用 LLM 的世界知识来帮助翻译和理解萨卡兹语，因为单纯的字符映射无法提供足够的上下文信息。

## Pipeline 概览

```
1. 语料清洗
   arknights_sentences.txt → clean_corpus.py → arknights_cleaned.txt
   (去噪声、去重、剥离说话人前缀)

2. 平行语料生成
   arknights_cleaned.txt → data_generator.py (--corpus) → train.jsonl / valid.jsonl / tokenizer_mix.txt
   (每条中文句子 → convert_chars_to_skz() 生成对应密文)

3. Tokenizer 构建
   a) Qwen3 tokenizer 编码中文 → 多字 token 投影到密文 (如 "管理"→"sb")
      只保留 char_len>=2 的 token（跳过字节级碎片）
   b) SPM 在 tokenizer_mix.txt 上训练，补充高频子词模式
   c) 合并：投影 token + SPM token + endfield_words 硬编码 → 注入 Qwen3 词表

4. 模型训练
   Qwen3-0.6B + 扩展词表 → LoRA SFT → 验证流程
   Qwen3-4B + 扩展词表 → LoRA SFT → 端到端翻译模型（目标）
```

## 使用 uv 启动本地实验

### 1) 安装与同步依赖

```bash
# 如果本机还没有 Python 3.12，可以先安装
uv python install 3.12

# 创建并同步环境
uv venv --python 3.12
uv sync
```

建议通过 hf-mirror 拉取模型：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 2) 语料清洗与平行语料生成

```bash
# 清洗明日方舟剧情语料
uv run python scripts/clean_corpus.py

# 用真实句子生成平行语料（默认 5000 条，可加大到 50000）
uv run skz-generate-data --corpus corpus/raw/ak/arknights_cleaned.txt

# 或从 wordlist 随机合成
uv run skz-generate-data --num-samples 5000
```

输出文件：
- `corpus/skz_parallel/base/train.jsonl`
- `corpus/skz_parallel/base/valid.jsonl`
- `corpus/skz_parallel/base/tokenizer_mix.txt`

### 3) 训练 Tokenizer 并合并词表

```bash
# 训练 SentencePiece
uv run skz-train-tokenizer

# 合并进 Qwen3 词表（含中文 token 投影 + endfield_words）
uv run skz-merge-tokenizer --init-embeddings
```

### 4) 本地验证（0.6B smoke test）

```bash
HF_ENDPOINT=https://hf-mirror.com uv run skz-train-base \
    --model-name Qwen/Qwen3-0.6B \
    --tokenizer-path models/tokenizer/merged \
    --max-train-samples 4900 \
    --max-steps 80
```

### 5) 云端正式训练（4B，目标）

需要在云端 GPU 上运行，数据规模需要扩大到 10 万-50 万条。

### 6) 构建 Aho-Corasick 词典并做一次解码

```bash
uv run skz-build-trie
uv run skz-decode --text ytqqrvqbsbtjyrernx
```

### 7) 一键执行验证流程

```bash
bash scripts/local_qwen3_verify.sh
```

## 致谢

- https://github.com/ZhangHB-321688/sarkaz_tools 给出了中文字符到萨卡兹密文的映射规律和 MCP 解密工具
- https://github.com/Kengxxiao/ArknightsGameData 提供了《明日方舟》本体原始解包数据
- https://github.com/050644zf/ASTR-Script/ 提供了从解包数据中提取剧情文本的工具，方便构建优质平行语料
