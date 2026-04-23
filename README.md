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

本repo想要达到的目标是：
- [ ] 收集和合成更多的《明日方舟：终末地》中文语料
- [ ] 将这些语料转换为萨卡兹语
- [ ] 微调一个 Qwen 小模型的 tokenizer 和模型本身，使其能够直接处理萨卡兹语，以便直接翻译为中文

为什么考虑微调小模型：从 Unicode 直接映射到 26 个字母会丢失几乎所有语义和语法信息。因此，需要首先调整 tokenizer，将连续的一大段文本先拆分为 token。举一个转换的例子（逐字对应，没有任何空格）：
```
你好，管理员，是来提交活动菜品的吗？
ytqqrvqbsbtjyrernx
```
在这个例子中，原始文本被转换为萨卡兹语后，完全失去了可读性和结构。通过微调 tokenizer，我们可以将连续的字符序列拆分成更小的 token，这些 token 可以捕捉到一些语义信息，从而使模型能够更好地理解和处理这些输入。除此之外，还需要利用 LLM 的世界知识来帮助翻译和理解萨卡兹语，因为单纯的字符映射无法提供足够的上下文信息。

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

### 2) 生成平行语料（最小验证）

```bash
uv run skz-generate-data --num-samples 2000 --valid-ratio 0.05
```

输出文件：
- `corpus/skz_parallel/base/train.jsonl`
- `corpus/skz_parallel/base/valid.jsonl`
- `corpus/skz_parallel/base/tokenizer_mix.txt`

### 3) 运行超小模型 smoke test

```bash
HF_ENDPOINT=https://hf-mirror.com uv run skz-train-base \
	--model-name Qwen/Qwen3-0.6B \
	--max-train-samples 2000 \
	--max-valid-samples 128 \
	--max-steps 40
```

默认输出目录：`models/base_model/qwen3_0_6b_verify/`

### 4) 构建 Aho-Corasick 词典并做一次解码

```bash
HF_ENDPOINT=https://hf-mirror.com uv run skz-build-trie
HF_ENDPOINT=https://hf-mirror.com uv run skz-decode --text ytqqrvqbsbtjyrernx
```

### 5) 一键执行以上流程

```bash
bash scripts/local_tiny_verify.sh
```

或使用 Qwen3 专用入口：

```bash
bash scripts/local_qwen3_verify.sh
```

如果要先做极速连通性验证（仅 smoke test，不代表效果），可替换为：

```bash
HF_ENDPOINT=https://hf-mirror.com uv run skz-train-base --model-name sshleifer/tiny-gpt2 --max-steps 20
```