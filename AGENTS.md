# AGENTS.md — endfield-sarkaz

> 本文件面向 AI 编码助手。项目内的注释与文档主要使用中文，因此本文件亦以中文撰写。

---

## 项目概述

`endfield-sarkaz` 是一个围绕《明日方舟：终末地》中“萨卡兹语”展开的研究型仓库。萨卡兹语本质上是一种将中文字符通过 Unicode 取模后映射到 26 个英文字母上的特殊字体/编码。项目目标包括：

1. 收集并合成更多《明日方舟：终末地》相关中文语料；
2. 将语料转换为萨卡兹语；
3. （远期）微调 Qwen 小模型的 tokenizer 与模型本身，使其能够直接处理萨卡兹语并翻译为中文。

当前阶段已实现基础的编码/解码脚本、词汇搜索工具，以及一个基于 MCP（Model Context Protocol）协议的 LLM 工具服务器，供 AI 自动调用以翻译萨卡兹语。

---

## 仓库结构

```
.
├── README.md                     # 项目根说明文档
├── corpus/                       # 语料目录
│   ├── arknights_stories.json    # 明日方舟剧情语料（结构化 JSON）
│   ├── arknights_stories.txt     # 明日方舟剧情语料（纯文本）
│   ├── arknights_sentences.txt   # 明日方舟剧情语料（每行一句）
│   ├── raw/                      # 原始语料（预留）
│   └── skz_parallel/             # 萨卡兹语平行语料
├── output/                       # 工具输出目录
│   ├── xlsx/                     # ASTR-Script 生成的 Excel 文件
│   └── corpus/                   # 示例语料备份
├── scripts/                      # 脚本工具
│   ├── extract_corpus.py         # 从 ArknightsGameData 提取剧情语料
│   ├── generate_corpus.py        # 生成萨卡兹语平行语料
│   └── evaluate.py               # 模型评估脚本
├── training/                     # 模型训练相关
│   ├── base_train.py             # 基础训练脚本
│   ├── cloud_train.py            # 云端训练脚本
│   ├── data_generator.py         # 数据生成器
│   ├── merge_tokenizer.py        # Tokenizer 合并
│   └── tokenizer_train.py        # Tokenizer 训练
├── inference/                    # 推理相关
│   ├── sarkaz_decoder.py         # 萨卡兹语解码器
│   └── trie_builder.py           # Trie 树构建
├── models/                       # 模型目录（预留）
├── vendors/
│   ├── ASTR-Script/              # 明日方舟剧情文本提取工具
│   │   ├── xlsxconvert.py        # Excel 格式导出
│   │   ├── jsonconvert.py        # JSON 格式导出
│   │   └── func.py               # 核心解析函数
│   ├── ArknightsGameData/        # 明日方舟游戏数据（子模块）
│   └── sarkaz_tools/             # 第三方/子模块工具集（内含独立 .git 仓库）
│       ├── README.md
│       ├── HumanTools/           # 人工使用的 CLI 工具
│       │   ├── decode.py         # 研究字符映射规律时的实验脚本
│       │   ├── sarkazEncoder.py  # 交互式萨卡兹语编码器（单行脚本）
│       │   ├── wordSearcher.py   # 根据萨卡兹字母串反查中文词汇（精确匹配）
│       │   ├── wordlist.txt      # 汉语常用词表（~56k 行）
│       │   ├── 对照表.txt         # 按 a-z 分组的手工整字表
│       │   └── wiki2019zh.json   # 仅含字频统计（char→count），非可用的文本语料
│       └── LLMTools/             # 面向大语言模型的工具
│           ├── prompt.txt        # LLM 角色设定提示词（Sarkaz Language Decoder）
│           └── MCPServer/
│               ├── Sarkaz_tools.py   # MCP Server 主入口（fastmcp）
│               ├── endfield_words.txt # 终末地专有名词表（~27 行）
│               ├── favorite_words.txt # 用户/AI 收藏的词汇（读写）
│               ├── single_char.txt    # 常用单字表（~3k 行）
│               └── wordlist.txt       # 汉语常用词表（~56k 行）
```

根目录已包含 `pyproject.toml`（uv 管理），使用 `uv run` 执行脚本。

---

## 技术栈与依赖

- **语言**：Python 3
- **核心外部依赖**：
  - `fastmcp`（仅 `LLMTools/MCPServer/Sarkaz_tools.py` 使用）
  - `openpyxl`（仅 `ASTR-Script/xlsxconvert.py` 使用）
- **无其他运行时依赖**。所有脚本均为可直接运行的 `.py` 文件。

---

## 构建、运行与使用方式

### 1. 人工工具（HumanTools）

所有脚本均可直接执行，无需构建步骤：

```bash
# 交互式编码器
python vendors/sarkaz_tools/HumanTools/sarkazEncoder.py

# 解码/规律研究脚本
python vendors/sarkaz_tools/HumanTools/decode.py

# 词汇搜索（精确匹配）
python vendors/sarkaz_tools/HumanTools/wordSearcher.py
```

> ⚠️ `wordSearcher.py` 内部硬编码了 Windows 绝对路径 `C:/Users/Administrator/...`，在其他系统上运行前需修改路径。

### 2. MCP 服务器（LLMTools）

```bash
# 需先安装 fastmcp
pip install fastmcp

# 启动 MCP 服务器（stdio 传输）
python vendors/sarkaz_tools/LLMTools/MCPServer/Sarkaz_tools.py
```

### 3. 剧情语料提取（ASTR-Script）

```bash
# 列出所有活动
uv run -- python vendors/ASTR-Script/xlsxconvert.py vendors/ArknightsGameData -L zh_CN -E

# 导出所有主线剧情
uv run -- python vendors/ASTR-Script/xlsxconvert.py vendors/ArknightsGameData -L zh_CN -m

# 导出所有干员密录
uv run -- python vendors/ASTR-Script/xlsxconvert.py vendors/ArknightsGameData -L zh_CN -r

# 导出特定活动（索引 0）
uv run -- python vendors/ASTR-Script/xlsxconvert.py vendors/ArknightsGameData -L zh_CN -e 0
```

### 4. 语料库生成

```bash
# 从 ArknightsGameData 提取完整语料库（JSON + 纯文本 + 每行一句）
uv run -- python scripts/extract_corpus.py
```

生成文件：
- `corpus/arknights_stories.json` — 结构化 JSON（事件 → 故事层级）
- `corpus/arknights_stories.txt` — 人类可读的纯文本
- `corpus/arknights_sentences.txt` — 每行一句，适合 NLP 处理

服务器暴露以下工具（tool）：

| 工具名 | 功能 |
|--------|------|
| `search_exact` | 精确搜索萨卡兹字符串对应的中文词汇（同时检索常用词与收藏词） |
| `search_general_in_favorite` | 在收藏词列表中模糊匹配 |
| `search_single_chinese_character` | 按单字搜索 |
| `convert` | 将字符串转换为萨卡兹语 |
| `bulkConvert` | 批量转换，返回字典 |
| `writeWordIntoFavorite` | 将词汇写入收藏列表 |
| `bulkWriteWordsIntoFavorite` | 批量写入收藏列表 |
| `readFavoriteWords` | 读取收藏列表并附带萨卡兹转换结果 |

---

## 核心算法

萨卡兹语编码规则极其简单，所有 Python 脚本均采用同一实现：

```python
"gkamztlbdqiyfucxbhsjoprnweygtjmevchdxsanqolkrvwiypjzquhe"[ord(char) % 56]
```

即：取字符的 Unicode 码点，对 56 取模，以上述 56 字母字符串为映射表得到对应字母。该规则在编码方向是确定性的（一个中文字符始终映射为一个萨卡兹字母），但反向解码时存在大量哈希碰撞（多对一），需依赖上下文消歧。

---

## 代码风格与约定

- **注释语言**：中文为主。
- **代码风格**：较为随意，未使用 Black、Ruff 等格式化工具约束。
- **类型注解**：仅在 `Sarkaz_tools.py` 的 MCP 工具函数签名中使用了 Python 类型提示（因 `fastmcp` 需要）。其余脚本基本没有类型注解。
- **文件编码**：所有文本文件均使用 **UTF-8**。
- **路径处理**：`Sarkaz_tools.py` 使用了 `os.chdir(os.path.dirname(os.path.abspath(__file__)))` 将工作目录切换到脚本所在目录，以便使用相对路径读取词表；而 `wordSearcher.py` 使用了硬编码的绝对路径，需注意跨平台兼容性。
- **字符串拼接**：部分脚本使用 `+=` 拼接字符串（如 `convertCharsToSKZ`），在性能敏感场景下可考虑改为 `list` + `join`。

---

## 测试策略

- **当前没有任何自动化测试**（无 `pytest`、`unittest`、CI 配置等）。
- 验证方式以手动运行脚本并观察输出为主。
- 若后续添加测试，建议：
  1. 为 `convertCharsToSKZ` 编写单元测试，覆盖常见汉字、标点（如中文逗号映射为 `q`）及边界字符；
  2. 为 MCP 工具函数编写集成测试，使用临时文件替代 `favorite_words.txt`；
  3. 对词表文件进行完整性校验（如检查空行、重复项）。

---

## 安全与注意事项

- **文件读写**：`Sarkaz_tools.py` 在运行时会读写同目录下的 `favorite_words.txt`，无额外鉴权或路径校验。若部署在不受信任的环境中，需防止路径遍历或文件注入。
- **无秘密管理**：项目中不存在 API Key、数据库凭证等敏感信息。
- **输入验证**：MCP 工具对输入仅做了最小限度的检查（如 `search_single_chinese_character` 检查 `len(char) != 1`），其他函数基本直接透传输入进行转换或文件写入。
- **依赖风险**：`fastmcp` 为唯一外部依赖，更新时需注意其 API 兼容性。

---

## 后续计划（来自 README）

- [ ] 收集和合成更多《明日方舟：终末地》中文语料
- [ ] 将这些语料转换为萨卡兹语
- [ ] 微调 Qwen 小模型的 tokenizer 和模型本身，使其能够直接处理萨卡兹语并翻译为中文

---

## 对 AI 助手的建议

- 修改脚本时，请**保持中文注释**，并维持现有代码的简洁风格。
- 若新增词表文件，统一放至 `LLMTools/MCPServer/` 或 `HumanTools/` 下，并使用 UTF-8 无 BOM 格式、每行一个词条。
- 若改动 MCP 服务器，注意 `fastmcp` 要求工具函数的 docstring 会作为工具描述暴露给 LLM，因此 docstring 应清晰、准确。
- 引入新的 Python 依赖时，建议同步创建 `requirements.txt` 或 `pyproject.toml`，以弥补当前缺失的依赖管理。
