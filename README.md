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