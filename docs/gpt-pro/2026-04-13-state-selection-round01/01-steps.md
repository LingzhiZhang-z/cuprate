# 步骤

1. 在 ChatGPT Web 里新建一个 Project。
2. 名字可以起得稳定一点，例如：
   `cuprate-state-selection-round01`
3. 把 `02-project-instructions.md` 的内容粘贴到 Project instructions。
4. 按 `04-upload-manifest.md` 里的“最小上传集”先上传文件。
5. 再上传 `03-review-context.md`，或者把它的内容直接贴到第一条消息里。
6. 用 `05-prompt-static-review.md` 开始第一轮静态审阅。
7. 如果 GPT Pro 明确说上下文不够，再按 `04-upload-manifest.md` 里的
   “扩展上传集”补文件。
8. 第一轮静态审阅结束后，如果你手里有定量结果，就按
   `08-selection-fit-summary-template.csv` 的格式整理一张紧凑结果表。
9. 上传这张结果表，然后运行 `06-prompt-fit-error-analysis.md`。
10. 最后运行 `07-prompt-adversarial-cases.md`，逼它给出真正具体的
    failure mode 和最小验证实验。
11. 三轮都尽量放在同一个 Project 里，这样上传文件和项目指令都能复用。

实用规则：

- 除非 GPT Pro 明确卡住，否则不要一开始就上传整个仓库。
- 先用最小文件集，只有在它提出明确需要时再补更多文件。
