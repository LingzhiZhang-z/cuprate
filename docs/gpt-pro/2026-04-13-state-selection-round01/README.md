# GPT Pro 审阅包：态选取算法 Round 01

这个目录是一套可直接用于 ChatGPT Web Project 的审阅材料，主题是：

- 态选取算法本身是否正确
- 态选取和 `T11` / `H_eff` / fitting error 的关系

文件分工如下：

- `01-steps.md`：
  你在网页里按什么顺序操作。
- `02-project-instructions.md`：
  粘贴到 ChatGPT Project instructions 的英文指令。
- `03-review-context.md`：
  给 GPT Pro 的英文技术背景。
- `04-upload-manifest.md`：
  先传哪些文件，后补哪些文件。
- `05-prompt-static-review.md`：
  第一轮静态算法审阅 prompt。
- `06-prompt-fit-error-analysis.md`：
  第二轮分析选态与 fit error 关系的 prompt。
- `07-prompt-adversarial-cases.md`：
  第三轮要求它给出对抗性 failure cases 的 prompt。
- `08-selection-fit-summary-template.csv`：
  如果你要它分析结果数据，可以按这个模板整理表格。

语言规则：

- 给你看的说明文件用中文。
- 给 GPT Pro 的项目指令、上下文和 prompts 用英文。
- 但英文 prompt 里已经明确要求 GPT Pro 用简体中文回复你。

推荐使用顺序：

1. 在 ChatGPT Web 新建一个 Project。
2. 按 `04-upload-manifest.md` 上传最小文件集。
3. 把 `02-project-instructions.md` 粘到 Project instructions。
4. 把 `03-review-context.md` 作为文件上传，或直接贴进第一条消息。
5. 先运行 `05-prompt-static-review.md`。
6. 如果你手里有定量结果，再上传按
   `08-selection-fit-summary-template.csv` 整理好的表，然后运行
   `06-prompt-fit-error-analysis.md`。
7. 最后运行 `07-prompt-adversarial-cases.md`。
