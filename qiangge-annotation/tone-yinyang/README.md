# 《韵学骊珠》四声阴阳标注

这是 `qiangge-annotation` 下独立的四声阴阳子项目。它读取戏曲标注系统导出的项目 JSON，只处理 `project.characterAnnotations` 中的内建逐字块，并输出可重新导入标注系统的四声草稿。

四声结果以后可以作为腔格自动标注的语言学特征之一，但本项目不负责定义或实现完整腔格算法。

## 数据约定

每个逐字块的 `tone` 写为：

```json
{
  "toneClass": "yin_ping | yang_ping | yin_shang | yang_shang | yin_qu | yang_qu | yin_ru | yang_ru",
  "yxlzShangSubtype": "yin_shang | yang_shang | yinyang_tongyong"
}
```

`yxlzShangSubtype` 只用于上声；非上声不得携带该字段。原项目的时间、轨道、工尺、板眼和其他标注不会被修改。

## 环境

项目使用 Python 3.12 和 `uv`：

```bash
cd qiangge-annotation/tone-yinyang
uv sync
cp .env.example .env
```

然后在 `.env` 中填写自己的 `DEEPSEEK_API_KEY`。仓库不会提供或保存任何成员的真实 API Key。

代码只使用 Python 标准库；`uv` 主要负责隔离环境、安装当前子项目及提供稳定 CLI。

## 先做只读检查

```bash
uv run yxlz-tone-annotate /path/to/project.annotation.json --dry-run
```

该命令会检查项目结构、统计待标字块并显示批次数，不调用 API。

## 正式运行

```bash
uv run yxlz-tone-annotate /path/to/project.annotation.json
```

默认在输入文件旁生成：

- `*.tone-ai-draft.json`：写入 `tone` 的可导入项目草稿。
- `*.tone-ai-review.json`：置信度、依据、待审核项和同字异标报告。
- `.*.tone-ai-work/`：逐批断点缓存，仅供本次四声任务续跑。

中断后执行同一命令会复用已经完成的批次。若输入、模型、提示词或批次参数改变，使用 `--restart` 明确重新开始。

## 常用参数

```text
--reference PATH               提供机器可读《韵学骊珠》字表
--model deepseek-v4-pro        更换模型
--batch-lines 3                每批最多句数
--batch-characters 24          每批最多字块数
--temperature 0.1              采样温度
--confidence-threshold 0.80    低于该值进入复核
--thinking disabled            是否启用模型思维模式
--max-tokens 4096              单批最大输出 token
--overwrite-existing           重新标注已有 tone
--omit-low-confidence          低置信结果保持 tone: null
--restart                      清空本次四声任务缓存后重跑
--dry-run                      只检查，不调用 API
```

## 《韵学骊珠》资料

`references/yxlz_reference.example.json` 展示机器可读字表结构。正式字表应记录可靠的字形、调类、韵部、页码和必要原文。

没有提供直接字表条目时，程序会把模型结论强制标记为 `context_inference`，置信度上限设为 `0.79`，并要求人工复核。模型不能自行声称已经查得原书依据。

## 示例数据

`data/examples/` 包含《寻梦》的一次 AI 草稿及脱敏复核报告：

- 共处理 427 个逐字块。
- 所有结果均为没有机器可读字表支持的语境推断，不是人工核定数据。
- 复核报告已移除 API 请求 ID、真实密钥、断点响应和本机绝对路径。

## 验证

```bash
uv run python -m unittest discover -s tests -v
uv run yxlz-tone-annotate data/examples/xunmeng-tone-draft.annotation.json \
  --overwrite-existing \
  --dry-run
```
