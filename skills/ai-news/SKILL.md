---
name: ai-news
description: 搜索今日 AI 新闻，聚合自中文科技媒体（IT之家、TechWeb、品玩、Solidot），无需 API Key。当用户询问 AI 新闻、人工智能资讯、大模型动态、OpenAI/DeepSeek 等话题时使用。依赖 curl 和 python3。
---

# AI News

搜索今日 AI 新闻，聚合自中文科技媒体（IT之家、TechWeb、品玩 PingWest、Solidot），无需 API Key。

## 核心流程

并行抓取 3-4 个源 → Python 正则提取 + 关键词过滤 → 去重 → 按主题分类输出。

## 多源并行抓取（推荐）

一次执行以下多条 curl，每条用 `python3 -c` 解析：

### 源 1：IT之家 (ithome.com)

```bash
curl -s -m 10 "https://www.ithome.com/" -H "User-Agent: Mozilla/5.0" | python3 -c "
import sys, re
html = sys.stdin.read()
texts = re.findall(r'>([^<]{20,300})<', html)
kw = ['AI', '人工智能', 'OpenAI', 'DeepSeek', '大模型', 'GPT', 'ChatGPT', '机器人', '智驾', 'Siri', '模型', 'Copilot', 'Gemini']
for t in texts:
    t = t.strip()
    if any(k in t for k in kw) and len(t) > 15:
        print(t)
        print('---')
"
```

### 源 2：TechWeb (techweb.com.cn)

```bash
curl -s -m 10 "https://www.techweb.com.cn/" -H "User-Agent: Mozilla/5.0" | python3 -c "
import sys, re
html = sys.stdin.read()
texts = re.findall(r'>([^<]{20,300})<', html)
kw = ['AI', '人工智能', 'OpenAI', 'DeepSeek', '大模型', 'GPT', 'ChatGPT', '机器人', '智驾', 'Siri', '模型', 'Copilot']
for t in texts:
    t = t.strip()
    if any(k in t for k in kw) and len(t) > 15:
        print(t)
        print('---')
"
```

### 源 3：品玩 PingWest (pingwest.com)

```bash
curl -s -m 10 "https://www.pingwest.com/" -H "User-Agent: Mozilla/5.0" | python3 -c "
import sys, re
html = sys.stdin.read()
texts = re.findall(r'>([^<]{20,300})<', html)
kw = ['AI', '人工智能', 'OpenAI', 'DeepSeek', '大模型', 'GPT', 'ChatGPT', '机器人', '模型', '具身智能', '世界模型']
for t in texts:
    t = t.strip()
    if any(k in t for k in kw) and len(t) > 15:
        print(t)
        print('---')
"
```

### 源 4：Solidot RSS (solidot.org)

```bash
curl -s -m 10 "https://www.solidot.org/index.rss" -H "User-Agent: Mozilla/5.0" | python3 -c "
import sys, re
content = sys.stdin.read()
titles = re.findall(r'<title>(.*?)</title>', content)
for t in titles[1:]:
    t = t.strip()
    if any(k in t for k in ['AI', 'OpenAI', 'ChatGPT', '人工智能', '模型', '机器人']):
        print(t)
"
```

## 关键词过滤列表

```
AI, 人工智能, OpenAI, DeepSeek, 大模型, GPT, ChatGPT, 机器人, 智驾,
Siri, 模型, Copilot, Gemini, Claude, LLM, 具身智能, 世界模型, Agent,
智能体, 自动驾驶, AIGC, GPU, 算力, 推理
```

## 输出格式

按主题分类整理，注明来源和采集时间：

```
## 🔥 今日头条（3-5 条）
## 🇨🇳 国内大厂动态
## 🤖 大模型 & 具身智能
## 💡 行业观点 & 趋势
## ⚠️ 争议 & 其他
> 📌 来源：IT之家、TechWeb、品玩、Solidot | 采集时间：{date}
```

## 备选源

当主流源全部超时时，尝试：

```bash
# cnBeta 移动版 AI 频道
curl -s -m 10 "https://m.cnbeta.com/wap/ai/" -H "User-Agent: Mozilla/5.0"

# 机器之心
curl -s -m 10 "https://www.jiqizhixin.com/" -H "User-Agent: Mozilla/5.0"
```

## 注意事项

- 所有 curl 加 `-m 10` 限制超时，避免卡死
- 网络环境可能无法访问境外源（Reddit、Google News 等），专注国内源
- 同一标题可能出现在正文摘要中，去重时保留最长版本
- 正文片段若被截断（>300 字符限制），保留已抓取部分即可
- 如果 4 个源全挂，如实告知用户"当前网络无法抓取 AI 新闻"
