---
name: weather
description: 获取当前天气和预报（通过 wttr.in，无需 API Key）。当用户询问天气、气温、降水、风速、湿度等信息时使用。依赖 curl。
---

# Weather

通过 `wttr.in` 获取实时天气，直接用 `bash` 工具调用 curl，无需 API Key。

## 快速上手

```bash
# 一行摘要：城市 + 天气图标 + 气温
curl -s "wttr.in/Beijing?format=3"

# 详细当前天气（无预报）
curl -s "wttr.in/Shanghai?0"

# 3 天预报
curl -s "wttr.in/Chengdu"

# 自定义格式（气温 + 风速 + 湿度）
curl -s "wttr.in/Beijing?format=%l:+%c+%t,+%w+风,+%h+湿度"

# JSON（供程序解析）
curl -s "wttr.in/Beijing?format=j1"
```

## 在 Agent 循环中调用

当用户询问天气时，直接用 `bash` 工具执行 curl：

```
tool: bash
input: { "command": "curl -s 'wttr.in/Beijing?format=3'" }
```

城市名中有空格时用 `+` 替换（如 `New+York`）。

## 格式代码速查

| 代码 | 含义       |
|------|------------|
| `%c` | 天气图标   |
| `%t` | 气温       |
| `%f` | 体感温度   |
| `%w` | 风速风向   |
| `%h` | 湿度       |
| `%p` | 降水量     |
| `%l` | 位置       |

## 注意

- 城市名支持英文、拼音、机场代码（如 `PEK`）
- 避免高频请求（wttr.in 有限速）
- 如果 curl 不存在，`bash` 工具会返回 command not found，提示用户安装
