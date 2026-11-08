# dsh-soundcheck

DSH 的音频分析 Bridge：把 [Music Evidence Engine](https://github.com/ALittleLov/music-evidence-engine-public) 的本地分析能力接进 DSH。

## 是什么

- 在 DSH 里分析音乐/音频，返回节拍、调性、能量、结构切分、分轨、歌词听写等**可核对的证据**。
- 不是 FL 参数推荐器，也不把测量当成音乐结论；原始 JSON 字段原样透传。

## 安装

1. 在 DSH 插件商城搜索 `dsh-soundcheck` 安装，或从 Release 下载 ZIP。
2. 打开 设置 → 音频分析，点「安装引擎」。
3. 首次安装需要网络：下载 Engine wheel、依赖与模型；分析全部在本机完成，音频不出本机。

## 与 Engine 的关系

- Engine（`music-evidence-engine`）独立发布、独立升级；Soundcheck 只负责 Bridge、安装调用、工具与 UI。
- 模型清单、SHA256、下载与缓存由 Engine 管理；Soundcheck 不携带 Engine 二进制。
- 安装配置：基础 Engine + 4 轨分轨（默认）；6 轨分轨与歌词听写是可选的扩展能力。
- Engine 有更新时，设置页会提示当前版本与目标版本，确认后才升级；升级失败保留原版本。

## 功能

- `audio_analyze`：整首/片段分析（节拍、调性、结构、分轨、听写）。
- `audio_section`：指定时间段的证据分析。
- `audio_bands`：分频段能量/起音/规律性测量。
- `audio_provision`：安装、修复、更新 Engine 与模型。

## 已知限制

- 首次安装依赖网络；目前主要在 Windows 验证。
- 分轨和听写首次运行较慢（CPU），结果会缓存。
