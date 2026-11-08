# dsh-soundcheck

[简体中文](README.md) | [English](README.en.md)

**DSH 的本地音乐分析插件**。通过独立的 [Music Evidence Engine](https://github.com/ALittleLov/music-evidence-engine-public) 提取可核对的音乐证据，让 Agent 能依据具体测量与时间位置讨论音频，而不只是猜测。

- **本地、CPU 运行**：不要求独立显卡；分析音频不会上传到云端。首次安装依赖、下载模型需要联网。
- **分析内容**：速度、调性候选、响度与能量、频段、结构变化、时间线、人声及乐器候选等；支持文本与 JSON 证据。
- **按需能力**：标准安装包含四轨音源分离（人声、鼓、贝斯、其他）；六轨分离与歌词听写可选。
- **独立更新**：Soundcheck 负责 DSH 连接、安装和界面；Engine 独立发布。检查到新版后由用户决定是否升级，失败时保留旧版。

## 安装

优先在 **DSH 插件商城**搜索 `dsh-soundcheck` 安装，也可以从 [GitHub Releases](https://github.com/ALittleLov/dsh-soundcheck/releases/latest) 获取 ZIP。

打开 DSH **设置 → 音频分析**，按界面提示安装 Engine。首次安装会准备 Python 3.12、隔离环境、依赖与模型；随后在本机分析音频。CPU 性能、文件长度及分轨/听写任务会影响处理时间。

## 工具

- `audio_analyze`：整首或指定范围的音乐证据
- `audio_section`：针对时间片段的分析
- `audio_bands`：频带、能量与起音相关测量
- `audio_provision`：Engine 与模型的安装、修复和更新

**定位**：这是证据工具，不是自动作曲/混音参数推荐器；测量数据、候选识别和音乐意义应加以区分。

## 项目与许可

[Music Evidence Engine（独立引擎）](https://github.com/ALittleLov/music-evidence-engine-public) · [最新 Soundcheck 版本](https://github.com/ALittleLov/dsh-soundcheck/releases/latest)

主要在 Windows 上验证。代码采用 MIT 许可，第三方依赖与模型遵循各自许可。
