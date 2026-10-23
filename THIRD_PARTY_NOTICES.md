# 第三方组件与许可

本项目以 MIT 许可发布（见 `LICENSE`）。以下是随插件或安装时一并使用的组件，
以及它们各自的许可与来源。

---

## 运行时依赖

安装时由 `audio_provision` 通过 pip 装进用户自己的环境，**不随仓库或 zip 分发**。

| 组件 | 开发时版本 | 许可 | 主页 |
|---|---|---|---|
| librosa | 1.0.0 | ISC | https://librosa.org |
| numpy | 2.5.4 | BSD-3-Clause（另含 0BSD / MIT / Zlib / CC0 片段） | https://numpy.org |
| scipy | 1.18.1 | BSD-3-Clause | https://scipy.org |
| onnxruntime | 1.31.0 | MIT | https://onnxruntime.ai |
| soundfile | 0.14.0 | BSD-3-Clause | https://python-soundfile.readthedocs.io |

### scipy 二进制中捆绑的组件

Windows 版 scipy 的 wheel 里带有下列库：

- OpenBLAS —— BSD-3-Clause
- LAPACK —— BSD-3-Clause-Open-MPI
- GCC runtime library —— GPL-3.0-or-later WITH GCC-exception-3.1

这些只随用户本地安装的 scipy 出现，本项目不分发它们。

---

## 模型

### YAMNet

- 原始模型：Google **YAMNet**，**Apache License 2.0**
- 原始地址：https://tfhub.dev/google/yamnet/1
- 本项目使用的 ONNX 转换：https://huggingface.co/niobures/YAMNet
- 该转换仓库未在根目录声明许可证，因此这里按原模型的 **Apache 2.0** 署名
- 模型文件（`yamnet.onnx`、`yamnet_class_map.csv`）**不随仓库提交**；
  分发包中包含一份，或由 `audio_provision` 在首次安装时下载
- 用途：只做声源/乐器方向的弱先验，不参与任何测量结论

---

## 只用于文档与视频（不随插件分发）

- **FFmpeg**（视频渲染，GPL 构建）—— 仅作为本机工具使用，未分发其二进制
- **Microsoft YaHei**（图表与视频渲染时的系统字体）—— 专有字体，未分发字体文件
- **示例音频** —— 未随仓库分发；视频中的背景音乐由作者自行确认使用权

---

## 许可全文

- MIT —— https://opensource.org/license/mit
- ISC —— https://opensource.org/license/isc-license-txt
- BSD-3-Clause —— https://opensource.org/license/bsd-3-clause
- Apache-2.0 —— https://www.apache.org/licenses/LICENSE-2.0
- PSF (matplotlib) —— https://matplotlib.org/stable/users/project/license.html
- MIT-CMU (Pillow) —— https://github.com/python-pillow/Pillow/blob/main/LICENSE
- GCC Runtime Library Exception —— https://www.gnu.org/licenses/gcc-exception-3.1.html
