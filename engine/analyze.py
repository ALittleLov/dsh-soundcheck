#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
音频结构分析 v2 —— 把一首歌压成一段「AI 能读」的紧凑文本。

用法:
    python analyze.py <音频文件> [--blocks 16] [--chord-block 2.0]

v2 改动:
    1. DYN 改成绝对 dBFS(原来用归一化,范围恒为 -80~0,是废话)
    2. SEG 只保留变化最大的前几个(原来噪声也报)
    3. 新增 CHORD:chroma + 和弦模板匹配(自己实现,无新依赖)

设计原则:
    - 只输出「测量结果」,不做推断
    - 尽量紧凑(目标几百 token)
    - 不确定的东西标明不确定
"""
import re
import sys
import os
import json
import csv
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt

# YAMNet 是可选增强:模型/运行时不在就跳过,不影响主体分析
try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
YAMNET_MODEL = os.path.join(HERE, "yamnet.onnx")
YAMNET_CLASSES = os.path.join(HERE, "yamnet_class_map.csv")
MUSICNN_MODEL = os.path.join(HERE, "musicnn.onnx")
# MusicNN(MTT) 的 50 个标签 —— 训练集是西方音乐,不含中国民乐
MUSICNN_LABELS = [
    "guitar", "classical", "slow", "techno", "strings", "drums", "electronic", "rock",
    "fast", "piano", "ambient", "beat", "violin", "vocal", "synth", "female", "indian",
    "opera", "male", "singing", "vocals", "no vocals", "harpsichord", "loud", "quiet",
    "flute", "woman", "male vocal", "no vocal", "pop", "soft", "sitar", "solo", "man",
    "classic", "choir", "voice", "new age", "dance", "male voice", "female vocal",
    "beats", "harp", "cello", "no voice", "weird", "country", "metal", "female voice",
    "choral",
]

# AudioSet 的『上位词/总类』——任何音乐都会在这些上面爆高分,把细分全压平
YAMNET_GENERIC = {
    "Music", "Musical instrument", "Sound effect", "Silence",
    "Speech", "Speech synthesizer",
    "Inside, small room", "Inside, large room or hall", "Inside, public space",
    "Outside, urban or manmade", "Outside, rural or natural",
    "Environmental noise", "Noise", "Ambient noise", "Background noise",
    "Vehicle", "Animal", "Wild animals",
}

# 音乐/人声相关的关键词,用于从 521 类里筛出有意义的项
YAMNET_MUSIC_HINTS = (
    "music", "singing", "song", "vocal", "choir", "humming", "rap", "chant",
    "guitar", "bass", "piano", "keyboard", "drum", "snare", "hi-hat", "cymbal",
    "percussion", "synthesizer", "synth", "violin", "cello", "viola", "strings",
    "trumpet", "saxophone", "flute", "clarinet", "harmonica", "organ", "harp",
    "bell", "mallet", "marimba", "xylophone", "banjo", "mandolin", "ukulele",
    "accordion", "brass", "woodwind", "electronic", "beat", "rhythm", "melody",
    "chord", "riff", "plucked", "bowed", "wind instrument", "string instrument",
)

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# 这些标签看着像『听到了什么』,其实描述的是情绪/场景/风格。
# AudioSet 没有『抒情/慢板』类别,模型找不到就跳到这些近似替身 ——
# 所以 brief 里直接剔除,不把它们当成乐器证据(乐器/地域标签则保留)。
YAMNET_SOFT_LABELS = {
    "Sad music", "Tender music", "Happy music", "Funny music", "Scary music",
    "Christmas music", "Christian music", "New-age music", "Background music",
    "Soundtrack music", "Lullaby", "Electronic music", "Pop music", "Rock music",
    "Hip hop music", "Heavy metal", "Jazz", "Classical music", "Folk music",
    "Country", "Blues", "Reggae", "Salsa", "Punk rock", "Drum and bass",
    "Dubstep", "Techno", "House music", "Electronica", "Ambient music",
    "Independent music", "Soul music", "Rhythm and blues", "Funk", "Disco",
    "Gospel music", "Children's music", "Karaoke", "Electronic dance music",
    "Dance music", "Dub", "Reggaeton", "Ska", "Punk", "Music for children",
}

# Krumhansl-Schmuckler 调性模板
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# 和弦模板(相对根音的半音间隔)
CHORD_TYPES = {
    "":     [0, 4, 7],          # 大三
    "m":    [0, 3, 7],          # 小三
    "7":    [0, 4, 7, 10],      # 属七
    "m7":   [0, 3, 7, 10],      # 小七
    "maj7": [0, 4, 7, 11],      # 大七
    "sus4": [0, 5, 7],          # 挂四
    "dim":  [0, 3, 6],          # 减三
}

# 和弦权重:sus4 与大三和弦只差『三度音 vs 四度音』,混音里三度被掩蔽时它极易抢答。
# 但它本身在实际音乐里罕见,不该和三和弦平起平坐 —— 降权,只在明显像时才判它。
CHORD_WEIGHT = {
    "":     1.00,
    "m":    1.00,
    "7":    0.98,
    "m7":   0.98,
    "maj7": 0.98,
    "sus4": 0.88,
    "dim":  0.95,
}

# 和弦性质族:七度/挂留/减三都折叠回『大三族』或『小三族』。
# 理由见 chord_analysis 的注释 —— 混音上的 chroma 只够猜根音。
CHORD_FAMILY = {
    "": "maj", "7": "maj", "maj7": "maj", "sus4": "maj",
    "m": "min", "m7": "min", "dim": "min",
}

# 判『根音未定』的门槛:最佳与次佳根音分数差小于它,就不报确定根音。
# 实测(钢琴曲样本):根音差距中位数 0.051,低于 0.025 的块约占 21%。
ROOT_AMBIG_GAP = 0.025
# 判『大/小性质未定』的门槛:同一根音下大三族与小三族分数差小于它,就不报性质。
QUALITY_AMBIG_GAP = 0.02


def detect_key(chroma):
    """返回 (调性名, 最佳相关度, 最佳与次佳的差距, 前 3 候选)"""
    profile = chroma.mean(axis=1)
    rs = []
    for i in range(12):
        for mode, tmpl in (("major", MAJOR), ("minor", MINOR)):
            r = float(np.corrcoef(profile, np.roll(tmpl, i))[0, 1])
            rs.append((r, f"{NOTES[i]} {mode}"))
    rs.sort(reverse=True)
    best_r, best_name = rs[0]
    gap = best_r - rs[1][0] if len(rs) > 1 else 0.0
    return best_name, best_r, gap, rs[:3]


def key_ambiguity(best_name, top3, gap):
    """
    区分调性不确定的『类型』—— 这几种完全不是一回事:

      clear    : 最佳明显胜出,直接就是明确调性,不必谈歧义
      relative : 关系大小调 —— 差小三度,模式相反,两人共用全部音阶
      fifth    : 属调/下属调 —— 差纯五度或纯四度,模式相同,只差一个音
      parallel : 平行大小调 —— 同根音,模式相反
      weak     : 候选之间没有已知关系,且差距不大

    先看『差距』再看『关系』:原来只按关系分类,于是 D minor(0.89)
    对上 A# major(0.72)这种差距明显的情况,也被说成了『调性不明确』。
    """
    if len(top3) < 2:
        return "weak", "调性不明确"
    if gap >= 0.15:
        return "clear", f"调性明确({best_name})"

    second_name = top3[1][1]
    try:
        root = NOTES.index(best_name.split()[0])
        second_root = NOTES.index(second_name.split()[0])
    except ValueError:
        return "weak", "调性不明确"
    best_mode = best_name.split()[1]
    second_mode = second_name.split()[1]
    diff = (root - second_root) % 12
    same_mode = (best_mode == second_mode)

    if not same_mode and diff in (3, 9):
        return "relative", f"关系大小调({best_name} / {second_name})"
    if same_mode and diff in (5, 7):
        return "fifth", f"属调/下属调({best_name} / {second_name})"
    if not same_mode and diff == 0:
        return "parallel", f"平行大小调({best_name} / {second_name})"
    return "weak", f"两个候选无已知关系({best_name} / {second_name})"


def _beatthis_model_dir():
    """Beat This 模型目录(用户数据区)。模型缺失时返回 None。"""
    _data = os.environ.get("DSH_AUDIO_DATA_DIR")
    root = (os.path.join(_data, "models", "beat_this") if _data
            else os.path.join(HERE, "dsh-soundcheck-data", "models", "beat_this"))
    return root if os.path.exists(os.path.join(root, "beat-this-final0.onnx")) else None


def detect_tempo_beatthis(y, sr):
    """Beat This ONNX 检测拍点(可选组件)。返回 (bpm, beat_times) 或 None。

    输入规格来自 frontend.json: 22050Hz / n_fft 1024 / hop 441 / 128 mel。
    整首按 30 秒分块 —— Transformer 的 attention 对长序列会吃掉大量内存。

    2026-10-21 实测:它能解决 librosa 的倍频锁错 ——
    《月之暗面》69.8(真值 69)、November Rain 78.9(真值约 80),
    而 librosa 分别报 139.7 / 156.6。
    """
    model_dir = _beatthis_model_dir()
    if model_dir is None:
        return None
    try:
        import onnxruntime as ort
        with open(os.path.join(model_dir, "frontend.json"), encoding="utf-8") as fh:
            fe = json.load(fh)
        win = np.array(fe["window"], dtype=np.float32)
        melf = np.array(fe["mel_filters"], dtype=np.float32)
        sr_bt, nfft, hop = fe["sample_rate"], fe["n_fft"], fe["hop_length"]
        fps = sr_bt / hop
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(os.path.join(model_dir, "beat-this-final0.onnx"),
                                    opts, providers=["CPUExecutionProvider"])
        yy = librosa.resample(y, orig_sr=sr, target_sr=sr_bt) if sr != sr_bt else y
        step = int(30 * sr_bt)
        peaks = []
        for start in range(0, len(yy), step):
            seg = yy[start:start + step]
            if len(seg) < sr_bt * 5:
                break
            S = librosa.stft(seg, n_fft=nfft, hop_length=hop, window=win)
            feat = np.log(melf @ (np.abs(S) ** 2) + 1e-6).T[None].astype(np.float32)
            beat, _down = sess.run(None, {"spectrogram": feat})
            b = 1.0 / (1.0 + np.exp(-beat[0]))
            pk = [i for i in range(1, len(b) - 1)
                  if b[i] > 0.5 and b[i] >= b[i - 1] and b[i] >= b[i + 1]]
            peaks.extend(start / sr_bt + p / fps for p in pk)
        ts = np.array(sorted(peaks))
        if len(ts) < 4:
            return None
        return 60.0 / float(np.median(np.diff(ts))), ts
    except Exception:
        return None


def detect_tempo(y, sr):
    """检测拍速。有 Beat This 模型就用它,否则回退 librosa。

    返回 (tempo, tempo_alt, beat_times, kick_bpm, source)。
    """
    bt = detect_tempo_beatthis(y, sr)
    if bt is not None:
        bpm, times = bt
        return bpm, float("nan"), times, float("nan"), "beat_this"

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    times = librosa.frames_to_time(beats, sr=sr)
    if len(times) > 3:
        gaps = np.diff(times)
        med = float(np.median(gaps))
        alt = 60.0 / med if med > 0 else tempo
    else:
        alt = float("nan")

    # 低频鼓点交叉验证。
    # beat tracker 对 bassline 重的电子乐很容易锁在半速(每两拍算一拍),
    # 而它自带的『节拍间隔反推』发现不了 —— 因为两者来自同一个错误判断。
    # 这里独立地看 30-200Hz 的 onset 间隔,给一个第二意见。
    kick_bpm = float("nan")
    try:
        sos = butter(2, [30 / (sr / 2), 200 / (sr / 2)], btype="band", output="sos")
        yl = sosfiltfilt(sos, y).astype(np.float32)
        env = librosa.onset.onset_strength(y=yl, sr=sr, hop_length=512)
        fr = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=512)
        kt = librosa.frames_to_time(fr, sr=sr, hop_length=512)
        kg = np.diff(kt)
        kg = kg[kg > 0.15]
        if len(kg) >= 20:
            kick_bpm = 60.0 / float(np.median(kg))
    except Exception:
        pass

    return tempo, alt, times, kick_bpm, "librosa"


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def build_family_templates():
    """返回 [(模板向量, 根音, 性质, 具体名, 权重)]"""
    out = []
    for tname, intervals in CHORD_TYPES.items():
        for root in range(12):
            v = np.zeros(12)
            for iv in intervals:
                v[(root + iv) % 12] = 1.0
            out.append((_unit(v), root, CHORD_FAMILY[tname],
                        f"{NOTES[root]}{tname}", CHORD_WEIGHT[tname]))
    return out


def chord_analysis(chroma, sr, hop=1024, block_sec=2.0, t_offset=0.0):
    """
    和弦:降级为『根音候选骨架』,不再报确定的和弦名。

    为什么降级 —— 两次实测说明的:
      1. 在混音上做 84 个和弦模板匹配,『最佳 − 次佳』的相似度差距中位数
         只有 0.026,这个量级和噪声分不开。所以旧的『71~82% 未定率』
         并不是识别失败,而是**和弦性质本来就分辨不出来**。
      2. 但把七度/挂留/减三折叠掉、只比『根音』之后,差距中位数升到 0.051,
         约 79% 的块有明显胜出者。
    结论:混音上的 chroma 够用来猜根音,不够用来定和弦性质。
    于是只输出根音骨架 + 性质(高置信时才给),其余一律标未定。

    另外:旧 JSON 里的 chord_unresolved_ratio 是 `1 - 变化点数/块数`,
    那其实是『和弦没变的比例』,标签撒谎。现在换成真正的未定比例。

    返回 dict(mode/skeleton/vocabulary/unresolved_ratio/...)
    """
    n_frames = chroma.shape[1]
    if n_frames == 0:
        return {
            "mode": "root_candidates", "skeleton": [], "vocabulary": [],
            "unresolved_ratio": 1.0, "quality_unresolved_ratio": 1.0,
            "mean_root_gap": 0.0, "block_sec": block_sec, "blocks_total": 0,
            "note": "空输入",
        }

    fam = build_family_templates()
    vecs = np.array([f[0] for f in fam])                              # (N, 12)
    weights = np.array([f[4] for f in fam])[:, None]
    sims = (vecs @ chroma) * weights                                  # (N, frames)

    # 每个根音分别取『大三族』『小三族』里的最高分 (12, frames)
    maj = np.zeros((12, n_frames))
    mn = np.zeros((12, n_frames))
    for i, f in enumerate(fam):
        target = maj if f[2] == "maj" else mn
        np.maximum(target[f[1]], sims[i], out=target[f[1]])

    per = max(1, int(block_sec * sr / hop))
    blocks = []
    for i0 in range(0, n_frames, per):
        i1 = min(i0 + per, n_frames)
        if i1 <= i0:
            continue
        bmaj = maj[:, i0:i1].mean(axis=1)
        bmin = mn[:, i0:i1].mean(axis=1)
        root_score = np.maximum(bmaj, bmin)
        order = np.argsort(-root_score)
        root = int(order[0])

        # 同根音下的具体和弦(七度/挂留):只留作参考,不进结论
        det_idx = [k for k, f in enumerate(fam) if f[1] == root]
        det_score = np.array([sims[k, i0:i1].mean() for k in det_idx])
        det_order = np.argsort(-det_score)

        blocks.append({
            "start": t_offset + i0 * hop / sr,
            "end": t_offset + i1 * hop / sr,
            "root": NOTES[root],
            "root_gap": float(root_score[order[0]] - root_score[order[1]]),
            "quality": "maj" if bmaj[root] >= bmin[root] else "min",
            "quality_gap": float(abs(bmaj[root] - bmin[root])),
            "detail": fam[det_idx[det_order[0]]][3],
            "detail_gap": float(det_score[det_order[0]] - det_score[det_order[1]])
            if len(det_order) > 1 else 0.0,
        })

    # 相邻同根音合并成 run
    runs = []
    for b in blocks:
        if runs and runs[-1]["root"] == b["root"]:
            runs[-1]["blocks"].append(b)
            runs[-1]["end"] = b["end"]
        else:
            runs.append({"root": b["root"], "blocks": [b],
                         "start": b["start"], "end": b["end"]})

    # 抖动过滤:一个根音至少持续一块多才算真的换了
    kept = []
    for idx, r in enumerate(runs):
        if idx == 0 or (r["end"] - r["start"]) >= block_sec * 1.25:
            if kept and kept[-1]["root"] == r["root"]:
                kept[-1]["blocks"].extend(r["blocks"])
                kept[-1]["end"] = r["end"]
            else:
                kept.append(r)

    skeleton = []
    for r in kept:
        shares = [b["quality"] == "maj" for b in r["blocks"]]
        maj_share = float(np.mean(shares))
        if maj_share >= 0.75:
            quality = "maj"
        elif maj_share <= 0.25:
            quality = "min"
        else:
            quality = None
        skeleton.append({
            "start_sec": round(r["start"], 1),
            "end_sec": round(r["end"], 1),
            "root": r["root"],
            "quality": quality,
            "root_gap": round(float(np.mean([b["root_gap"] for b in r["blocks"]])), 3),
            "major_share": round(maj_share, 2),
        })

    total = len(blocks)
    acc = {}
    for b in blocks:
        a = acc.setdefault(b["root"], {"n": 0, "maj": 0})
        a["n"] += 1
        if b["quality"] == "maj":
            a["maj"] += 1
    vocabulary = [
        {"root": root, "share": round(a["n"] / total, 3),
         "major_share": round(a["maj"] / a["n"], 2)}
        for root, a in sorted(acc.items(), key=lambda kv: -kv[1]["n"])
    ]

    gaps = np.array([b["root_gap"] for b in blocks])
    qgaps = np.array([b["quality_gap"] for b in blocks])
    return {
        "mode": "root_candidates",
        "skeleton": skeleton,
        "vocabulary": vocabulary,
        "unresolved_ratio": round(float((gaps < ROOT_AMBIG_GAP).mean()), 3),
        "quality_unresolved_ratio": round(float((qgaps < QUALITY_AMBIG_GAP).mean()), 3),
        "mean_root_gap": round(float(gaps.mean()), 4),
        "block_sec": block_sec,
        "blocks_total": total,
        "note": ("只报根音级候选:混音上做模板匹配,七度/挂留/转位会被掩蔽;"
                 "根音也不是每块都定得下来 —— unresolved_ratio 是定不下来的比例。"
                 "skeleton 是候选骨架,不是和弦进行"),
    }



def assess_fit(chroma, beat_times, key_gap, key_r, chord_mean_gap):
    """
    评估本工具对这段音频的『适用性』(0~1,越高越可信)。

    每个维度都必须区分两种完全不同的失败 —— 只看其中一种就会误判:

      ① 体系外:绝对可信度低(例:维吾尔木卡姆 vs 西方调性模板)
                → 该报『不适用』
      ② 体系内歧义:绝对可信度高,但最佳与次佳咬得紧(例:属调混淆)
                → 该报『可用,但有歧义』

    早先只看 ②,结果把 ① 也当成了『可用但有歧义』—— 这是最危险的那种错。
    """
    fit = {}

    # --- beat:节拍间隔的稳定性(变异系数越小越稳)---
    if len(beat_times) > 3:
        gaps = np.diff(beat_times)
        cv = float(np.std(gaps) / (np.mean(gaps) + 1e-8))
        fit["beat"] = float(np.clip(1.0 - cv, 0.0, 1.0))
    else:
        fit["beat"] = 0.0

    # --- tonality:绝对相关度(门槛) × 最佳次佳差距(质量)---
    # 绝对相关度:0.55 以下认为模板不适用,0.90 以上认为贴合
    abs_conf = float(np.clip((key_r - 0.55) / 0.35, 0.0, 1.0))
    # 差距:0.25 以上认为调性唯一
    gap_conf = float(np.clip(key_gap / 0.25, 0.0, 1.0))
    # 取两者的『短板』:绝对度低就直接压下去,绝不被差距大抬回来
    fit["tonality"] = min(abs_conf, 0.35 + 0.65 * gap_conf)

    # --- chords:用『根音』层面的最佳与次佳差距,而不是具体和弦的差距 ---
    # 具体和弦的差距中位数只有 0.03,低于噪声;根音层面约 0.05,才有区分力。
    # 所以这里评估的是『根音骨架能信多少』,不是『和弦名对不对』。
    fit["chords"] = float(np.clip(chord_mean_gap / 0.10, 0.0, 1.0))

    return fit


def analyze_sources(y, sr, top_k=8):
    """
    YAMNet 声源/乐器分类(可选增强)。

    返回 None = 不可用(onnxruntime 或模型文件缺失),调用方应静默跳过。
    y/sr 由调用方加载后传入 —— 以前这里自己再解码一次文件,等于把
    mp3 解两遍(4 分钟的歌多花十几秒)。y 只在内存里一份,重采样到 16k 即可。

    ⚠ 使用限制(实测四首歌得出的):
      - 乐器/地域线索相对可靠(Percussion / Singing / Music of Asia)
      - 『情绪』不可信 —— AudioSet 没有"抒情/慢板"这类类别,
        模型找不到时会跳到 Christmas / Christian 这种特征相近的类里
      - 分数普遍很低(0.01~0.05),因为训练数据偏西方流行
    """
    if ort is None:
        return None
    if not (os.path.exists(YAMNET_MODEL) and os.path.exists(YAMNET_CLASSES)):
        return None

    try:
        names = []
        with open(YAMNET_CLASSES, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                names.append(row["display_name"])

        y16 = y if sr == 16000 else librosa.resample(y, orig_sr=sr, target_sr=16000)
        sess = ort.InferenceSession(YAMNET_MODEL, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        scores = sess.run(None, {inp: np.ascontiguousarray(y16, dtype=np.float32)})[0]
    except Exception as e:
        return {"available": False, "error": str(e)[:200]}

    mean = scores.mean(axis=0)
    mask = np.array([0.0 if n in YAMNET_GENERIC else 1.0 for n in names])
    sel = mean * mask

    top = []
    for i in np.argsort(-sel)[:top_k]:
        if float(sel[i]) <= 0:
            continue
        top.append({"label": names[i], "score": round(float(sel[i]), 3)})

    music = []
    for i in np.argsort(-sel):
        n = names[i]
        if n in YAMNET_GENERIC:
            continue
        if any(h in n.lower() for h in YAMNET_MUSIC_HINTS) and float(sel[i]) >= 0.005:
            music.append({"label": n, "score": round(float(sel[i]), 3)})
        if len(music) >= 10:
            break

    max_score = max((t["score"] for t in top), default=0.0)
    # 去掉情绪/场景/风格替身,只留『可能是某种乐器/地域』的线索
    hints = [t for t in top if t["label"] not in YAMNET_SOFT_LABELS][:6]

    # 人声证据(逐窗口): Singing + Speech 取较大者。
    # 不用 Humming —— 实测纯钢琴段给 0.205、真人声段只给 0.020,完全反向。
    # 阈值 0.15/0.06 来自 4 首歌 + 1 个构造样本(3 分钟纯器乐 + 末尾 10 秒人声):
    # 有唱 max 0.28–0.34,没唱 0.03–0.05,中间留 unknown。
    vocal = None
    try:
        vc_idx = [i for i, n in enumerate(names) if n in ("Singing", "Speech")]
        if vc_idx:
            vs = scores[:, vc_idx].max(axis=1)
            hop_sec = (len(y16) / 16000) / max(1, scores.shape[0])
            peak = float(vs.max())
            peak_t = float(int(np.argmax(vs)) * hop_sec)
            # 逐帧 >=0.15 的连续区间，间隔 <=2 秒的接起来，短于 1 秒的丢掉。
            # 这是「哪几秒像有人在唱」—— 之前只取峰值，把时间信息扔了。
            _segs, _cur = [], None
            for _fi, _val in enumerate(vs):
                _t = _fi * hop_sec
                if _val >= 0.15:
                    _cur = [_t, _t + hop_sec] if _cur is None else [_cur[0], _t + hop_sec]
                elif _cur is not None:
                    _segs.append(tuple(_cur)); _cur = None
            if _cur is not None:
                _segs.append(tuple(_cur))
            _merged = []
            for _s, _e in _segs:
                if _merged and _s - _merged[-1][1] <= 2.0:
                    _merged[-1] = (_merged[-1][0], _e)
                else:
                    _merged.append((_s, _e))
            segs = [(round(_s, 1), round(_e, 1)) for _s, _e in _merged if _e - _s >= 1.0]
            vocal = {
                "peak": round(peak, 3),
                "peak_t": round(peak_t, 1),
                "p95": round(float(np.percentile(vs, 95)), 3),
                "segments": segs,
                "verdict": "有" if peak >= 0.15 else ("无" if peak <= 0.06 else "unknown"),
            }
    except Exception:  # noqa: BLE001
        vocal = None

    return {
        "available": True,
        "top": top,
        "music_related": music,
        "hints": hints,
        "max_score": max_score,
        "vocal": vocal,
        "low_confidence": bool(max_score < 0.1),
        "caveat": "AudioSet 缺少『抒情/慢板』类别,情绪标签(Sad/Tender/Christmas)不可信;"
                  "乐器与地域线索相对可靠",
    }


def analyze_musicnn(y, sr, min_score=0.15, top_k=3):
    """
    MusicNN(MTT 50 类) 音乐标签 —— 可选增强,模型缺失时返回 None。

    ⚠ 这是『模型打的标签』,不是『识别结果』:
      - 训练集是西方音乐,标签里没有马头琴/古筝/唢呐
      - 会给出明显错误的标签(实测把一首大提琴 cover 打成 techno=0.40)
      - 所以只在 score 超过阈值时才报,措辞必须是「模型标签」
    y/sr 由调用方传入,不重复解码文件。
    """
    if ort is None or not os.path.exists(MUSICNN_MODEL):
        return None
    try:
        y16 = y if sr == 16000 else librosa.resample(y, orig_sr=sr, target_sr=16000)
        mel = librosa.feature.melspectrogram(
            y=y16, sr=16000, hop_length=256, n_fft=512, n_mels=96).T
        mel = np.log10(10000 * mel + 1).astype(np.float32)
        n_frames = int(librosa.time_to_frames(3.0, sr=16000, n_fft=512,
                                              hop_length=256)) + 1
        step = max(1, n_frames)
        patches = [mel[i:i + n_frames]
                   for i in range(0, max(1, len(mel) - n_frames + 1), step)]
        patches = [p for p in patches if p.shape[0] == n_frames]
        if not patches:
            return None
        batch = np.stack(patches)
        sess = ort.InferenceSession(MUSICNN_MODEL, providers=["CPUExecutionProvider"])
        logits = sess.run(None, {"mel": batch})[0]
        probs = 1.0 / (1.0 + np.exp(-logits))
        avg = probs.mean(axis=0)
    except Exception as e:
        return {"available": False, "error": str(e)[:200]}

    order = np.argsort(-avg)
    tags = [{"label": MUSICNN_LABELS[i], "score": round(float(avg[i]), 2)}
            for i in order if float(avg[i]) >= min_score][:top_k]
    return {
        "available": True,
        "tags": tags,
        "caveat": "MTT 50 类训练集为西方音乐,不含中国民乐;标签可能明显错误",
    }


# 分频段诊断的默认频段(Hz),低→高
BANDS_DEFAULT = [
    ("30-250", 30, 250),
    ("250-800", 250, 800),
    ("800-2500", 800, 2500),
    ("2500-6000", 2500, 6000),
    ("6000-12000", 6000, 12000),
]


def analyze_bands(y, sr, bands=None):
    """分频段统计打击层:每段的能量占比、打击密度、间隔规律性(cv)、主周期。

    用途(内部诊断,不是给用户看的报告):
      - 判断一段音频是音乐还是野外录音(BPM/调性在非音乐内容上会硬套)
      - 验证用户说的「沙锤/鼓/镲」落在哪个频段
      - 分不清高低频谁在主导时定位
    cv < 0.5 视为「规律」(间隔稳定),越大越散。
    """
    from scipy.signal import find_peaks

    bands = bands or BANDS_DEFAULT
    hop = 512
    fps = sr / hop
    dur = len(y) / sr if len(y) else 0.0
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    total = float(np.sum(S ** 2)) + 1e-12

    rows = []
    for name, lo, hi in bands:
        sel = (freqs >= lo) & (freqs < hi)
        if not sel.any():
            continue
        band = S[sel, :]
        energy = float(np.sum(band ** 2)) / total
        env = np.sum(band, axis=0)
        diff = np.diff(env)
        diff[diff < 0] = 0
        peak = float(diff.max()) if diff.size else 0.0
        if peak > 0:
            diff = diff / peak
        pk, _ = find_peaks(diff, height=0.20, distance=max(1, int(0.12 * fps)))
        row = {"band": name, "energy": round(energy, 3), "hits": int(len(pk))}
        if len(pk) >= 4 and dur > 0:
            times = pk / fps
            gaps = np.diff(times)
            gaps = gaps[gaps > 0]
            if len(gaps) >= 3:
                med = float(np.median(gaps))
                row["per_sec"] = round(len(pk) / dur, 2)
                row["cv"] = round(float(np.std(gaps) / (np.mean(gaps) + 1e-12)), 2)
                row["period"] = round(med, 3)
        rows.append(row)
    return rows


def format_bands(rows):
    """把 analyze_bands 的结果格式化成可读行。"""
    lines = []
    for r in rows:
        seg = f"{r['band']:>9}Hz 能量 {r['energy'] * 100:>5.1f}%"
        if r.get("cv") is None:
            seg += " | 打击太少,不成节奏"
        else:
            cv = r["cv"]
            tag = "规律" if cv < 0.5 else ("一般" if cv < 1.0 else "不规律")
            seg += f" | 打击 {r.get('per_sec', 0):.1f}/秒 | cv {cv:.2f}({tag})"
            if r.get("period"):
                seg += f" | 周期 {r['period']:.2f}s"
        lines.append("       " + seg)
    return lines


def activity_timeline(y, sr, block_sec=16.0, hop=1024):
    """
    编曲活动度 —— 和「电平」分开的另一个维度。

    为什么要分开:RMS 涨了只能证明「电平变了」,
    不能证明「配器变密了、乐器变多了、鼓更凶了」。
    这两个维度混在一起,就分不清「编曲有对比但响度被压平」这种情况。

    三个指标都与整体增益弱相关(逐帧归一化后算):
      onset_rate   起音密度(次/秒)—— 多少瞬态事件
      active_bands 活跃频带占比 —— 有多少个倍频程带显著高于噪声底
      flux         频谱变化率 —— 频谱随时间抖动的强度

    返回 (整体汇总, 分块时间线)
    """
    dur = len(y) / sr

    # 1) 起音密度
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop)
    onset_rate = len(onsets) / dur if dur > 0 else 0.0

    # 2) 频谱(逐帧归一化,削掉整体增益的影响)
    S = np.abs(librosa.stft(y, hop_length=hop))
    S_norm = S / (S.sum(axis=0, keepdims=True) + 1e-9)

    # 3) 活跃频带占比(6 个倍频程带)
    freqs = librosa.fft_frequencies(sr=sr)
    band_edges = [(30, 120), (120, 300), (300, 800), (800, 2000), (2000, 5000), (5000, 12000)]
    band_energy = []
    for lo, hi in band_edges:
        m = (freqs >= lo) & (freqs < hi)
        band_energy.append(float(S[m].mean()) if m.any() else 0.0)
    be = np.array(band_energy)
    active_bands = float((be > be.max() * 0.1).mean()) if be.max() > 0 else 0.0

    # 4) 频谱变化率
    flux = float(np.mean(np.sqrt(np.sum(np.diff(S_norm, axis=1) ** 2, axis=0)))) if S_norm.shape[1] > 1 else 0.0

    summary = {
        "onset_rate_per_sec": round(onset_rate, 2),
        "active_band_ratio": round(active_bands, 2),
        "spectral_flux": round(flux, 4),
    }

    # 分块:让「活动度」也能看时间变化,好和电平曲线对照
    per = max(1, int(block_sec * sr / hop))
    blocks = []
    for i in range(0, S.shape[1], per):
        seg = S_norm[:, i:i + per]
        if seg.shape[1] < 2:
            continue
        # 本块内有起音吗(用 onset 帧号落在本块的比例近似密度)
        lo_f = i
        hi_f = i + per
        n_on = int(np.sum((onsets >= lo_f) & (onsets < hi_f)))
        seg_flux = float(np.mean(np.sqrt(np.sum(np.diff(seg, axis=1) ** 2, axis=0))))
        # 本块的频带占比
        seg_e = []
        for lo, hi in band_edges:
            m = (freqs >= lo) & (freqs < hi)
            seg_e.append(float(S[m, i:i + per].mean()) if m.any() else 0.0)
        se = np.array(seg_e)
        ratio = float((se > se.max() * 0.1).mean()) if se.max() > 0 else 0.0
        blocks.append({
            "start_sec": round(lo_f * hop / sr, 1),
            "end_sec": round(hi_f * hop / sr, 1),
            "onsets": n_on,
            "active_band_ratio": round(ratio, 2),
            "spectral_flux": round(seg_flux, 4),
        })

    return summary, blocks


def narrate(d):
    """
    把测量结果组织成人能读的话。

    规矩:每一句都必须能追回到上面的某个数字。
    做不到的就说"不知道",不许用"感觉"补。
    """
    out = []
    dur, tempo, kick = d["dur"], d["tempo"], d["kick_bpm"]
    rms, estd = d["rms_db"], d["energy_std"]
    act = d["activity"]
    fit, key, key_r = d["fit"], d["key"], d["key_r"]
    src = d["sources"]
    section = bool(d.get("section"))

    # --- 1. 一句话定性:时长 + 密度 ---
    minutes = dur / 60
    density = act["onset_rate_per_sec"]
    if density > 3.0:
        den_txt = "节奏很密"
    elif density > 2.0:
        den_txt = "节奏偏密"
    elif density > 1.0:
        den_txt = "节奏中等"
    else:
        den_txt = "节奏稀疏"

    band = act["active_band_ratio"]
    if band < 0.55:
        band_txt = "能量压在低频,高频比较干净"
    else:
        band_txt = "频谱铺得比较开"

    lead = f"{dur:.0f} 秒的片段" if section else f"{minutes:.1f} 分钟"
    out.append(f"{lead},{den_txt}({density:.1f} 次/秒起音),{band_txt}。")

    # --- 2. 声源(只取剔除情绪/风格替身后的线索,有就说是猜的) ---
    if src and src.get("available") and src.get("hints"):
        top = src["hints"][:3]
        names = "、".join(t["label"] for t in top)
        weak = "(分数很低,只是方向,不是结论)" if src.get("low_confidence") else ""
        out.append(f"声源线索:{names}{weak}。")

    # --- 3. 结构:电平 vs 密度 分开讲 ---
    if estd < 5:
        out.append(f"电平几乎一直满着(段间只差 {estd:.1f} dB),没有明显的段落对比。")
    elif estd < 9:
        out.append(f"电平有些起伏(段间 {estd:.1f} dB),但不算剧烈。")
    else:
        out.append(f"电平起伏明显(段间 {estd:.1f} dB),段落之间是有强弱对比的。")

    # 活动度在时间上的变化 → 和电平对比着说
    ab = d.get("activity_blocks") or []
    if ab:
        ons = [b["onsets"] for b in ab]
        if ons and max(ons) > 0:
            ratio = (max(ons) - min(ons)) / max(max(ons), 1)
            if ratio > 0.7:
                out.append("但「密度」在各段差别很大 —— 说明编曲层面是有变化的。")
            else:
                tail = "这段" if section else "整首"
                out.append(f"「密度」在各段也比较平均,{tail}是一个稳定推进的状态。")

    # --- 4. 调性 ---
    if key_r >= 0.75:
        out.append(f"调性 {key},匹配得比较稳。")
    elif key_r >= 0.65:
        out.append(f"调性大概在 {key} 附近,但有摇摆。")
    else:
        out.append("调性拿不准 —— 西方调性模板对这段音乐不太适用。")

    # --- 5. BPM 存疑要明说 ---
    if not np.isnan(kick) and tempo > 0 and (1.6 < kick / tempo < 2.4 or 0.4 < kick / tempo < 0.65):
        # 不能说『真速是后者』:钢琴/无鼓曲目里,低频第二意见常把低音音符当鼓点。
        # 两个数差约 2 倍时,更可能是『整拍 vs 细分』,不是谁算错了。
        out.append(f"速度有两个层级:检测器说 {tempo:.0f},低频第二意见说 {kick:.0f}"
                   f"(约 {kick / tempo:.1f} 倍)。两边都有能量,分不清是整拍还是细分 —— "
                   "按你数拍的方式定,别把其中一个当唯一答案。")
    else:
        if section:
            out.append(f"片段测速 {tempo:.0f} BPM —— 短片段容易锁到别的拍层级,"
                       "要全曲速度请分析整首。")
        else:
            out.append(f"速度 {tempo:.0f} BPM。")

    # --- 6. 和弦:说清楚只到根音级 ---
    cr = d.get("chords")
    if cr and cr.get("vocabulary"):
        roots = "、".join(v["root"] for v in cr["vocabulary"][:5])
        out.append(f"和弦只能到根音级 —— 主要出现 {roots};"
                   f"根音未定 {cr['unresolved_ratio'] * 100:.0f}% 的块、"
                   f"性质未定 {cr['quality_unresolved_ratio'] * 100:.0f}% 的块。"
                   "这只能说明『大概在这几个根音之间转』,不是和弦进行。")
    else:
        out.append("和弦完全没结果。")

    # --- 7. 局限(必须说) ---
    limits = []
    limits.append("听不到旋律、音色、唱的是什么")
    limits.append("没有做音源分离,所有结论都来自混音整体")
    out.append("不知道的:" + ";".join(limits) + "。")

    return "\n".join("  " + line for line in out)


def fit_line(name, val, good_hint, bad_hint):
    """把 0~1 的适用性渲染成一行的可读文本。

    bad_hint 必须自己说清楚『为什么』—— 早先这里硬加前缀『本曲不适用』,
    于是出现了『tonality 0.41 不可用 —— 与属调混淆,中心基本明确』
    这种自相矛盾的行。前缀不再撒谎:只标 ⚠,理由由调用方给。
    """
    if val >= 0.65:
        tag, hint = "可用", good_hint
    elif val >= 0.45:
        tag, hint = "偏低", f"仅作参考 —— {bad_hint}"
    else:
        tag, hint = "不可用", f"⚠ {bad_hint}"
    return f"  {name:<9}{val:.2f}  {tag:<5} {hint}"


def tonality_hint(key_r, key_gap, ka_type, ka_msg, fit_value):
    """
    调性那一行要说三种不同的话,不能混:
      ① 模板不贴合(绝对相关度低)→ 体系外,别报调
      ② 贴合且差距够大 → 明确
      ③ 贴合但差距小 → 体系内歧义,只能算猜测
    早先 ③ 用了 ② 的措辞,和 0.41 的分数摆在一起自相矛盾。
    """
    if key_r < 0.65:
        return (f"模板不适用(相关度仅 {key_r:.2f})—— "
                "可能是非西方调式、微分音,或素材本身和声信息太少")
    if fit_value >= 0.45:
        return {
            "clear":    f"调性明确 —— 次佳差距 {key_gap:.2f},足够区分",
            "relative": "关系大小调歧义 —— 中心明确,只是大/小调分不清",
            "fifth":    "与属调混淆 —— 只差一个音,中心基本明确",
            "parallel": "平行大小调歧义 —— 同根音,只是调式分不清",
            "weak":     "候选间无已知关系且差距不大,调性中心不明确",
        }[ka_type]
    return (f"模板勉强贴合(相关度 {key_r:.2f}),但次佳差距只有 {key_gap:.2f} —— "
            "候选调之间分不够开,具体是哪个调只能算猜测")


def integrated_lufs(y, sr):
    """整曲整合响度（ITU-R BS.1770），单位 LUFS。

    与 dBFS 的区别：dBFS 是数字满刻度能量，对 2–4 kHz 以上不敏感；
    LUFS 做 K 加权（高架 + 高通），更接近人耳听到的“多响”。
    同一段弱奏钢琴，dBFS 可能 −52，LUFS 只低十几——这就是那条评论说的差距。
    """
    from scipy.signal import lfilter

    if sr != 48000:
        y = librosa.resample(y, orig_sr=sr, target_sr=48000)
        sr = 48000
    # Stage 1：高架（+4 dB @ 1.5k 附近）
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    y = lfilter(b1, a1, y)
    # Stage 2：高通（RLB）
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    y = lfilter(b2, a2, y)

    block = int(0.4 * sr)
    hop = int(0.1 * sr)
    if len(y) < block:
        return float("nan")
    ms = np.array([float(np.mean(y[i:i + block] ** 2))
                   for i in range(0, len(y) - block + 1, hop)])
    ms = ms[ms > 0]
    if len(ms) == 0:
        return float("nan")
    # 绝对门限 −70 LUFS：静音块不参与平均
    lk = -0.691 + 10 * np.log10(ms)
    ms = ms[lk > -70]
    if len(ms) == 0:
        return float("nan")
    return float(-0.691 + 10 * np.log10(np.mean(ms)))


def energy_blocks(rms_db, sr, hop, block_sec):
    per = max(1, int(block_sec * sr / hop))
    out = []
    for i in range(0, len(rms_db), per):
        seg = rms_db[i:i + per]
        if len(seg) == 0:
            continue
        out.append((i * hop / sr, (i + len(seg)) * hop / sr, float(seg.mean())))
    return out


def fmt_time(t):
    """秒 → m:ss.s。先四舍五入到 0.1 秒再拆分,否则 239.999 会打成 '3:60.0'。"""
    tenths = int(round(float(t) * 10))
    if tenths < 0:
        tenths = 0
    minutes, rest = divmod(tenths, 600)
    return f"{minutes}:{rest / 10:04.1f}"


def parse_time(text):
    """接受 '90' / '1:30' / '1:02:03',返回秒。也接受 float。"""
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text).strip()
    if not s:
        raise ValueError("时间不能为空")
    parts = s.split(":")
    if len(parts) > 3:
        raise ValueError(f"时间格式不对:{text}")
    total = 0.0
    for p in parts:
        total = total * 60 + float(p)
    return total


def chord_lines(cr, chord_block):
    """渲染『根音候选骨架』。"""
    sk = cr["skeleton"]
    if not sk:
        return ["[CHORD] 未能识别"]
    out = [f"[CHORD] 根音候选骨架(每 {chord_block:g} 秒一块,共 {cr['blocks_total']} 块)",
           "  记法:D#(M)=大三、D#(m)=小三;括号内 ? = 性质未定,末尾 ? = 根音也不确定"]
    tag = {"maj": "M", "min": "m", None: "?"}
    items = []
    prev_end = None
    for s in sk:
        # 骨架里被过滤掉的抖动段会留下空洞,必须显式标出来 ——
        # 否则读起来会像『同一个根音持续了 12 秒』。
        if prev_end is not None and s["start_sec"] - prev_end > chord_block * 1.5:
            items.append(f"⟨{int(round(s['start_sec'] - prev_end))}s未定⟩")
        mark = "?" if s["root_gap"] < ROOT_AMBIG_GAP else ""
        items.append(f"{fmt_time(s['start_sec'])} {s['root']}({tag[s['quality']]}){mark}")
        prev_end = s["end_sec"]
    for i in range(0, len(items), 4):
        out.append("  " + "  |  ".join(items[i:i + 4]))
    vocab = " ".join(f"{v['root']}({v['share'] * 100:.0f}%)"
                     for v in cr["vocabulary"][:6])
    out.append(f"  候选词表:{vocab}")
    out.append(f"  根音未定 {cr['unresolved_ratio'] * 100:.0f}% 的块"
               f" | 性质未定 {cr['quality_unresolved_ratio'] * 100:.0f}% 的块")
    out.append("  ⚠ 这是候选骨架,不是和弦进行;七度/挂留/转位在混音里分不干净,已折叠")
    return out


def fmt_dur(sec):
    """秒 → 'X 分 YY 秒' / 'XX 秒'。"""
    sec = float(sec)
    if sec < 90:
        return f"{sec:.0f} 秒"
    m = int(sec // 60)
    sec_is = int(round(sec - m * 60))
    if sec_is >= 60:
        m, sec_is = m + 1, 0
    return f"{m} 分 {sec_is:02d} 秒"


def analyze_stems(path, mix_rms_db, *, model="htdemucs", cache_dir=None, want_pitch=False):
    """分轨分析:把歌拆成 4/6 轨,给出各轨能量占比和人声判定。

    分离模型(demucs-onnx)是可选组件;未安装时返回 available=False。
    结果按 (文件, 大小, 修改时间, 模型) 缓存成小 JSON,重复分析秒回。

    返回 dict:
        {"available": True, "model": "htdemucs", "mix_rms_db": -14.4,
         "stems": {"vocals": {"rms_db": -15.7, "ratio": 0.38}, ...},
         "vocals": {"present": True, "ratio": 0.38, "label": "有人声"}}
    """
    import hashlib
    import time as _time

    # 数据目录(缓存/模型/HF 缓存):优先环境变量(插件传入),否则用脚本目录。
    # 插件把这三个变量指到插件目录外, 覆盖升级插件时数据不丢。
    _data = os.environ.get("DSH_AUDIO_DATA_DIR")
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _cache_default = (os.path.join(_data, "cache", "stems") if _data
                      else os.path.join(_script_dir, "cache", "stems"))
    _model_default = (os.path.join(_data, "models", "stems") if _data
                      else os.path.join(_script_dir, "models", "stems"))
    if _data:
        os.environ.setdefault("HF_HOME", os.path.join(_data, "hf_home"))
    root = cache_dir or os.environ.get("DSH_AUDIO_CACHE_DIR") or _cache_default
    st = os.stat(path)
    # 版本号进 key：改过分析逻辑就递增，旧缓存自然失效。
    # 教训 2026-10-23：把字段名从 longest_run 改成 longest_sec 后，缓存没版本号，
    # 旧 json 里读不到新字段，于是全曲显示「最长连续 0.00 秒」——误以为是算法坏了，
    # 查到最后是缓存。凡是往缓存里加/改字段，都要动这个数字。
    raw_key = (f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}|{model}|v8")
    key = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:16]
    cpath = os.path.join(root, f"{key}.json")
    if os.path.exists(cpath):
        try:
            with open(cpath, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass

    try:
        from demucs_onnx.inference import separate
    except Exception as e:  # noqa: BLE001
        return {"available": False,
                "reason": f"分离组件未安装(demucs-onnx): {e}"}

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    model_dir = os.environ.get("DSH_AUDIO_MODEL_DIR") or _model_default
    # 优先用安装器预置的本地模型,没有再走 HF 下载(省流量、离线也能用)
    _local = os.path.join(
        model_dir,
        "htdemucs_6s_fp16weights.onnx" if model == "htdemucs_6s"
        else "htdemucs_fp16weights.onnx",
    )
    if os.path.exists(_local):
        try:
            from demucs_onnx import inference as _inf
            _inf.download_single_model = lambda *a, **k: _local
        except Exception:
            pass
    t0 = _time.time()
    print(f"[分轨] 正在分离({model})...", file=sys.stderr)
    try:
        res = separate(path, None, model=model, precision="fp16weights",
                       cache_dir=model_dir, progress=False)
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"分离失败: {e}"}

    out = {"available": True, "model": model,
           "mix_rms_db": round(float(mix_rms_db), 1),
           "elapsed_sec": round(_time.time() - t0, 1),
           "stems": {}}
    mix_energy = 10.0 ** (float(mix_rms_db) / 10.0)
    for name, wav in res.items():
        arr = np.asarray(wav, dtype=np.float64)
        rms = float(np.sqrt(np.mean(arr ** 2)))
        rms_db = 20.0 * np.log10(rms + 1e-12)
        ratio = min(1.0, (10.0 ** (rms_db / 10.0)) / max(mix_energy, 1e-12))
        out["stems"][name] = {"rms_db": round(rms_db, 1), "ratio": round(ratio, 4)}

    v = out["stems"].get("vocals")
    if v:
        r = v["ratio"]
        if r < 0.001:
            present, label = False, "纯器乐"
        elif r < 0.05:
            present, label = False, "人声很轻"
        else:
            present, label = True, "有人声"
        out["vocals"] = {"present": present, "ratio": r, "label": label}

    # 逐 16 秒块的四轨能量（给事件时间线用）：
    # 有了它才能说「是哪一部分变了」，而不只是「整体变响了」。
    try:
        import soundfile as _sf
        from demucs_onnx.inference import SAMPLE_RATE as _SR4
        try:
            _orig_dur = float(_sf.info(path).duration)
        except Exception:
            _orig_dur = None
        _step = int(16.0 * _SR4)
        _blk = {}
        for _nm2, _w2 in res.items():
            _a4 = np.asarray(_w2, dtype=np.float64)
            if _a4.ndim == 2:
                _a4 = _a4.mean(axis=0) if _a4.shape[0] <= _a4.shape[1] else _a4.mean(axis=1)
            _a4 = _a4.reshape(-1)
            _vals = []
            for _i2 in range(0, len(_a4), _step):
                # 分轨可能比原曲长（demucs 补零）——超出原曲时长的部分不要
                if _orig_dur is not None and (_i2 / _SR4) >= _orig_dur:
                    break
                _seg2 = _a4[_i2:_i2 + _step]
                if len(_seg2) < _step // 2:
                    continue
                _vals.append({
                    "start_sec": round(_i2 / _SR4, 1),
                    "end_sec": round((_i2 + len(_seg2)) / _SR4, 1),
                    "db": round(20 * np.log10(float(np.sqrt(np.mean(_seg2 ** 2))) + 1e-12), 1),
                })
            _blk[_nm2] = _vals
        out["block_db"] = _blk
    except Exception as _e2:  # noqa: BLE001
        out["block_db_error"] = f"{type(_e2).__name__}: {_e2}"

    # 无人声区间是分轨的直接产物，不该只在跑 ASR 时才算。
    vw = res.get("vocals")
    if vw is not None:
        try:
            from demucs_onnx.inference import SAMPLE_RATE as _DEMUCS_SR
            _m = np.asarray(vw, dtype=np.float64)
            if _m.ndim == 2:   # demucs 可能给 (channels, T) 或 (T, channels)
                _m = _m.mean(axis=0) if _m.shape[0] <= _m.shape[1] else _m.mean(axis=1)
            _m = _m.reshape(-1)
            out["instrumental"] = _instrumental_ranges(_m, _DEMUCS_SR)
        except Exception as e:  # noqa: BLE001
            out["instrumental"] = []
            out["instrumental_error"] = f"{type(e).__name__}: {e}"

    # 相对电平(逐 2 秒窗口,0.5 秒步进): 人声轨比混音低多少 dB。
    # 窗口必须短 —— 2 秒的人声切片放进 16 秒窗口会被稀释约 9 dB，直接压到阈值以下。
    # 混音太安静的窗口跳过：安静段会让相对电平虚高，测的不是人声。
    try:
        from demucs_onnx.inference import SAMPLE_RATE as _SR2
        _ym, _srm = librosa.load(path, sr=_SR2, mono=True)
        _vm = np.asarray(vw, dtype=np.float64)
        if _vm.ndim == 2:
            _vm = _vm.mean(axis=0) if _vm.shape[0] <= _vm.shape[1] else _vm.mean(axis=1)
        _vm = _vm.reshape(-1)
        _win = 2 * _srm
        _hop = int(0.5 * _srm)
        _vmin = 10 ** (-35 / 20)     # vocals 绝对门槛 −35 dBFS（候选实验线，非最终）
        n = min(len(_vm), len(_ym))
        _rel = []
        for i in range(0, n - _win + 1, _hop):
            a = float(np.sqrt(np.mean(_vm[i:i + _win] ** 2)))
            b = float(np.sqrt(np.mean(_ym[i:i + _win] ** 2)))
            # 双门槛：混音和 vocals 都得有实质能量。
            # 只挡混音不够 —— 实测 Martin 276s 处 rel=−3.7 但 vocals 只有 −61.3 dBFS，
            # 安静段的比值照样接近 1，会被误当成"最强人声窗口"。
            if b < 1e-3 or a < _vmin:      # 混音 < −60 / 人声 < −35 dBFS：跳过
                continue
            _rel.append((round(i / _srm, 1),
                         round(20 * np.log10(max(a, 1e-9) / b), 1),
                         round(20 * np.log10(max(a, 1e-9)), 1)))
        if _rel:
            vals = np.array([r[1] for r in _rel])
            out["relative_level"] = {
                "window_sec": 2.0, "hop_sec": 0.5, "blocks": len(_rel),
                "max_db": round(float(vals.max()), 1),
                "p95_db": round(float(np.percentile(vals, 95)), 1),
                "median_db": round(float(np.median(vals)), 1),
                "top": sorted(_rel, key=lambda r: -r[1])[:10],
                "all": _rel,     # 全部窗口 (t, rel_db, vocals_rms)，供离线核验
            }
        else:
            # 全被门槛滤掉 —— 要么纯器乐，要么分离没分出人声。这本身是结论。
            out["relative_level"] = {
                "window_sec": 2.0, "hop_sec": 0.5, "blocks": 0,
                "note": "没有任何窗口的 vocals 超过 −40 dBFS（纯器乐或分离失败）",
            }
    except Exception as e:  # noqa: BLE001
        out["relative_level_error"] = f"{type(e).__name__}: {e}"

    # 音高轮廓(可选):分轨后两轨各跑一次 pyin。
    # 为什么不在混音上跑 —— 混音里几条旋律线叠在一起,pyin 追到谁算谁,
    # 音域会被撑宽(实测《Другая Жизнь》混音给 35.8 半音 = 3 个八度,人声不可能)。
    if want_pitch:
        try:
            from demucs_onnx.inference import SAMPLE_RATE as _SR3

            def _ratio_of(_w):
                _a2 = np.asarray(_w, dtype=np.float64)
                if _a2.ndim == 2:
                    _a2 = _a2.mean(axis=0) if _a2.shape[0] <= _a2.shape[1] else _a2.mean(axis=1)
                _a2 = _a2.reshape(-1)
                _db = 20 * np.log10(float(np.sqrt(np.mean(_a2 ** 2))) + 1e-12)
                return (10.0 ** (_db / 10.0)) / max(10.0 ** (mix_rms_db / 10.0), 1e-12)

            _vr = (out.get("vocals") or {}).get("ratio")
            _or = _ratio_of(res["other"]) if res.get("other") is not None else None

            # 哪条轨值得跑 —— 不是一刀切跳过。
            # 纯器乐（钢琴/弦乐独奏）的旋律在 other 里，人声轨占比低是正常的，
            # 之前因为 vocals<1% 就把两条轨一起跳掉，等于把纯器乐的旋律也丢了。
            _targets = []
            if _vr is not None and _vr >= 0.01:
                _targets.append("vocals")
            if _or is not None and _or >= 0.01:
                _targets.append("other")

            if not _targets:
                out["pitch"] = {
                    "status": "skipped_low_energy",
                    "vocals_ratio": _vr,
                    "other_ratio": _or,
                    "note": "两轨能量都太低，测不出稳定旋律；不等于原曲没有内容",
                }
            else:
                _pit = {}
                for _nm in _targets:
                    _w = res.get(_nm)
                    if _w is None:
                        continue
                    _a = np.asarray(_w, dtype=np.float64)
                    if _a.ndim == 2:
                        _a = _a.mean(axis=0) if _a.shape[0] <= _a.shape[1] else _a.mean(axis=1)
                    _pit[_nm] = pitch_summary(_a.reshape(-1).astype(np.float32), _SR3)
                out["pitch"] = _pit
        except Exception as e:  # noqa: BLE001
            out["pitch_error"] = f"{type(e).__name__}: {e}"

    try:
        os.makedirs(root, exist_ok=True)
        with open(cpath, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False)
    except Exception:
        pass
    return out


def _instrumental_ranges(mono, sr, block_sec=4.0, min_sec=8.0, delta_db=35.0,
                         merge_gap=4.0):
    """切出「人声轨几乎没能量」的区间 —— 段落角色的证据,不是段落名。

    判据用**绝对能量 + 自适应阈值**,不用「人声占伴奏比」:
    占比会被伴奏强度稀释,鼓一响人声占比就掉,会被误读成「人声消失」。
    实测《拉古拉古》112s:人声 -31.8 dBFS 仍在唱,但鼓涨到 -18.9,
    占比只剩 3.8%,按占比会误标成器乐段。

    阈值不写死:先去掉明显静音块,取人声电平中位数再下沉 delta_db。
    不同曲目的增益水平差得很大(母带压限 vs 手机录音),固定阈值会在
    响度低的曲目上失效。中位数-35 对《拉古拉古》正好落在 -60 附近,
    与实测吻合。
    """
    win = max(1, int(block_sec * sr))
    n = len(mono)
    dbs = []
    for i in range(0, n, win):
        seg = mono[i:i + win]
        if seg.size == 0:
            break
        db = 20.0 * np.log10(float(np.sqrt(np.mean(seg ** 2))) + 1e-12)
        dbs.append((i / sr, min((i + win) / sr, n / sr), db))
    if not dbs:
        return []

    hi = max(b for _, _, b in dbs)
    valid = [b for _, _, b in dbs if b > hi - 50.0]   # 去掉明显静音块
    med = float(np.median(valid)) if valid else -60.0
    thr = max(min(med - delta_db, -40.0), -85.0)       # 钳在 [-85, -40]

    spans, cur, cur_end = [], None, None
    for a, b, db in dbs:
        if db < thr:
            if cur is None:
                cur = a
            cur_end = b
        elif cur is not None:
            spans.append([cur, cur_end])
            cur = None
    if cur is not None:
        spans.append([cur, cur_end])

    merged = []
    for a, b in spans:
        if merged and a - merged[-1][1] <= merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [{"start_sec": round(a, 1), "end_sec": round(b, 1)}
            for a, b in merged if b - a >= min_sec]


# ---------------------------------------------------------------------------
# 歌词听写(可选组件)
# ---------------------------------------------------------------------------
def analyze_lyrics(path, *, model="base", cache_dir=None):
    """歌词听写:4s 分轨取人声轨 + faster-whisper,只输出「可信段」。

    **宁缺勿滥**:宁可少给一句,不把幻觉写成歌词。所以分三级:
      可信    avg_logprob >= -0.8 且 compression_ratio < 1.8
      存疑    落在中间带(不给正文,只算数量)
      疑幻觉  avg_logprob < -0.85 或 compression_ratio >= 2.4 或 同句连续 >=3 次

    **不用 no_speech_prob**:歌声该值普遍 0.5–0.9(它测的是「像不像说话」),
    拿它分级会把正确段落全杀掉 —— 与 Silero VAD 同一个病根。

    2026-10-21 实测依据(4s 分轨 vocals + faster-whisper int8,CPU):
      普通话样本 base+抑制参数: 可信段正好是副歌,疑幻觉段正好是循环/英语瞎编;
      梵语样本 base+抑制参数: 100% 判疑幻觉(模型在编,正确拒绝);
      不抑制重复(temperature 默认回退)时小模型会「自信地编」,分级失效 ——
      所以 repetition_penalty / no_repeat_ngram_size / temperature=0 是必需品。

    返回 dict:
      available=True 时: model / segments / trusted / stats
      available=False 时: reason (组件缺失、分离失败等)
    """
    import re as _re
    import time as _time

    _data = os.environ.get("DSH_AUDIO_DATA_DIR")
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _model_root = os.path.join(_data, "whisper") if _data else os.path.join(_script_dir, "whisper")

    try:
        from faster_whisper import WhisperModel
    except Exception as e:  # noqa: BLE001
        return {"available": False,
                "reason": f"未安装 faster-whisper(pip install faster-whisper): {e}"}

    try:
        from demucs_onnx.inference import separate, SAMPLE_RATE
    except Exception as e:  # noqa: BLE001
        return {"available": False,
                "reason": f"分离组件未安装(demucs-onnx),取不到人声轨: {e}"}

    model_dir = os.environ.get("DSH_AUDIO_MODEL_DIR") or (
        os.path.join(_data, "models", "stems") if _data
        else os.path.join(_script_dir, "models", "stems"))
    local_stem = os.path.join(model_dir, "htdemucs_fp16weights.onnx")
    if os.path.exists(local_stem):
        try:
            from demucs_onnx import inference as _inf
            _inf.download_single_model = lambda *a, **k: local_stem
        except Exception:
            pass

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    if _data:
        os.environ.setdefault("HF_HOME", os.path.join(_data, "hf_home"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # hf-mirror 不支持 xet CAS

    t0 = _time.time()

    # 人声轨缓存:第一次分离后存 flac,之后同一首歌直接读。
    # 分离是最大瓶颈(4 分钟歌约 2 分钟),缓存后二次分析能秒出。
    import hashlib as _hl
    _cache_audio = os.path.join(
        os.environ.get("DSH_AUDIO_CACHE_DIR")
        or (os.path.join(_data, "cache") if _data else os.path.join(_script_dir, "cache")),
        "audio")
    _st = os.stat(path)
    _key = _hl.sha1(
        f"{os.path.abspath(path)}|{_st.st_size}|{int(_st.st_mtime)}|htdemucs".encode()
    ).hexdigest()[:16]
    _wpath = os.path.join(_cache_audio, f"{_key}_vocals.mp3")
    _wpath_old = os.path.join(_cache_audio, f"{_key}_vocals.flac")
    arr = None
    for _wp in (_wpath, _wpath_old):
        if not os.path.exists(_wp):
            continue
        try:
            import soundfile as _sf
            arr, _sr = _sf.read(_wp, dtype="float32")
            arr = arr.T if arr.ndim == 2 else arr
            print("[听写] 命中人声缓存,跳过分离", file=sys.stderr)
            break
        except Exception:
            arr = None
    if arr is None:
        try:
            print("[听写] 正在分离人声轨(htdemucs)...", file=sys.stderr)
            res = separate(path, None, model="htdemucs", precision="fp16weights",
                           cache_dir=model_dir, progress=False)
        except Exception as e:  # noqa: BLE001
            return {"available": False, "reason": f"人声分离失败: {e}"}
        arr = np.asarray(res.get("vocals"), dtype=np.float32)
        try:
            import soundfile as _sf
            os.makedirs(_cache_audio, exist_ok=True)
            data = arr.T if arr.ndim == 2 else arr
            try:
                _sf.write(_wpath, data, SAMPLE_RATE, format="MP3",
                          subtype="MPEG_LAYER_III")
            except Exception:
                _sf.write(_wpath_old, data, SAMPLE_RATE)   # mp3 不支持时退回 flac
        except Exception:
            pass

    if arr.size == 0:
        return {"available": False, "reason": "没有分离出人声轨"}
    mono = arr.mean(axis=0) if arr.ndim == 2 else arr
    v_db = 20.0 * np.log10(float(np.sqrt(np.mean(mono ** 2))) + 1e-12)
    # 纯器乐保护：绝对能量低 **或** 相对混音低都算「人声几乎没有」。
    # 只卡绝对 -60 dBFS 不够 —— 实测人声占比 0.08% 的纯配乐，vocals 仍在 -31 dBFS 左右，
    # 会漏进 Whisper，然后被「听写」出一句幻听歌词（用户 2026-10-23 反馈的真实案例）。
    _mix_db = None
    try:
        import soundfile as _sf2
        _info = _sf2.info(path)
        _mix_rms = float(getattr(_info, "frames", 0) or 0)
        # 用整首混音 RMS 估一个参照（不必精确，只用来判断「人声是不是几乎不存在」）
        _y_mix, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=30.0)
        _mix_db = 20.0 * np.log10(float(np.sqrt(np.mean(_y_mix ** 2))) + 1e-12)
    except Exception:
        _mix_db = None

    _rel_db = (v_db - _mix_db) if _mix_db is not None else None
    _ev = {"vocals_dbfs": round(v_db, 1),
           "mix_dbfs": None if _mix_db is None else round(_mix_db, 1),
           "vocals_minus_mix_db": None if _rel_db is None else round(_rel_db, 1)}
    _inst_all = [{"start_sec": 0.0, "end_sec": round(len(mono) / SAMPLE_RATE, 1)}]

    # 三级门控（阈值是临时工程参数，未用人标样本校准）
    #   A ≤ -20 dB        证据不足，不跑 ASR
    #   B -20 ~ -15 dB    灰区，不跑也不给歌词
    #   C ≥ -15 dB        允许跑，但结果仍要过可信段筛选
    if v_db < -60 or (_rel_db is not None and _rel_db <= -20):
        _why = ("人声轨几乎没有能量" if v_db < -60
                else f"人声轨相对混音低 {_rel_db:.0f} dB，证据不足以支持歌词")
        return {"available": True, "model": model, "segments": [], "trusted": [],
                "draft": [], "instrumental": _inst_all,
                "transcription_status": "skipped_low_vocal_evidence",
                "vocal_evidence": _ev,
                "stats": {**_ev, "segments": 0, "trusted": 0, "doubt": 0,
                          "hallucination": 0, "trusted_ratio": 0.0,
                          "note": f"{_why}；未执行听写"}}

    if _rel_db is not None and _rel_db < -15:
        return {"available": True, "model": model, "segments": [], "trusted": [],
                "draft": [], "instrumental": _inst_all,
                "transcription_status": "withheld_ambiguous_vocal_evidence",
                "vocal_evidence": _ev,
                "stats": {**_ev, "segments": 0, "trusted": 0, "doubt": 0,
                          "hallucination": 0, "trusted_ratio": 0.0,
                          "note": f"人声轨相对混音低 {_rel_db:.0f} dB，证据处于灰区；"
                                  "未执行听写，也不给歌词。这只说明证据不足，"
                                  "不代表已证明没有人声"}}

    # 无人声区间:用绝对能量判(实测结论),占比法会误报
    inst_ranges = _instrumental_ranges(mono, SAMPLE_RATE)

    if SAMPLE_RATE != 16000:
        mono = librosa.resample(mono, orig_sr=SAMPLE_RATE, target_sr=16000)

    local_model = os.path.join(_model_root, model)
    target = local_model if os.path.exists(os.path.join(local_model, "model.bin")) else model
    try:
        print(f"[听写] 加载 Whisper {model}(首次会下载模型)...", file=sys.stderr)
        wm = WhisperModel(target, device="cpu", compute_type="int8",
                          download_root=_model_root)
        segs, info = wm.transcribe(mono, language="zh", beam_size=5,
                                   vad_filter=False,
                                   condition_on_previous_text=False,
                                   repetition_penalty=1.15,
                                   no_repeat_ngram_size=3,
                                   temperature=0.0)
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"Whisper 转写失败: {e}"}

    rows = []
    prev, run = None, 0
    for s in segs:
        txt = (s.text or "").strip()
        norm = _re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", txt)
        run = run + 1 if (norm and norm == prev) else 1
        prev = norm
        lp = float(s.avg_logprob)
        cr = float(s.compression_ratio)
        if lp < -0.85:
            label, why = "疑幻觉", f"logprob {lp:.2f}"
        elif cr >= 2.4:
            label, why = "疑幻觉", f"压缩比 {cr:.2f}"
        elif run >= 3:
            label, why = "疑幻觉", f"连续重复 {run} 次"
        elif lp < -0.8 or cr >= 1.8:
            label, why = "存疑", f"中间带 lp={lp:.2f} cr={cr:.2f}"
        else:
            label, why = "可信", ""
        rows.append({
            "start_sec": round(float(s.start), 2),
            "end_sec": round(float(s.end), 2),
            # 疑幻觉段不给正文:它编得越顺,越容易被当成歌词转述出去
            "text": txt if label != "疑幻觉" else None,
            "label": label,
            "why": why,
            "avg_logprob": round(lp, 3),
            "compression_ratio": round(cr, 3),
        })

    trusted = [r for r in rows if r["label"] == "可信"]
    total = sum(r["end_sec"] - r["start_sec"] for r in rows) or 1e-9
    trusted_dur = sum(r["end_sec"] - r["start_sec"] for r in trusted)
    stats = {
        "segments": len(rows),
        "trusted": len(trusted),
        "doubt": sum(1 for r in rows if r["label"] == "存疑"),
        "hallucination": sum(1 for r in rows if r["label"] == "疑幻觉"),
        "trusted_ratio": round(trusted_dur / total, 2),
        "vocals_dbfs": round(v_db, 1),
        "elapsed_sec": round(_time.time() - t0, 1),
    }
    _draft_rows = [r for r in rows if r["label"] == "存疑" and r["text"]]
    stats["instrumental"] = len(inst_ranges)
    stats["draft_withheld"] = len(_draft_rows)
    # 2026-10-23 撤掉「降级稿」：未通过筛选的识别文本不进正式输出。
    # 理由（用户决策 + 外部建议）：把幻听文字连同警告一起交给下游模型，
    # 模型仍会优先采用具体文字；真正的隔离是不让它出现在输入里。
    _status = ("completed_with_accepted_segments" if trusted
               else "completed_no_accepted_segments")
    # segments 里「存疑 / 疑幻觉」两类的正文一并清空，只留标签和时间，
    # 否则它会从另一个字段泄漏出去。
    _safe_rows = [({"start_sec": r["start_sec"], "end_sec": r["end_sec"],
                     "label": r["label"]}
                    if r["label"] != "可信" else r) for r in rows]
    return {"available": True, "model": model, "segments": _safe_rows,
            "trusted": trusted, "draft": [],
            "instrumental": inst_ranges, "stats": stats,
            "transcription_status": _status,
            "vocal_evidence": _ev}


def quick_transcribe(y, sr, model="tiny", max_lines=4):
    """预览用:对切出的片段做「分离人声 → 转写」,只给前几句。

    为什么不对整首做:整首分离约 2 分钟,预览要快。
    但**必须分离**——直接转混合音频时伴奏会盖住歌声,几乎听不出词。
    40 秒片段分离约 15 秒,加上 tiny 转写,总代价可接受。
    """
    tmp_name = None
    try:
        import tempfile

        import soundfile as _sf
        from demucs_onnx.inference import SAMPLE_RATE as DEMUCS_SR
        from demucs_onnx.inference import separate

        _data = os.environ.get("DSH_AUDIO_DATA_DIR")
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.environ.get("DSH_AUDIO_MODEL_DIR") or (
            os.path.join(_data, "models", "stems") if _data
            else os.path.join(_script_dir, "models", "stems"))
        _local = os.path.join(model_dir, "htdemucs_fp16weights.onnx")
        if os.path.exists(_local):
            from demucs_onnx import inference as _inf
            _inf.download_single_model = lambda *a, **k: _local
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        if _data:
            os.environ.setdefault("HF_HOME", os.path.join(_data, "hf_home"))
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

        arr = y if sr == DEMUCS_SR else librosa.resample(
            y, orig_sr=sr, target_sr=DEMUCS_SR)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        tmp_name = tmp.name
        _sf.write(tmp_name, arr, DEMUCS_SR)

        res = separate(tmp_name, None, model="htdemucs", precision="fp16weights",
                       cache_dir=model_dir, progress=False)
        vocals = np.asarray(res.get("vocals"), dtype=np.float32)
        if vocals.size == 0:
            return []
        mono = vocals.mean(axis=0) if vocals.ndim == 2 else vocals

        from faster_whisper import WhisperModel
        _root = (os.path.join(_data, "whisper") if _data
                 else os.path.join(_script_dir, "whisper"))
        lw = os.path.join(_root, model)
        target = lw if os.path.exists(os.path.join(lw, "model.bin")) else model
        wm = WhisperModel(target, device="cpu", compute_type="int8",
                          download_root=_root)
        if DEMUCS_SR != 16000:
            mono = librosa.resample(mono, orig_sr=DEMUCS_SR, target_sr=16000)
        segs, _ = wm.transcribe(mono, language="zh", beam_size=5,
                                vad_filter=False,
                                condition_on_previous_text=False,
                                repetition_penalty=1.15, no_repeat_ngram_size=3,
                                temperature=0.0)
        out = []
        for s in segs:
            t = (s.text or "").strip()
            if t:
                out.append(t)
            if len(out) >= max_lines:
                break
        return out
    except Exception:
        return []
    finally:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except Exception:
                pass


def pick_preview_segment(y, sr, block_sec=4.0, min_sec=8.0, max_sec=64.0):
    """用数据选预览段:找「又响又密」的连续块,长度由歌本身决定。

    阈值来自整首的中位数(不预设秒数、不拍阈值):
      - 能量  >= 全曲中位数
      - 起音  >= 全曲中位数
    两者同时满足的连续块视为副歌候选,取最长的一段。

    回退:没有候选时用最响的单块 + 邻块。
    返回 (start_sec, length_sec, how)。
    """
    hop = 512
    fps = sr / hop
    try:
        rms = librosa.feature.rms(y=y, hop_length=hop, frame_length=2048)[0]
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        onsets = librosa.onset.onset_detect(onset_envelope=env, sr=sr,
                                            hop_length=hop, units="time")
    except Exception:
        return 0.0, min_sec, "回退:开头"

    dur = len(y) / sr
    n = int(dur / block_sec)
    if n < 2:
        return 0.0, min(dur, min_sec), "回退:太短"

    rms_db = np.empty(n)
    on_rate = np.empty(n)
    for i in range(n):
        t0, t1 = i * block_sec, (i + 1) * block_sec
        seg = rms[int(t0 * fps):int(t1 * fps)]
        rms_db[i] = 20.0 * np.log10(float(np.sqrt(np.mean(seg ** 2))) + 1e-12)
        on_rate[i] = float(np.sum((onsets >= t0) & (onsets < t1)) / block_sec)

    good = (rms_db >= np.median(rms_db)) & (on_rate >= np.median(on_rate))
    if not good.any():
        i = int(np.argmax(rms_db))
        start = max(0.0, (i - 1) * block_sec)
        return start, min(max_sec, min_sec), "回退:最响块"

    best_len = best_start = 0
    i = 0
    while i < n:
        if not good[i]:
            i += 1
            continue
        j = i
        while j < n and good[j]:
            j += 1
        if j - i > best_len:
            best_len, best_start = j - i, i
        i = j

    length = best_len * block_sec
    start = best_start * block_sec
    if length < min_sec:
        length = min(min_sec, dur)
        start = max(0.0, min(start, dur - length))
    elif length > max_sec:
        mid = start + length / 2
        start = max(0.0, mid - max_sec / 2)
        length = max_sec
    return start, length, f"数据选段:{best_len}块"


def bpm_layers(tempo):
    """给出两个倍频层级:检测值本身 + 最近的倍频(×2 或 ÷2)。

    **不判断哪个是「真速」**——实测两个方向都会错:
      November Rain 官方约 80,检测 156.6(约两倍)
      《月之暗面》网易云 69,检测 139.7(约两倍)
      用户反馈的摇滚报 80、听感 160(约一半)
    所以只把两层摆出来,由听的人按自己的数拍方式判断。
    """
    if tempo <= 0:
        return None
    other = tempo / 2 if tempo >= 90 else tempo * 2
    if other < 40:
        other = tempo * 2
    if other > 240:
        other = tempo / 2
    return round(tempo, 1), round(other, 1)


def pitch_summary(y, sr, fmin=120.0, fmax=1200.0):
    """pyin 基频追踪 → 音域 / 有声占比。只在 --pitch 时跑。

    **必须先高通 150 Hz**：混音里贝斯能量最强，pyin 会追到底鼓/贝斯的基频，不是旋律。
    实测 Queen 60–90s：不滤波给 C2–G2（7.1 半音，那是贝斯），高通后给 C3–A3（8.9 半音）。

    ⚠ 它不知道这条轨迹是谁的 —— 混音里可能是人声，也可能是钢琴、弦乐或合成器。
       调用方必须标注来源，否则等于骗人。
    """
    try:
        from scipy.signal import butter, sosfilt
        sos = butter(4, 150, btype="highpass", fs=sr, output="sos")
        yh = sosfilt(sos, y).astype(np.float32)
        f0, voiced, vp = librosa.pyin(yh, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048)
        v = f0[voiced & ~np.isnan(f0)]
        if len(v) == 0:
            return {"available": False, "reason": "没有稳定的基频"}
        semi = 69 + 12 * np.log2(v / 440.0)
        lo, hi = float(np.percentile(semi, 10)), float(np.percentile(semi, 90))
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        def nm(s):
            m = int(round(s))
            return f"{names[m % 12]}{m // 12 - 1}"

        # 趋势：对半音序列做最小二乘斜率，乘帧数得到整段总变化量。
        # 阈值 ±2 半音 —— 小于这个数的漂移听感上就是「基本持平」。
        _total = 0.0
        if len(semi) > 4:
            _idx = np.arange(len(semi), dtype=float)
            _total = float(np.polyfit(_idx, semi, 1)[0] * len(semi))
        trend = "上行" if _total > 2 else ("下行" if _total < -2 else "基本持平")

        # 连续段（带起止索引）—— 算大跳和归一化都要用
        _segs, _c, _st = [], 0, 0
        for _i, _b in enumerate(voiced):
            if _b:
                if _c == 0:
                    _st = _i
                _c += 1
            elif _c:
                _segs.append((_st, _st + _c))
                _c = 0
        if _c:
            _segs.append((_st, _st + _c))
        _hop_sec = 512.0 / sr
        longest_sec = (round(max((e - s) for s, e in _segs) * _hop_sec, 1)
                       if _segs else 0.0)

        # 段内「大跳」统计不出来 —— 不是没有大跳，是算法不许有：
        # pyin 的 max_transition_rate 默认 35.92 八度/秒，按 44100/512 就是每帧最多
        # 5.0 半音，所以「相邻帧差 ≥7 半音」在段内结构上不可能出现。
        # 结论：正式输出不给「大跳」，等以后有不受帧间约束的音符级算法再设计。
        # 下面这个是**段间**音高差（跨过无声区间），只作调试：可能是换音，也可能是断裂。
        _gaps = 0
        for _k in range(1, len(_segs)):
            _ei = _segs[_k - 1][1] - 1
            _si = _segs[_k][0]
            if _ei < len(f0) and _si < len(f0):
                _fa, _fb = f0[_ei], f0[_si]
                if (not np.isnan(_fa) and not np.isnan(_fb) and _fa > 0 and _fb > 0):
                    if abs(12 * np.log2(_fb / _fa)) >= 7:
                        _gaps += 1

        return {
            "available": True,
            "status": "computed_unverified",
            "lo": nm(lo), "hi": nm(hi),
            "span": round(hi - lo, 1),
            "voiced_ratio": round(float(voiced.mean()), 3),
            "trend": trend,
            "inter_seg_gaps": _gaps,
            "longest_sec": longest_sec,
            "vprob_mean": round(float(np.mean(vp)), 3),
            "vprob_median": round(float(np.median(vp)), 3),
            "vprob_p90": round(float(np.percentile(vp, 90)), 3),
            "frames": int(len(v)),
        }
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}


def emit_pitch(stems, y, sr):
    """输出音高轮廓。默认输出与 --detail 两条路径都要调它。"""
    print()
    _pit = (stems or {}).get("pitch") or {}

    # 两轨都没能量 → 直接跳过，不给数字
    if _pit.get("status") == "skipped_low_energy":
        _vr = _pit.get("vocals_ratio") or 0.0
        _or = _pit.get("other_ratio") or 0.0
        print(f"[音高] 未计算 —— 两条轨能量都太低（vocals {_vr * 100:.2f}% / other {_or * 100:.2f}%）。")
        print("       没有稳定的旋律线可追。注意：这不代表原曲没有内容。")
        return

    if _pit:
        for _nm in ("vocals", "other"):
            _ps = _pit.get(_nm)
            if not _ps:
                continue
            _zh = {"vocals": "人声轨", "other": "器乐轨"}.get(_nm, _nm)
            if _ps.get("available"):
                print(f"[音高] {_zh}：{_ps['lo']}–{_ps['hi']}（{_ps['span']} 个半音）"
                      f" | 有声 {_ps['voiced_ratio'] * 100:.0f}%"
                      f" | 趋势 {_ps.get('trend', '?')}"
                      f" | 最长连续 {_ps.get('longest_sec', 0):.2f} 秒")
            else:
                print(f"[音高] {_zh}：测不出稳定基频（{_ps.get('reason', '未知')}）")
        _vr = ((stems or {}).get("vocals") or {}).get("ratio")
        if "vocals" not in _pit and _vr is not None:
            print(f"       人声轨只占 {_vr * 100:.2f}%，没有可用的人声旋律 —— "
                  "上面只有器乐轨（纯器乐的音高来自这里，不是人唱的）。")
        elif _vr is not None and _vr < 0.10:
            print(f"       ⚠ 人声轨只占 {_vr * 100:.1f}% —— 这一轨里大半是泄漏，"
                  "它的音域不能当「人唱的音域」用。")
        print("       ⚠ 状态：已计算、未验证（computed_unverified）。这些是 pyin 的候选音高，"
              "不保证来自真人演唱 —— 钢琴/弦乐的基频比人声更稳，指标好看不等于人声。")
    else:
        _ps = pitch_summary(y, sr)
        if _ps and _ps.get("available"):
            print(f"[音高] 音域 {_ps['lo']}–{_ps['hi']}（{_ps['span']} 个半音）"
                  f" | 有声 {_ps['voiced_ratio'] * 100:.0f}% | 来源：混音（未分轨）")
            print("       ⚠ 混音里几条旋律线叠在一起，音域会被撑宽 —— 要分清得加 --stems。")
        else:
            print(f"[音高] 测不出稳定基频（{(_ps or {}).get('reason', '未知')}）")


def _cross_links(ab, es, inst_ranges, med_db):
    """把「无人声区间」和「能量块」对起来。

    单看任何一项都只剩事实：「2:12–2:40 没人唱」+「这段能量 −15 dB」。
    对起来才说得出人话：是伴奏顶着的纯器乐段，还是整段留白。
    """
    if not inst_ranges or not ab or len(es) < 3:
        return []
    out = []
    for r in inst_ranges:
        a, b = r["start_sec"], r["end_sec"]
        seg = [e for blk, e in zip(ab, es)
               if blk["start_sec"] < b and blk["end_sec"] > a]
        if not seg:
            continue
        drop = med_db - float(np.mean(seg))
        span = f"{fmt_time(a)}–{fmt_time(b)}"
        if drop <= 3:
            out.append(f"{span} 没人唱，能量却没降（跟中段持平）—— 纯器乐段，伴奏在顶。")
        elif drop >= 12:
            out.append(f"{span} 没人唱，能量也塌了 {drop:.0f} dB —— 这一段是留白，不是伴奏在顶。")
    return out


def _find_hook(ab, es, blocks, inst_ranges, asr):
    """找「钩子」——几个测量同时到达峰值的位置。

    不叫"副歌"：这只是"能量 / 起音密度 /（有歌词时）字数密度同时最高"，
    位置不等于段落角色。没人唱的位置会被扣分。

    注意 blocks 的元素是元组 (start_sec, end_sec, energy)，不是字典。
    """
    if es is None or not blocks or len(es) < 4:
        return None
    n = min(len(es), len(blocks))
    e_arr = np.asarray(es[:n], dtype=float)

    act = np.full(n, np.nan)
    for i, blk in enumerate(blocks[:n]):
        t0, t1 = blk[0], blk[1]
        vals = [a.get("onsets", 0.0) for a in (ab or [])
                if a.get("start_sec", 0) < t1 and a.get("end_sec", 0) > t0]
        if vals:
            act[i] = float(np.mean(vals))

    lyr = np.zeros(n)
    if asr and asr.get("trusted"):
        for r in asr["trusted"]:
            for i, blk in enumerate(blocks[:n]):
                if blk[0] < r["end_sec"] and blk[1] > r["start_sec"]:
                    lyr[i] += len(str(r.get("text", "")))
    has_lyr = bool(lyr.any())

    def _z(a):
        m, s = float(np.nanmean(a)), float(np.nanstd(a))
        if not np.isfinite(m) or s < 1e-9:
            return np.zeros(n)
        return (a - m) / s

    score = _z(e_arr) * 0.5 + (_z(act) * 0.3 if not np.all(np.isnan(act)) else 0.0)
    if has_lyr:
        score = score + _z(lyr) * 0.2

    # 没人唱的块扣分：钩子通常在唱的地方
    if inst_ranges:
        for i, blk in enumerate(blocks[:n]):
            for r in inst_ranges:
                if blk[0] < r["end_sec"] and blk[1] > r["start_sec"]:
                    score[i] -= 0.6
                    break

    order = np.argsort(-score)
    picks = [int(i) for i in order[:2] if score[i] > 0.3]
    if not picks:
        return None

    _a_ok = act[~np.isnan(act)]
    _a_p75 = float(np.nanpercentile(_a_ok, 75)) if len(_a_ok) else float("nan")

    def _desc(i):
        parts = []
        if e_arr[i] >= float(np.nanpercentile(e_arr, 75)):
            parts.append("能量在峰值区")
        if not np.isnan(act[i]) and np.isfinite(_a_p75) and act[i] >= _a_p75:
            parts.append("起音最密")
        if has_lyr and lyr[i] >= float(np.percentile(lyr, 75)):
            parts.append("歌词最密")
        if inst_ranges and any(blocks[i][0] < r["end_sec"]
                               and blocks[i][1] > r["start_sec"] for r in inst_ranges):
            parts.append("这段没人唱")
        return " / ".join(parts) if parts else "综合分最高"

    lines = []
    b0 = blocks[picks[0]]
    lines.append(f"最可能是钩子的位置：{fmt_time(b0[0])}–{fmt_time(b0[1])}"
                 f"（{_desc(picks[0])}）")
    if len(picks) > 1:
        b1 = blocks[picks[1]]
        lines.append(f"次选：{fmt_time(b1[0])}–{fmt_time(b1[1])}"
                     f"（{_desc(picks[1])}）")
    lines.append("说明：这是几个测量同时到峰值的位置，不是已经确认的副歌——"
                 "段落角色要听过才知道。")
    return lines


def _timeline(ab, es, blocks, inst_ranges, block_db=None):
    """把能量 / 起音密度 / 人声状态串成事件时间线。

    三条规矩（按 GPT 的建议改的）：
      1. **不做均匀采样** —— 那会让曲式看起来「每分钟一件事」，是假的
      2. 相邻同向的变化**合并成一个过程**，并记下它持续了多久
      3. 事件按「幅度 × 持续」筛，不是按时间均分
    只报发生了什么，不命名段落角色。
    """
    if es is None or not blocks or len(es) < 4:
        return []
    n = min(len(es), len(blocks))
    e = np.asarray(es[:n], dtype=float)
    t0 = [blocks[i][0] for i in range(n)]
    t1 = [blocks[i][1] for i in range(n)]

    def _has_voc(i):
        if not inst_ranges:
            return None
        for r in inst_ranges:
            if t0[i] < r["end_sec"] and t1[i] > r["start_sec"]:
                return False
        return True

    voc = [_has_voc(i) for i in range(n)]
    act = np.full(n, np.nan)
    for i in range(n):
        vals = [a.get("onsets", 0.0) for a in (ab or [])
                if a.get("start_sec", 0) < t1[i] and a.get("end_sec", 0) > t0[i]]
        if vals:
            act[i] = float(np.mean(vals))

    # ---- 第一层：找变化候选 ----
    cand = []
    for i in range(1, n):
        de = e[i] - e[i - 1]
        da = (act[i] - act[i - 1]) if (not np.isnan(act[i]) and not np.isnan(act[i - 1])) else 0.0
        vch = (voc[i] != voc[i - 1]) if (voc[i] is not None and voc[i - 1] is not None) else False
        if abs(de) >= 1.5 or abs(da) >= 0.8 or vch:
            cand.append({"i": i, "de": de, "da": da, "vch": vch})

    # ---- 第二层：相邻同向的合并成一个过程 ----
    groups = []
    for c in cand:
        if groups:
            g = groups[-1]
            near = c["i"] - g["end"] <= 2
            same_dir = (np.sign(c["de"]) == np.sign(g["de"]) or
                        np.sign(c["da"]) == np.sign(g["da"]))
            if near and same_dir and not c["vch"]:
                g["end"] = c["i"]
                g["de"] += c["de"]
                g["da"] += c["da"]
                continue
        groups.append({"start": c["i"], "end": c["i"],
                       "de": c["de"], "da": c["da"], "vch": c["vch"]})

    # ---- 输出 ----
    events = []
    for g in groups:
        i0, i1 = g["start"], g["end"]
        span = t1[i1] - t0[i0]
        # 归纳在前，绝对值在后 —— 两组 A/B 测试里各赢一半，合起来信息最全
        _e0 = e[min(i0 - 1, n - 1)]
        _e1 = e[min(i1, n - 1)]
        _a0 = act[min(i0 - 1, n - 1)] if not np.isnan(act[min(i0 - 1, n - 1)]) else None
        _a1 = act[min(i1, n - 1)] if not np.isnan(act[min(i1, n - 1)]) else None
        _vals = f"能量 {_e0:.0f}→{_e1:.0f} dB"
        if _a0 is not None and _a1 is not None:
            _vals += f"、起音 {_a0:.0f}→{_a1:.0f}"
        parts = []
        if abs(g["de"]) >= 1.5:
            parts.append(f"能量{'上升' if g['de'] > 0 else '回落'} {abs(g['de']):.0f} dB")
        if abs(g["da"]) >= 0.8:
            parts.append(f"起音{'变密' if g['da'] > 0 else '变疏'}")
        if g["vch"]:
            parts.append("人声进入" if voc[i0] else "人声退出")
        if not parts:
            continue
        if g["de"] > 2 and g["da"] > 0.8:
            parts = [f"编曲加码（{_vals}）"]
        elif g["de"] < -2 and g["da"] < -0.8:
            parts = [f"动态收缩（{_vals}）"]
        elif g["de"] > 2 and g["da"] < -0.8:
            # 变响但瞬态更少 —— 不是靠「加东西」堆上去的
            parts = [f"能量升、起音未增（{_vals}）"]
        elif g["de"] < -2 and g["da"] > 0.8:
            parts = [f"能量降、起音反增（{_vals}）"]
        # 哪条轨变了 —— 分轨逐块能量（有 --stems 时才有）
        ZH = {"vocals": "人声", "drums": "鼓", "bass": "贝斯", "other": "伴奏"}
        if block_db:
            _best = None
            # 只在混音能量覆盖的时间范围内引用分轨 —— demucs 输出可能比输入长（补零），
            # 补零块的分轨值是假数据，会让尾部事件出现「整体塌了但伴奏变响」的假象。
            _mix_end = t1[n - 1]
            _t_a = min(t0[i0], _mix_end - 0.1)
            _t_b = min(t1[i1], _mix_end - 0.1)
            for _nm, _vals in block_db.items():
                def _db_at(_t):
                    for _bx in _vals:
                        if _bx["start_sec"] <= _t < _bx["end_sec"]:
                            # 块必须完全落在混音时间范围内，否则是 demucs 在尾部编出来的
                            if _bx["end_sec"] > _mix_end + 1.0:
                                return None
                            return _bx["db"]
                    return None
                _da, _dbb = _db_at(_t_a), _db_at(_t_b)
                if _da is not None and _dbb is not None:
                    _d = _dbb - _da
                    if _best is None or abs(_d) > abs(_best[1]):
                        _best = (_nm, _d)
            if _best and abs(_best[1]) >= 3:
                # 人声刚进出时不再报「人声变响/变轻」——那是边界上的残余，会自相矛盾
                if not (g["vch"] and _best[0] == "vocals"):
                    parts.append(f"{ZH.get(_best[0], _best[0])}"
                                 f"{'变响' if _best[1] > 0 else '变轻'} {abs(_best[1]):.0f} dB")
        tail = f"，该状态持续 {span:.0f} 秒" if span >= 32 else ""
        events.append((t0[i0], "；".join(parts) + tail,
                       abs(g["de"]) / 3.0 + abs(g["da"]) * 2 + span / 60.0))

    # 开头状态
    head = []
    if e[0] <= float(np.percentile(e, 25)):
        head.append("能量偏低")
    if voc[0] is False:
        head.append("人声未进")
    if not np.isnan(act[0]) and act[0] <= float(np.nanpercentile(act, 25)):
        head.append("起音稀疏")
    if head:
        events.insert(0, (0.0, "开场：" + "、".join(head), 1.0))

    # 按重要性筛（不是按时间均分），最多 8 条
    events.sort(key=lambda x: -x[2])
    keep = sorted(events[:8], key=lambda x: x[0])
    return [(ts, txt) for ts, txt, _ in keep]


def verdict_lines(d):
    """
    把测量结果合成一段人能直接读的判断。

    三条规矩:
      - 正文只说结论,数字集中放到 [依据] —— 别让人对着一屏块发呆
      - 说不出来的就直说"听不出来",不用形容词去补
      - 不报流派:流派是听出来的,不是测出来的
    """
    dur, tempo = d["dur"], d["tempo"]
    kick = d.get("kick_bpm", float("nan"))
    key, key_r = d["key"], d["key_r"]
    rms, peak, std = d["rms_db"], d["peak_db"], d["std_db"]
    es = d["es"]
    blocks = d["blocks"]
    ab = d.get("activity_blocks") or []
    section = d["section"]
    chords = d["chords"]
    fit = d["fit"]

    lines = []

    # 1) 速度:Beat This 给的是音乐学拍,直接报;librosa 回退时给两层并列
    #    人耳数的拍和检测器数的拍常常差一倍 —— 推荐错了比不推荐更糟。
    _lay = bpm_layers(tempo)
    if not _lay:
        speed = "速度测不出来"
    elif d.get("bpm_source") == "beat_this":
        speed = f"速度约 {tempo:.0f} 拍"
    elif d["bpm_suspect"]:
        speed = (f"拍速是两层:{_lay[0]:.0f} 和 {_lay[1]:.0f},"
                 "低频层和瞬态层各说各的,别只信一个")
    else:
        speed = (f"拍速是两层:{_lay[0]:.0f} 和 {_lay[1]:.0f},"
                 "两个都可能,取决于你怎么数拍")

    # 2) 调性:分"贴合"与"只是候选",和 [可信] 保持一致
    if fit["tonality"] >= 0.65 and key_r >= 0.75:
        mode = "小调" if "minor" in key else "大调"
        key_txt = f"{key.split()[0]} {mode},底色{'偏暗' if mode == '小调' else '偏亮'}"
    elif key_r >= 0.65:
        key_txt = f"调性大概在 {key.split()[0]},但不太稳"
    else:
        key_txt = "调性判断不了(可能不是西方调式,也可能素材太稀)"

    # 3) 响度与母带
    if section:
        if rms >= -16:
            loud = "整段压得很满"
        elif rms >= -20:
            loud = "整段响度偏高"
        else:
            loud = "整段留了余量"
    elif rms >= -16 and peak > -1:
        loud = "响度压得很满,几乎没有留余量"
    elif rms >= -20:
        loud = "响度偏高,但还留了点余量"
    else:
        loud = "留了动态余量,不像压满的"
    loud += ",段落之间几乎没落差" if std < 7 else ",段落之间有明显的起伏"
    # 削波只说证据,不下『过载 / 母带』这种结论
    n_plateau = d.get("plateau_runs", 0)
    n_over = d.get("over_scale", 0)
    if n_plateau > 0:
        loud += f",有 {n_plateau} 处连续贴顶的平台(疑似削波)"
    elif n_over > 0:
        loud += ",解码后有样本超满刻度,但没找到连续平台,无法确认源文件削波"

    lines.append(f"{fmt_dur(dur)}的{'片段' if section else '曲子'},{speed}。")
    lines.append(f"{key_txt};{loud}。")

    # 4) 开头是否留白 —— 能量低只说明"更弱"，改由谱平坦度判断"有没有内容"：
    #    静音 flatness 0.5+，弱奏 0.01–0.06，有内容的中段 <0.001。能量分不开的，频谱分得开。
    if len(es) > 2 and es[0] < es[1] - 8 and es[0] < es[2] - 8 and ab:
        first_len = ab[0]["end_sec"] - ab[0]["start_sec"]
        med = float(np.median(es))
        drop = med - es[0]
        flat0 = d.get("head_flat")
        if flat0 is not None and not np.isnan(flat0) and flat0 > 0.30:
            lines.append(f"开头 {first_len:.0f} 秒几乎是空的(谱平坦度 {flat0:.2f},接近噪声),之后一路铺满。")
        elif drop >= 8:
            lines.append(f"开头 {first_len:.0f} 秒很轻(比中段低 {drop:.0f} dB),之后一路铺满。")

    # 5) 密度重心
    ons = [b["onsets"] for b in ab]
    if len(ons) >= 4:
        half = len(ons) // 2
        a = float(np.mean(ons[:half])) or 1e-9
        b = float(np.mean(ons[half:]))
        pi = int(np.argmax(ons))
        t0, t1 = ab[pi]["start_sec"], ab[pi]["end_sec"]
        if b / a < 0.8:
            lines.append(f"编曲重心在前半段:起音最密的是 {fmt_time(t0)}–{fmt_time(t1)},越往后越稀。")
        elif b / a > 1.25:
            lines.append(f"编曲重心在后半段:越往后越密,起音最密的是 {fmt_time(t0)}–{fmt_time(t1)}。")
        else:
            lines.append("前后密度差不多,靠音色和层次推进,不是靠加东西堆上去的。")

    # 6) 最大的一次反差:必须是"曲中"的,结尾淡出另算,别把收尾当高潮
    if len(es) >= 4:
        delta = np.diff(es)
        last_idx = len(es) - 2          # 最后一个可用的变化点(排除结尾块)
        drops = [j for j in np.where(delta < -4)[0] if j <= last_idx - 1]
        rises = [j for j in np.where(delta > 4)[0] if j <= last_idx]
        pair = None
        for j in drops:
            after = [k for k in rises if 0 <= k - j <= 2]
            if after and (pair is None or abs(delta[j]) > abs(delta[pair[0]])):
                pair = (int(j), int(after[0]))
        if pair:
            j, k = pair
            verb = "drop" if abs(delta[j]) >= 8 else "略收"
            lines.append(f"{fmt_time(blocks[j + 1][0])} {verb} {abs(delta[j]):.0f} dB,"
                         f"{fmt_time(blocks[k + 1][0])} 才回来 —— 这是最明显的一次反差。")
        elif rises:
            k = int(max(rises, key=lambda i: delta[i]))
            lines.append(f"{fmt_time(blocks[k + 1][0])} 进来,是最响的地方。")

    # 7) 收尾
    if len(es) >= 2:
        last, prev = float(es[-1]), float(es[-2])
        if last < prev - 6:
            tail_onsets = ab[-1]["onsets"] if ab else 0
            if tail_onsets > 0:
                lines.append(f"{fmt_time(blocks[-1][0])} 开始 fade out,但底下还留着东西,不是硬切。")
            else:
                lines.append(f"{fmt_time(blocks[-1][0])} hard cut。")

    # 8) 和声:只到根音
    vocab = chords.get("vocabulary") or []
    if fit["chords"] >= 0.45 and vocab:
        roots = "、".join(v["root"] for v in vocab[:4])
        lines.append(f"和声只到根音级:大致在 {roots} 之间转,别当谱子看。")
    elif vocab:
        lines.append("和声这层不可靠,只有几个候选根音,建议别引用。")

    # 9) 配器:不写进判断。
    #    实测(2026-10):这个声源模型把"唢呐"报进了两首根本没有唢呐的曲子,
    #    中文乐器整层是噪声。写进判断等于给读的人递一个编故事的材料。
    #    原始分数只在 --detail 的 [SRC] 里存档。

    out = []
    if section:
        tail = f"(全曲 {fmt_time(d['total_dur'])})" if d.get("total_dur") else ""
        out.append(f"[SCOPE] 只分析了 {fmt_time(d['offset'])}–{fmt_time(d['offset'] + dur)}{tail}")
    out.append("[判断] " + lines[0])
    out.extend("       " + ln for ln in lines[1:])
    out.append("")
    if d.get("bpm_source") == "beat_this":
        bpm_txt = f"{tempo:.0f} 拍(Beat This)"
    else:
        _lay2 = bpm_layers(tempo)
        bpm_txt = (f"{_lay2[0]:.0f} / {_lay2[1]:.0f} 拍(两层)"
                   if _lay2 else f"{tempo:.0f} 拍")
        if d["bpm_suspect"] and d.get("kick_bpm") and not np.isnan(d["kick_bpm"]):
            bpm_txt += f" / 低频换算 {d['kick_bpm']:.0f}"
    _lufs = d.get("lufs")
    _lufs_txt = (f" | 整合 {_lufs:.1f} LUFS"
                 if _lufs is not None and not np.isnan(_lufs) else "")
    out.append(f"[依据] 峰值 {peak:.1f} dBFS{_lufs_txt} | RMS {rms:.1f} dBFS"
               f" | 起音 {d['activity']['onset_rate_per_sec']:.2f} 次/秒"
               f" | {key} {key_r:.2f} | {bpm_txt}")
    out.append(f"[可信] beat {fit['beat']:.2f} | 调性 {fit['tonality']:.2f}"
               f" | 根音 {fit['chords']:.2f}")
    # 模型标签:只在超阈值时出现,且标明是「标签」不是「识别」
    mn = d.get("musicnn")
    if mn and mn.get("available") and mn.get("tags"):
        tags_txt = "、".join(f"{t['label']} {t['score']:.2f}" for t in mn["tags"])
        out.append(f"[标签] 模型预测标签(MTT 50 类,仅作辅助参考,不等同于流派识别): {tags_txt}")
    # 分轨(可选):分离后各轨能量占比 + 人声判定。
    # 分轨标签只当证据,不进 [判断] —— other 轨的 classical 不能变成「这是古典」。
    st = d.get("stems")
    if st and st.get("available"):
        order = [n for n in ("drums", "bass", "other", "vocals", "guitar", "piano")
                 if n in st.get("stems", {})]
        parts = [f"{n} {st['stems'][n]['ratio'] * 100:.0f}%" for n in order]
        vv = st.get("vocals") or {}
        if parts:
            out.append(f"[分轨] {vv.get('label', '已分离')} · " + " | ".join(parts))
        if vv:
            if vv.get("present"):
                out.append(f"       人声轨有实质能量({vv['ratio'] * 100:.0f}%),是唱的,不是纯器乐。")
            else:
                out.append(f"       人声轨只占 {vv['ratio'] * 100:.3f}%,整首没有真的人声。")
    # 歌词听写(可选组件):机器听写,只列可信段;疑幻觉段连内容都不给。
    asr = d.get("asr")
    _inst = ((asr or {}).get("instrumental")
             or (d.get("stems") or {}).get("instrumental") or [])
    if _inst:
        _rng = ", ".join(f"{fmt_time(r['start_sec'])}–{fmt_time(r['end_sec'])}"
                         for r in _inst)
        out.append(f"[分轨] 无人声区间: {_rng}")
        _med = float(np.median(es)) if len(es) else 0.0
        _links = _cross_links(ab, es, _inst, _med)
        if _links:
            out.append("[关联] " + _links[0])
            out.extend("       " + ln for ln in _links[1:])

    # 事件时间线：把能量 / 起音 / 人声串成先后关系
    # 不要加 d.get("detail") 条件 —— verdict_lines 只在默认路径被调用，
    # 加了条件等于永远不显示（--detail 走的是另一条输出路径）。
    _tl = _timeline(ab, es, blocks, _inst,
                    (d.get("stems") or {}).get("block_db"))
    if _tl:
        out.append("[时间线]")
        for _ts, _txt in _tl:
            out.append(f"       {fmt_time(_ts)}  {_txt}")

    # 钩子检测：能量 / 起音 /（有歌词时）字数，几项同时到峰值的位置
    _hook = _find_hook(ab, es, blocks, _inst, asr)
    if _hook:
        out.append("[钩子] " + _hook[0])
        out.extend("       " + ln for ln in _hook[1:])
    if asr and asr.get("available"):
        st = asr.get("stats") or {}
        if st.get("note"):
            out.append(f"[听写] {st['note']}")
        elif not asr.get("trusted"):
            # 2026-10-23 撤掉降级稿：不给未经筛选的识别正文，只给状态和统计。
            _n_draft = int(st.get("draft_withheld", 0))
            _n_hall = int(st.get("hallucination", 0))
            if _n_draft or _n_hall:
                out.append(f"[听写] 已执行听写，但没有任何一段通过可信度筛选"
                           f"（{_n_draft} 段存疑、{_n_hall} 段疑幻觉，正文不提供）—— "
                           "不把未验证的文字当歌词用")
            else:
                out.append("[听写] 已执行听写，没有可用的歌词段"
                           "(可能是外语/吟唱，也可能是模型在编)；别因此当成纯器乐")
        else:
            out.append(f"[听写] 机器听写候选(可能有错),较确定"
                       f" {len(asr['trusted'])}/{st.get('segments', '?')} 段:")
            for r in asr["trusted"][:12]:
                out.append(f"       {fmt_time(r['start_sec'])} {r['text']}")
            dropped = int(st.get("doubt", 0)) + int(st.get("hallucination", 0))
            if dropped:
                out.append(f"       其余 {dropped} 段不可信(疑幻觉/听不清),已省略")
    miss = ("音符级的旋律;乐器/人声只能给方向,不是识别"
            if (asr and asr.get("available"))
            else "音符级的旋律、歌词和语言;乐器/人声只能给方向,不是识别")
    out.append("[听不到] " + miss)

    # 自检:判断正文里不许出现测量值。这条规则以前只写在文档和注释里，没人校验；
    # 现在是断言—— 一旦有数字混进来就喊一声，下次改动能立刻发现回退。
    # 自检:判断正文里不许堆测量值。允许单个锚点(如「drop 9 dB」),
    # 因为一个数字能防误读;但一行里凑两个以上,就是在甩数据了。
    _bad = [ln for ln in out
            if (ln.startswith("[判断]") or ln.startswith("       "))
            and len(re.findall(r"\d\s*(?:dBFS|dB|拍|倍|次/秒)", ln)) >= 2]
    if _bad:
        print("[自检] 判断里混进了测量值,应挪到 [依据]: " + _bad[0].strip()[:60],
              file=sys.stderr)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--blocks", type=float, default=16.0, help="能量分块秒数(默认 16)")
    ap.add_argument("--chord-block", type=float, default=2.0, help="和弦判定块秒数(默认 2)")
    ap.add_argument("--start", type=parse_time, default=None,
                    help="只分析从这个时间点开始的片段,接受秒或 1:30")
    ap.add_argument("--end", type=parse_time, default=None,
                    help="分析到这个时间点为止,接受秒或 1:30")
    ap.add_argument("--json", action="store_true", help="输出结构化 JSON(供插件调用),默认输出人类可读文本")
    ap.add_argument("--brief", action="store_true",
                    help="配合 --json:去掉逐块时间线,只留结论级字段(约 1/5 的 token)")
    ap.add_argument("--detail", action="store_true",
                    help="输出完整测量明细(FIT/能量曲线/和弦骨架/声源),默认只给一段判断")
    ap.add_argument("--stems", nargs="?", const="4s", choices=["4s", "6s"],
                    help="分轨分析(可选组件):4s=人声/鼓/贝斯/其他,6s=再加吉他/钢琴")
    ap.add_argument("--asr", nargs="?", const="base", choices=["tiny", "base", "small"],
                    help="歌词听写(可选组件):4s 分轨取人声 + faster-whisper;只输出可信段")
    ap.add_argument("--preview", nargs="?", const=-1.0, type=float,
                    help="快速预览:自动选「又响又密」的段落(数据驱动);给数字则强制该秒数")
    ap.add_argument("--bands", action="store_true",
                    help="只输出分频段诊断(能量/打击密度/规律性 cv),不跑完整分析")
    ap.add_argument("--pitch", action="store_true",
                    help="音高轮廓(pyin):只给音域与有声占比,混音高通 150 Hz;慢,单独用")
    args = ap.parse_args()

    offset = float(args.start or 0.0)
    if args.end is not None and args.end <= offset:
        print(f"ERROR: --end({args.end:g}s)必须大于 --start({offset:g}s)", file=sys.stderr)
        sys.exit(2)
    duration = (args.end - offset) if args.end is not None else None

    # 预览模式:用数据选段(能量 + 起音都高于全曲中位数的连续块),跳过分轨/ASR。
    # 长度由歌本身决定,不预设秒数;只有给了数字才强制。
    preview_sec = None
    preview_how = ""
    preview_asr = None
    if args.preview is not None and args.end is None:
        try:
            _y, _sr = librosa.load(args.path, sr=None, mono=True)
            if args.preview and args.preview > 0:
                _hop = 512
                _rms = librosa.feature.rms(y=_y, hop_length=_hop)[0]
                _win = max(1, int(float(args.preview) * _sr / _hop))
                if len(_rms) > _win:
                    _sm = np.convolve(_rms, np.ones(_win) / _win, mode="valid")
                    offset = int(np.argmax(_sm)) * _hop / _sr
                duration = float(args.preview)
                preview_how = f"指定 {args.preview:g} 秒(最响处)"
            else:
                offset, duration, preview_how = pick_preview_segment(_y, _sr)
            preview_sec = duration
        except Exception:
            duration = 30.0
            preview_sec = duration
            preview_how = "回退:30 秒"
        preview_asr = "tiny"   # 预览默认带一段快速听写,证明「听到了」
        if args.stems or args.asr:
            print("[预览] 模式忽略 --stems/--asr(那两项要 1-2 分钟,预览要快)",
                  file=sys.stderr)
            args.stems = None
            args.asr = None

    total_dur = None
    try:
        total_dur = float(librosa.get_duration(path=args.path))
    except Exception:
        pass

    y, sr = None, None
    if total_dur is not None and offset >= total_dur:
        print(f"ERROR: 起点 {fmt_time(offset)} 已超出文件长度({fmt_time(total_dur)})",
              file=sys.stderr)
        sys.exit(2)
    try:
        y, sr = librosa.load(args.path, sr=None, mono=True, offset=offset, duration=duration)
    except Exception as e:
        # 越界时 libsndfile 会直接抛异常而不是返回空数组,翻译成可读的话
        print(f"ERROR: 无法读取 {fmt_time(offset)} 起的音频:{e}", file=sys.stderr)
        sys.exit(2)
    if len(y) == 0:
        print("ERROR: 该时间区间没有音频 —— 起点可能已超出文件长度", file=sys.stderr)
        sys.exit(2)
    dur = len(y) / sr
    section = offset > 0 or args.end is not None

    if args.bands:
        rows = analyze_bands(y, sr)
        print("[频段] 分频段诊断(能量 / 打击密度 / 规律性):")
        for _ln in format_bands(rows):
            print(_ln)
        return

    # chroma 只算一次:CQT 是这条流水线里最贵的一步,旧版为调性和和弦各算一遍。
    hop = 1024
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    chroma = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-8)

    tempo, tempo_alt, beat_times, kick_bpm, bpm_source = detect_tempo(y, sr)
    n_beats = len(beat_times)

    key, key_r, key_gap, key_top3 = detect_key(chroma)
    ka_type, ka_msg = key_ambiguity(key, key_top3, key_gap)

    # ---- 能量(绝对 dBFS,不归一化;时间统一换成全曲绝对时间)----
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    rms_dbfs = librosa.amplitude_to_db(rms, ref=1.0)   # 绝对 dBFS
    blocks = [(t0 + offset, t1 + offset, e)
              for t0, t1, e in energy_blocks(rms_dbfs, sr, hop, args.blocks)]

    # ---- 和弦:根音候选骨架(先算,[FIT] 要用它的差距)----
    chords = chord_analysis(chroma, sr, hop=hop, block_sec=args.chord_block,
                            t_offset=offset)

    # ---- 适用性评估:决定下面哪些结论能下、哪些只能算参考 ----
    fit = assess_fit(chroma, beat_times, key_gap, key_r, chords["mean_root_gap"])

    # 已知的关系型歧义(关系大小调/属调/平行调)意味着『中心明确』,可给一个下限 ——
    # 但必须先过绝对相关度这关,否则会把『模板不适用』(如非西方音乐)也抬成可用。
    if ka_type in ("relative", "fifth", "parallel") and key_r >= 0.75:
        fit["tonality"] = max(fit["tonality"], 0.55)

    # ---- 响度测量(提前算:JSON 与文本两种输出都要用)----
    # 削波判据必须看『贴顶采样点的数量』,不能看峰值 ——
    # 峰值高只说明电平高(波形仍光滑),真削波是波形被削平、大量点卡在 1.0。
    peak_db = 20 * np.log10(float(np.max(np.abs(y))) + 1e-12)
    rms_db = 20 * np.log10(float(np.sqrt(np.mean(y ** 2))) + 1e-12)
    lufs = integrated_lufs(y, sr)
    # 开头 16 秒的谱平坦度：用来区分「真的空」和「只是轻」——
    # 静音是平的（flatness 0.5+），弱奏有峰（0.01–0.06），中段更集中（<0.001）。
    _head = y[:int(16 * sr)] if len(y) > int(16 * sr) else y
    try:
        _S = np.abs(librosa.stft(_head, n_fft=2048, hop_length=512))
        head_flat = float(np.mean(librosa.feature.spectral_flatness(S=_S)))
    except Exception:  # noqa: BLE001
        head_flat = float("nan")

    # 削波判据(2026-10-20 修订):峰值高不等于削波。
    # 分开记三个量,让读的人自己判断,而不是给一个布尔值:
    #   over_scale : 浮点样本 > 1.0 —— 有损解码常见过冲,不代表源文件削波
    #   near_full  : 接近满刻度的样本数 —— 只是统计量,不是判据
    #   plateau    : 同极性连续 >=3 个样本贴着满刻度 —— 才是『波形被削平』的形态
    absy = np.abs(y)
    over_scale = int(np.sum(absy > 1.0))
    near_full = int(np.sum(absy >= 0.999))

    def _plateau_runs(mask, min_len=3):
        """数『连续 True 段』中长度 >= min_len 的个数。"""
        if not mask.any():
            return 0
        d_ = np.diff(mask.astype(np.int8))
        starts = np.where(d_ == 1)[0] + 1
        ends = np.where(d_ == -1)[0] + 1
        if mask[0]:
            starts = np.r_[0, starts]
        if mask[-1]:
            ends = np.r_[ends, len(mask)]
        if len(starts) != len(ends):
            return 0
        return int(np.sum((ends - starts) >= min_len))

    plateau_runs = _plateau_runs(y >= 0.999) + _plateau_runs(y <= -0.999)
    n_top = near_full          # 旧字段名保留,JSON 兼容

    # ---- 编曲活动度(与电平分开的另一个维度)----
    activity, activity_blocks = activity_timeline(y, sr, block_sec=args.blocks, hop=hop)
    for b in activity_blocks:
        b["start_sec"] = round(b["start_sec"] + offset, 1)
        b["end_sec"] = round(b["end_sec"] + offset, 1)

    # ---- 声源/乐器分类(可选增强:YAMNet;模型缺失时返回 None)----
    sources = analyze_sources(y, sr)
    musicnn = analyze_musicnn(y, sr)

    es = np.array([b[2] for b in blocks]) if blocks else np.array([-120.0])
    std_db = float(es.std()) if len(es) > 1 else 0.0

    tempo_stable = bool(np.isnan(tempo_alt) or
                        abs(tempo - tempo_alt) / max(tempo, 1) < 0.05)
    bpm_suspect = bool((not np.isnan(kick_bpm)) and tempo > 0
                       and (1.6 < kick_bpm / tempo < 2.4 or 0.4 < kick_bpm / tempo < 0.65))
    # 低频第二意见离题太远(比如安静的片段)就不报 —— 报出来只会被当成『另一个速度』。
    kick_ratio = (kick_bpm / tempo) if tempo > 0 else float("nan")
    bpm_alt = (None if np.isnan(kick_bpm) or np.isnan(kick_ratio)
               or not (0.3 <= kick_ratio <= 3.0) else round(kick_bpm, 1))

    # ---- 分轨(可选组件;失败不影响主分析)----
    stems = None
    if args.stems:
        if args.start is not None or args.end is not None:
            print("ERROR: --stems 只支持整首分析,不能和 --start/--end 同用", file=sys.stderr)
            sys.exit(2)
        _model = "htdemucs_6s" if args.stems == "6s" else "htdemucs"
        stems = analyze_stems(args.path, rms_db, model=_model, want_pitch=args.pitch)
        if not stems.get("available"):
            print(f"[分轨] 跳过: {stems.get('reason')}", file=sys.stderr)

    # ---- 歌词听写(可选组件;失败不影响主分析)----
    # 会再分离一次人声轨(与 --stems 不共用);慢是慢,但失败时主分析照常出结果。
    asr = None
    if args.asr:
        if args.start is not None or args.end is not None:
            print("ERROR: --asr 只支持整首分析,不能和 --start/--end 同用", file=sys.stderr)
            sys.exit(2)
        asr = analyze_lyrics(args.path, model=args.asr)
        if not asr.get("available"):
            print(f"[听写] 跳过: {asr.get('reason')}", file=sys.stderr)

    # ---- 结构化输出(插件调用用)----
    # 严格只整理已有结果:不新增算法、不引入依赖。
    if args.json or args.brief:
        bounds = []
        d = np.abs(np.diff(es)) if len(es) > 1 else np.array([])
        if len(d):
            for j in sorted(int(i) for i in np.argsort(-d)[:5] if d[i] > 4.0):
                bounds.append({
                    "time_sec": round(blocks[j + 1][0], 1),
                    "jump_db": round(float(d[j]), 1),
                })

        scope = {
            "start_sec": round(offset, 2),
            "end_sec": round(offset + dur, 2),
            "duration_sec": round(dur, 2),
            "file_duration_sec": None if total_dur is None else round(total_dur, 2),
        }

        if args.brief:
            hints = [{"label": h["label"], "score": h["score"]}
                     for h in (sources or {}).get("hints", [])][:4]
            print(json.dumps({
                "scope": scope,
                "section": section,
                "dur": round(dur, 1),
                "bpm": round(tempo, 1),
                "bpm_source": bpm_source,
                "bpm_alt": bpm_alt,
                "bpm_suspect": bpm_suspect,
                "bpm_layers": (None if bpm_source == "beat_this"
                               else ({"primary": bpm_layers(tempo)[0],
                                      "secondary": bpm_layers(tempo)[1]}
                                     if bpm_layers(tempo) else None)),
                "key": {"best": key, "r": round(key_r, 2), "rel": ka_type},
                "loud": {"peak": round(peak_db, 1), "rms": round(rms_db, 1),
                         "clipped": plateau_runs > 0,
                         "plateau_runs": plateau_runs,
                         "over_scale": over_scale},
                "energy": {"min": round(float(es.min()), 1),
                           "max": round(float(es.max()), 1),
                           "std": round(std_db, 1)},
                "fit": {"beat": round(fit["beat"], 2),
                        "tonal": round(fit["tonality"], 2),
                        "chord": round(fit["chords"], 2)},
                "chord_roots": [v["root"] for v in chords["vocabulary"][:6]],
                "chord_unresolved": chords["unresolved_ratio"],
                "seg": [b["time_sec"] for b in bounds],
                "src": [h["label"] for h in hints],
                "src_low": bool((sources or {}).get("low_confidence")),
                "stems": ({"vocals": stems.get("vocals"),
                           "ratios": {k: v["ratio"] for k, v in stems.get("stems", {}).items()}}
                          if stems and stems.get("available") else None),
                "asr": ({"available": True, "model": asr.get("model"),
                         "trusted": [{"t": r["start_sec"], "text": r["text"]}
                                     for r in (asr.get("trusted") or [])],
                         # 2026-10-23 撤掉降级稿：未通过筛选的文字不给正文。
                         # 只留数量，让人知道「跑过了但没有可用段」。
                         "draft_withheld": int((asr.get("stats") or {}).get("draft_withheld", 0)),
                         "transcription_status": asr.get("transcription_status"),
                         "vocal_evidence": asr.get("vocal_evidence"),
                         "inst": [{"a": r["start_sec"], "b": r["end_sec"]}
                                  for r in (asr.get("instrumental") or [])],
                         "stats": asr.get("stats")}
                        if asr and asr.get("available")
                        else ({"available": False, "reason": asr.get("reason")} if asr else None)),
            }, ensure_ascii=False, separators=(",", ":")))
            return

        # 读法说明:每个置信度怎么理解,直接给读 JSON 的模型看
        def read_fit(v):
            if v >= 0.65:
                return "可用"
            if v >= 0.45:
                return "偏低,仅作参考"
            return "不可用"

        caveats = [
            "和弦已降级为『根音候选骨架』:只报根音,unresolved_ratio 是根音定不下来的块比例",
            "七度/挂留/转位在混音上分不干净,已折叠到根音;quality 为 null 表示大三/小三未定",
            "所有数值均来自混音整体,未做音源分离 —— 无法归因到单个乐器",
            "confidence 表示『本方法对该音频的适用性』,不是『结论正确的概率』",
        ]
        if key_r < 0.65:
            caveats.append("调性相关度低,西方调性模板对本曲不适用 —— 候选应读作『找不到合适的调』")

        result = {
            "scope": scope,
            "summary": {
                "duration_sec": round(dur, 2),
                "bpm": round(tempo, 1),
                "bpm_source": bpm_source,
                "bpm_alternative": bpm_alt,
                "bpm_maybe_double": bpm_suspect,
                "bpm_layers": (None if bpm_source == "beat_this"
                               else ({"primary": bpm_layers(tempo)[0],
                                      "secondary": bpm_layers(tempo)[1]}
                                     if bpm_layers(tempo) else None)),
                "tempo_stable": tempo_stable,
                "key": {
                    "best": key,
                    "confidence": round(key_r, 3),
                    "gap_to_second": round(key_gap, 3),
                    "candidates": [{"name": n, "score": round(r, 3)} for r, n in key_top3],
                    "relation": ka_type,
                },
                "loudness": {
                    "peak_dbfs": round(peak_db, 2),
                    "rms_dbfs": round(rms_db, 2),
                    "over_scale_samples": over_scale,
                    "near_full_scale_samples": near_full,
                    "plateau_runs": plateau_runs,
                    "clipped": plateau_runs > 0,
                },
                "energy_range_dbfs": {
                    "min": round(float(es.min()), 2),
                    "max": round(float(es.max()), 2),
                    "std": round(std_db, 2),
                },
            },
            "timeline": {
                "energy": [
                    {"start_sec": round(t0, 1), "end_sec": round(t1, 1),
                     "mean_dbfs": round(e, 1)}
                    for t0, t1, e in blocks
                ],
                "section_boundaries": bounds,
            },
            "chords": chords,
            "confidence": {
                "fit": {k: round(float(v), 3) for k, v in fit.items()},
                "reading": {
                    "beat": read_fit(fit["beat"]),
                    "tonality": read_fit(fit["tonality"]),
                    "chords": read_fit(fit["chords"]),
                },
                "caveats": caveats,
            },
            "measurements": {
                "sample_rate": int(sr),
                "beat_count": int(n_beats),
                "energy_block_sec": args.blocks,
                "chord_block_sec": args.chord_block,
                "chord_blocks_total": int(chords["blocks_total"]),
            },
            "sources": sources if sources else {"available": False,
                                                "reason": "onnxruntime 或 YAMNet 模型缺失"},
            "activity": {
                "summary": activity,
                "timeline": activity_blocks,
                "note": "密度维度,与 loudness 分开;逐帧归一化后计算,与整体增益弱相关",
            },
        }
        if stems:
            result["stems"] = stems
        if asr:
            result["asr"] = asr
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # ---- 默认输出:先给判断,不甩数据 ----
    if preview_sec:
        print(f"[预览] 选取 {fmt_time(offset)}–{fmt_time(offset + dur)}"
              f"({preview_how};长度由能量与起音数据决定)"
              f" —— BPM 和结构在短片段上会偏,完整分析更准")

    # 用户要的是“这首歌是什么”,不是“这些数字是多少”。数据只在 --detail 里给。
    if not args.detail:
        for line in verdict_lines({
            "dur": dur, "section": section,
            "tempo": tempo, "bpm_suspect": bpm_suspect, "kick_bpm": kick_bpm,
            "bpm_source": bpm_source,
            "key": key, "key_r": key_r, "key_gap": key_gap, "ka_type": ka_type,
            "rms_db": rms_db, "peak_db": peak_db, "std_db": std_db,
            "lufs": lufs, "head_flat": head_flat,
            "n_top": n_top, "over_scale": over_scale, "plateau_runs": plateau_runs,
            "es": es, "blocks": blocks, "activity_blocks": activity_blocks,
            "chords": chords, "fit": fit, "sources": sources, "activity": activity,
            "musicnn": musicnn, "stems": stems, "asr": asr,
            "offset": offset, "total_dur": total_dur,
            "detail": bool(args.detail),
        }):
            print(line)
        if preview_asr and preview_sec:
            _lines = quick_transcribe(y, sr, preview_asr)
            if _lines:
                print("[听写] 听到这些词(可能错):" + " / ".join(_lines))
            else:
                print("[听写] 这段没听出词(可能纯器乐/人声太轻)")
        if args.pitch:
            emit_pitch(stems, y, sr)
        return

    # ---- 输出:先报适用性,再报数据 ----
    if section:
        total_txt = f" | 全曲 {fmt_time(total_dur)}" if total_dur else ""
        print(f"[SCOPE] 只分析了 {fmt_time(offset)}–{fmt_time(offset + dur)}"
              f"(区间时长 {dur:.1f}s{total_txt})")
        if dur < 20:
            print("       ⚠ 片段太短,节拍/调性/和弦的统计都不稳,FIT 分数会偏低")
    print("[FIT] 适用性(0-1,越高越可信)—— 先看这段,再决定信多少")
    print(fit_line("beat", fit["beat"], "BPM 可用", "节拍不稳或检测失败,BPM 不可信"))
    # 措辞分优先级:『模板不适用』要盖过『体系内歧义』——
    # 相关度本身就低时,再说『关系大小调/属调歧义』是拿错误的框架描述问题。
    print(fit_line("tonality", fit["tonality"], "调性可用",
                   tonality_hint(key_r, key_gap, ka_type, ka_msg, fit["tonality"])))
    print(fit_line("chords", fit["chords"], "根音骨架可用",
                   "只到根音级,和弦性质(大/小、七度)基本分不清"))
    print()
    meta_label = "区间时长" if section else "时长"
    print(f"[META] {meta_label} {fmt_time(dur)} | 采样率 {sr} | {n_beats} 个节拍点")
    if tempo <= 0:
        print("[BPM]  检测失败(片段太短或没有稳定脉冲)")
    elif tempo_stable:
        print(f"[BPM]  {tempo:.1f}(测速稳定)")
    else:
        print(f"[BPM]  {tempo:.1f}  可能差一倍(节拍间隔反推 {tempo_alt:.1f}),需人工确认")
    if section and dur < 90:
        print("       ⚠ 片段测速:节拍器在短片段上容易锁到『别的拍层级』,"
              "这个数不等于全曲速度")
    # 低频鼓点第二意见:与 beat tracker 差接近 2 倍时,说明两个拍层级都有能量
    # 拍速永远有两个层级:人耳和检测器常常差一倍,别让人把上面的数当唯一答案
    # 拍速永远有两个层级:人耳和检测器常常差一倍,别让人把上面的数当唯一答案。
    # Beat This 模式下不再提两层 —— 它本来就是给出一个确定的拍速,而 JSON 那边已经
    # 把 bpm_layers 置 null 了,文本这层以前没跟齐。
    _lay3 = None if bpm_source == "beat_this" else bpm_layers(tempo)
    if _lay3:
        print(f"       ⚠ 拍速是两层:{_lay3[0]:.0f} 和 {_lay3[1]:.0f}"
              " —— 人耳数的拍和检测器数的拍常常差一倍,别把上面那个当成唯一答案")
    if bpm_suspect:
        print(f"       低频鼓点独立推算 {kick_bpm:.0f} BPM(约 {kick_bpm / tempo:.1f}×)")
    top3_str = " | ".join(f"{n} ({r:.2f})" for r, n in key_top3)
    print(f"[KEY]  {top3_str}")
    if key_r < 0.65:
        print(f"       → ⚠ 最高相关度仅 {key_r:.2f}:西方调性模板对这段音乐不适用")
        print("         候选仍列出,但请读作『找不到合适的调』而不是『就是这个调』")
        print("         也可能是素材本身和声信息太少(例如只有鼓和低音),不是调式问题")
    else:
        note = {
            "clear":    f"次佳差距 {key_gap:.2f},最佳明显胜出",
            "relative": "两者共用全部音阶,大/小调倾向不明",
            "fifth":    "两者只差一个音(属调关系),调性中心基本明确",
            "parallel": "同根音,只是调式不同",
            "weak":     "候选之间无已知关系且差距不大,调性中心不明确",
        }[ka_type]
        # 和 [FIT] 保持一致:贴合度不够时,别把『基本明确』说成结论
        if fit["tonality"] < 0.45:
            _bn = key_top3[0][1]
            print(f"       → 最可能是 {_bn} —— 但最佳相关度只有 {key_r:.2f}"
                  f"(次佳 {key_r - key_gap:.2f},差 {key_gap:.2f}),具体调只能算猜测")
        else:
            print(f"       → {ka_msg} —— {note}")
    if plateau_runs > 0:
        clip = f"  ⚠{plateau_runs} 处连续贴顶平台(疑似削波)"
    elif over_scale > 0:
        clip = f"  {over_scale} 点超满刻度(可能是解码过冲,未发现平台)"
    else:
        clip = ""
    print(f"[DYN]  真峰值 {peak_db:.1f} dBFS{clip} | 整体RMS {rms_db:.1f} dBFS"
          f" | 段间波动 {std_db:.1f} dB")

    # 活动度:和响度是两回事 —— 响度看"多大声",活动度看"多密/多满"
    print(f"[ACT]  起音密度 {activity['onset_rate_per_sec']:.2f} 次/秒"
          f" | 活跃频带 {activity['active_band_ratio']:.2f}/1.0"
          f" | 频谱变化率 {activity['spectral_flux']:.4f}"
          f"  (← 密度维度,与上面响度分开看)")

    print(f"[FORM] 能量走势(每 {args.blocks:g} 秒一块,0 dBFS 为满刻度):")
    for t0, t1, e in blocks:
        n = int(np.clip((e + 60) / 3.0, 0, 20))
        print(f"  {fmt_time(t0)}–{fmt_time(t1)}  {e:6.1f}dB  {'█' * n}")

    # ---- 和弦:候选骨架 ----
    print()
    for line in chord_lines(chords, args.chord_block):
        print(line)

    # ---- 段落边界:只保留变化最大的 ----
    if len(blocks) > 3:
        d = np.abs(np.diff(es))
        if len(d):
            order = np.argsort(-d)[:5]
            pts = sorted(int(j) for j in order if d[j] > 4.0)
            if pts:
                s = ", ".join(f"{fmt_time(blocks[j + 1][0])}({d[j]:.0f}dB)" for j in pts)
                print(f"[SEG]  变化最大的段落边界: {s}")

    # ---- 声源/乐器(可选增强:YAMNet)----
    if sources and sources.get("available"):
        hints = sources.get("hints") or []
        hint_names = {h["label"] for h in hints}
        others = [t for t in sources.get("top", []) if t["label"] not in hint_names][:4]
        print()
        print("[SRC]  声源标签存档(YAMNet 521 类)—— 仅供参考,不要写进结论")
        _ve = sources.get("vocal")
        if _ve:
            _v = {"有": "有演唱", "无": "没检测到演唱"}.get(_ve["verdict"], "说不准")
            print(f"  [人声证据] max(Singing,Speech) 峰值 {_ve['peak']:.3f}"
                  f" @{fmt_time(_ve['peak_t'])} | p95 {_ve['p95']:.3f} → {_v}")
            # 「人声段」= YAMNet 当门卫 + 相对电平当手电筒。
            # YAMNet 单独做不了定位：它是事件检测，Queen 的 p95 只有 0.09，
            # 只有 4 秒超过 0.15。相对电平单独做不了判定：电子乐里 riser 也会被
            # demucs 分进 vocals 轨（Martin 实测三处全是音效，人耳确认）。
            if _ve["verdict"] == "有":
                _rel = (stems or {}).get("relative_level") or {}
                _all = _rel.get("all") or []
                _hit = [(t, t + 2.0) for t, r, _ in _all if r >= -12.0]
                _segs = []
                for s, e in _hit:
                    if _segs and s - _segs[-1][1] <= 1.5:
                        _segs[-1] = (_segs[-1][0], e)
                    else:
                        _segs.append((s, e))
                _segs = [(round(s, 1), round(e, 1)) for s, e in _segs if e - s >= 3.0]
                if _segs:
                    print("  [人声段] " + "、".join(
                        f"{fmt_time(a)}–{fmt_time(b)}" for a, b in _segs))
                else:
                    print("  [人声段] 有演唱，但分轨没给出明显区间（没跑分轨时也会这样）")
            else:
                print("  [人声段] YAMNet 判没检测到演唱 —— 不找段（避免把 riser/合成器当人声）")
            # 旋律来源:YAMNet 判「有没有唱过」,分轨判「分离干不干净」,两件事合起来才敢选轨
            _sr = (stems or {}).get("vocals", {}).get("ratio")
            if _ve["verdict"] == "有":
                if _sr is None:
                    _ms = "vocals（没跑分轨,没验证人声轨干不干净）"
                elif _sr > 0.05:
                    _ms = f"vocals（人声轨占 {_sr * 100:.0f}%,可以取）"
                else:
                    _ms = f"other（检测到演唱,但人声轨只占 {_sr * 100:.1f}%,分离不干净）"
            elif _ve["verdict"] == "无":
                _ms = "other（没检测到演唱）"
            else:
                _ms = "两条都跑（证据不足,别硬选）"
            print(f"  [旋律来源] {_ms}")
        if hints:
            for it in hints[:6]:
                print(f"  {it['score']:5.3f}  {it['label']}")
        else:
            print("  没有可用的乐器/地域线索 —— 模型给出的全是情绪/风格替身")
        if others:
            print("  以下高分标签是情绪/风格替身,不能当作『听到了什么』:")
            print("    " + "、".join(t["label"] for t in others))
        print("  ⚠ 中文乐器不可信:实测把『唢呐』报进了两首没有唢呐的曲子;"
              "这层是风格先验,不是识别")
        if sources.get("low_confidence"):
            print("  ⚠ 分数普遍偏低 —— 模型对本曲不确信(训练数据偏西方流行)")
    elif sources is None:
        print()
        print("[SRC]  (YAMNet 未安装,跳过声源分类)")

    # ---- 音高轮廓（--detail 路径；默认路径在 return 前已经调过）----
    if args.pitch:
        emit_pitch(stems, y, sr)

    # ---- 说人话:把上面的数字组织成能读的结论 ----
    print()
    print("[说人话]")
    print(narrate({
        "dur": dur,
        "tempo": tempo,
        "kick_bpm": kick_bpm,
        "rms_db": rms_db,
        "energy_std": std_db,
        "activity": activity,
        "activity_blocks": activity_blocks,
        "fit": fit,
        "key": key,
        "key_r": key_r,
        "sources": sources,
        "chords": chords,
        "section": section,
    }))


if __name__ == "__main__":
    main()
