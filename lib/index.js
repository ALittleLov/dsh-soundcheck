/** dsh-soundcheck host half —— 本地音乐分析工具,音频不出本机。 */
import { createHash } from 'node:crypto'
import { spawn } from 'node:child_process'
import { existsSync, writeFileSync, mkdirSync, readFileSync } from 'node:fs'
import { join, dirname, basename } from 'node:path'
import { fileURLToPath } from 'node:url'

export const name = 'dsh-soundcheck'

/**
 * 引擎默认跟着插件走:插件目录/../engine。
 * 分享出去时插件放在哪都能找到自己那份引擎;开发机上用 config.root 指回工作区。
 */
const PLUGIN_DIR = dirname(fileURLToPath(import.meta.url))
const DEFAULT_ROOT = join(PLUGIN_DIR, '..', 'engine')

function resolvePaths(config) {
  const root = config?.root ?? DEFAULT_ROOT
  // 用户数据目录(缓存/模型/HF 缓存)放在插件目录外, 覆盖升级插件不会丢
  const dataRoot = config?.dataRoot ?? join(PLUGIN_DIR, '..', 'dsh-soundcheck-data')
  return {
    root,
    dataRoot,
    python: config?.pythonPath ?? join(root, 'venv', 'Scripts', 'python.exe'),
    script: config?.scriptPath ?? join(root, 'analyze.py'),
    model: config?.modelPath ?? join(root, 'yamnet.onnx'),
    classes: config?.classMapPath ?? join(root, 'yamnet_class_map.csv'),
    musicnn: config?.musicnnPath ?? join(root, 'musicnn.onnx'),
    env: {
      DSH_AUDIO_DATA_DIR: dataRoot,
      DSH_AUDIO_CACHE_DIR: join(dataRoot, 'cache', 'stems'),
      DSH_AUDIO_MODEL_DIR: join(dataRoot, 'models', 'stems'),
      DSH_AUDIO_HF_HOME: join(dataRoot, 'hf_home'),
    },
  }
}

/** 起一个子进程并收齐输出;超时杀掉。 */
function run(python, args, timeoutMs, extraEnv = {}) {
  return new Promise(resolve => {
    let child
    try {
      child = spawn(python, args, {
        windowsHide: true,
        env: { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8', ...extraEnv },
      })
    } catch (error) {
      resolve({ ok: false, error: `SPAWN_FAILED: ${error.message}`, stdout: '', stderr: '' })
      return
    }

    let out = ''
    let err = ''
    let settled = false
    const finish = value => { if (!settled) { settled = true; clearTimeout(timer); resolve(value) } }
    const timer = setTimeout(() => {
      try { child.kill() } catch { /* already gone */ }
      finish({ ok: false, error: `TIMEOUT after ${timeoutMs}ms`, stdout: out, stderr: err })
    }, timeoutMs)

    child.stdout?.on('data', d => { out += d.toString() })
    child.stderr?.on('data', d => { err += d.toString() })
    child.on('error', e => finish({ ok: false, error: `SPAWN_FAILED: ${e.message}`, stdout: out, stderr: err }))
    child.on('close', code => finish({ ok: code === 0, code, stdout: out, stderr: err }))
  })
}

/** 接受秒数或 "1:30" / "1:02:03",与 analyze.py 的 --start/--end 一致。 */
function parseTime(value) {
  if (typeof value === 'number') return Number.isFinite(value) ? value : NaN
  const s = String(value ?? '').trim()
  if (!s) return NaN
  const parts = s.split(':')
  if (parts.length > 3) return NaN
  let total = 0
  for (const p of parts) {
    const n = Number(p)
    if (!Number.isFinite(n)) return NaN
    total = total * 60 + n
  }
  return total
}

/* ───────────────── provision:把引擎装到用户机器上 ───────────────── */

/** 找一个可用的 Python 3.10+。找不到就返回 null。 */
async function findPython() {
  for (const cmd of ['python', 'python3', 'py']) {
    const r = await run(cmd, ['-c', 'import sys;print("%d.%d" % sys.version_info[:2])'], 15000)
    if (!r.ok) continue
    const m = String(r.stdout).trim().match(/(\d+)\.(\d+)/)
    if (m && Number(m[1]) === 3 && Number(m[2]) >= 10) {
      return { cmd, version: `${m[1]}.${m[2]}` }
    }
  }
  return null
}

// 模型文件的 sha256，发布时写死。
// 为什么写死在代码里而不是运行时从网上下：如果哈希也从镜像站取，镜像可以连文件带哈希一起换，
// 校验就失效了。写死在代码里，镜像只能「送错货」，不能「偷换货」。
const CHECKSUMS = {
  'yamnet.onnx': '04e27fca08e7a3aea2630d1a63a51e6b437c803e0ff6e26399f80870ac251dda',
  'musicnn.onnx': 'a4d64c1dac195bdb118e789ddd38115c68028d26ad692448f5a8748e39490df6',
  'beat-this-final0.onnx': 'e5e37b7d1802895e42559c5a1ced7619b1601d257a5c8e167c9d45de37c1b078',
  'frontend.json': '7b2ba71db92b79cb33f45feb21854161de329b86a7e6ea686c3fec51ce85eb9f',
  'htdemucs_fp16weights.onnx': 'd05c269d0178d2a72ad484b10b11dd370193fc923201c3b27a99f848745db70a',
  'htdemucs_6s_fp16weights.onnx': '7ce55792e2231c93fbf92de95f5fd5b3a5e6c89f7db690dfd693e8f1dce56869',
}

/** 校验下载到的字节是否与登记一致。没登记的返回 ok（不阻断旧流程）。 */
function verifySha256(buf, name) {
  const want = CHECKSUMS[name]
  if (!want) return { ok: true }
  const got = createHash('sha256').update(buf).digest('hex')
  if (got !== want) {
    return { ok: false, error: `sha256 不匹配（期望 ${want.slice(0, 12)}… 实际 ${got.slice(0, 12)}…），文件可能被替换` }
  }
  return { ok: true }
}

/** 下载声源模型(跟随重定向,多镜像兜底)。 */
async function downloadModel(dest) {
  const urls = [
    'https://hf-mirror.com/niobures/YAMNet/resolve/main/yamnetonnx/yamnet.onnx',
    'https://huggingface.co/niobures/YAMNet/resolve/main/yamnetonnx/yamnet.onnx',
  ]
  let lastErr = ''
  for (const url of urls) {
    try {
      const res = await fetch(url, { redirect: 'follow' })
      if (!res.ok) { lastErr = `HTTP ${res.status}`; continue }
      const buf = Buffer.from(await res.arrayBuffer())
      // 太小说明拿到的是错误页,不是模型
      if (buf.length < 1_000_000) { lastErr = `too small: ${buf.length}B`; continue }
      const v = verifySha256(buf, basename(dest))
      if (!v.ok) {
        // 校验失败不写盘，换下一个源；全部失败后由调用方报错
        lastErr = v.error
        continue
      }
      mkdirSync(dirname(dest), { recursive: true })
      writeFileSync(dest, buf)
      return { ok: true, bytes: buf.length }
    } catch (e) {
      lastErr = e.message
    }
  }
  return { ok: false, error: lastErr }
}

/** 下载标签模型(约 3 MB)。
 *  分发包里已经带了,只有从源码/git 安装的人才会缺 —— 他们 .gitignore 挡了 *.onnx。
 *  缺失时插件照常工作,只是不输出 [标签] 行。 */
async function downloadMusicnn(dest) {
  // 固定模型资产放在 models-v1 Release;插件版本不再各自带模型
  const urls = [
    'https://github.com/ALittleLov/dsh-soundcheck/releases/download/models-v1/musicnn.onnx',
  ]
  let lastErr = ''
  for (const url of urls) {
    try {
      const res = await fetch(url, { redirect: 'follow' })
      if (!res.ok) { lastErr = `HTTP ${res.status}`; continue }
      const buf = Buffer.from(await res.arrayBuffer())
      // 太小说明拿到的是 404 页或 LFS 指针,不是模型
      if (buf.length < 1_000_000) { lastErr = `too small: ${buf.length}B`; continue }
      const v = verifySha256(buf, basename(dest))
      if (!v.ok) {
        // 校验失败不写盘，换下一个源；全部失败后由调用方报错
        lastErr = v.error
        continue
      }
      mkdirSync(dirname(dest), { recursive: true })
      writeFileSync(dest, buf)
      return { ok: true, bytes: buf.length }
    } catch (e) {
      lastErr = e.message
    }
  }
  return { ok: false, error: lastErr }
}

/** 下载节拍模型(78.3 MB + frontend.json 0.3 MB)。
 *  和"可选组件"不同,这个默认装:缺了它 BPM 会静默回退 librosa,
 *  0.6.6 修过的倍频问题会原样回来,而且不报错、没人知道。
 *  文件放 dataRoot/models/beat_this/,与 analyze.py 的查找路径一致。 */
async function downloadBeatThis(dir) {
  const base = 'https://github.com/ALittleLov/dsh-soundcheck/releases/download/models-v1'
  const files = [
    { name: 'beat-this-final0.onnx', min: 50_000_000 },
    { name: 'frontend.json', min: 1000 },
  ]
  let total = 0
  for (const f of files) {
    const dest = join(dir, f.name)
    if (existsSync(dest)) continue
    let lastErr = ''
    try {
      const res = await fetch(`${base}/${f.name}`, { redirect: 'follow' })
      if (!res.ok) {
        lastErr = `HTTP ${res.status}`
      } else {
        const buf = Buffer.from(await res.arrayBuffer())
        if (buf.length < f.min) {
          lastErr = `too small: ${buf.length}B`
        } else if (!verifySha256(buf, f.name).ok) {
          lastErr = verifySha256(buf, f.name).error
        } else {
          mkdirSync(dir, { recursive: true })
          writeFileSync(dest, buf)
          total += buf.length
          continue
        }
      }
    } catch (e) {
      lastErr = e.message
    }
    return { ok: false, error: `${f.name} 下载失败:${lastErr}` }
  }
  return { ok: true, bytes: total }
}

/** 下载分轨模型(htdemucs 4-stem 或 6-stem,fp16 权重,各约 150 MB)。
 *  只有用户明确要分轨时才下 —— 默认安装不占这 150 MB。 */
async function downloadStemModel(dest, sixs) {
  const urls = sixs
    ? [
      // 固定模型资产:models-v1 Release(不随插件版本变化) → hf-mirror 兜底
      'https://github.com/ALittleLov/dsh-soundcheck/releases/download/models-v1/htdemucs_6s_fp16weights.onnx',
      'https://hf-mirror.com/StemSplitio/htdemucs-6s-onnx/resolve/main/htdemucs_6s_fp16weights.onnx',
    ]
    : [
      // 固定模型资产:models-v1 Release(不随插件版本变化) → hf-mirror 兜底
      'https://github.com/ALittleLov/dsh-soundcheck/releases/download/models-v1/htdemucs_fp16weights.onnx',
      'https://hf-mirror.com/StemSplitio/htdemucs-onnx/resolve/main/htdemucs_fp16weights.onnx',
    ]
  let lastErr = ''
  for (const url of urls) {
    try {
      const res = await fetch(url, { redirect: 'follow' })
      if (!res.ok) { lastErr = `HTTP ${res.status}`; continue }
      const buf = Buffer.from(await res.arrayBuffer())
      // 50 MB 以下说明拿到的是错误页,不是模型
      if (buf.length < 50_000_000) { lastErr = `too small: ${buf.length}B`; continue }
      const v = verifySha256(buf, basename(dest))
      if (!v.ok) { lastErr = v.error; continue }
      mkdirSync(dirname(dest), { recursive: true })
      writeFileSync(dest, buf)
      return { ok: true, bytes: buf.length }
    } catch (e) {
      lastErr = e.message
    }
  }
  return { ok: false, error: lastErr }
}

/* ───────────────── 两个分析工具共用的执行体 ───────────────── */

const READ_RULES = `How to read the output:
- If a call returns ENGINE_NOT_INSTALLED, the engine is not set up yet. Tell the user that
  /audio-install does it in one step (~570 MB, a few minutes) — or, if the user asks you to
  handle it, call audio_provision yourself and then retry. Never walk them through pip.
- Default text output is a VERDICT ([判断]) followed by [依据] (the numbers behind it),
  [可信] (how far each method applies) and [听不到] (what this tool cannot hear).
  Lead with the verdict. Quote [依据] only when the user asks "why" or "how sure".
- NEVER copy numbers from [依据] into your reply. That block is for your own verification,
  not for forwarding. Say "推得很满,只剩一点余量" — not "-14.4 dBFS". Say "偏快" — not
  "161 拍". The only measurements you may repeat are timestamps and duration, because they
  point at something the user can relisten to. If the user explicitly asks for the numbers,
  then give them.
- [可信] below 0.45 means that method does not apply to this audio (e.g. non-Western
  scales) and must not be quoted as fact.
- [听写] is machine transcription, CANDIDATES only. Expect wrong characters; repeated
  lines are often the only reliable part. If it says "本曲没有可用的歌词段", do not
  conclude the song is instrumental — check the [分轨] vocals share first; if vocals are
  present, say the lyrics could not be made out (possibly foreign-language, chanting or
  vocalising) rather than inventing Chinese words. Never translate non-Chinese
  vocalising into Chinese — give an approximate transliteration ("shai-lu-way",
  "om banza…") and say it is a SOUND, not a meaning.
- Chord information is ROOT-LEVEL CANDIDATES only, never a progression; 7th/sus/inversion
  details are collapsed because a full mix cannot resolve them.
- Sound-source labels are weak evidence (scores around 0.01–0.03). Emotion and genre labels
  are filtered out; instrument and region hints are still only a direction, not a fact.
- Nothing is source-separated: all values come from the full mix and cannot be attributed
  to a single instrument.
- Genre is an inference, not a measured property. When asked, give ONE primary hypothesis at
  the most specific level the evidence supports.
  Evidence rules: never pick a genre solely because it has the highest tag score;
  model tags are predictions, not calibrated probabilities of mutually exclusive genres;
  treat tags as cross-checks, not as the source of the answer; multiple tags from the SAME
  model are one piece of evidence, not several; tempo/key/energy/onset density give context,
  but none of them alone identifies a genre.
  Decision rules: if different kinds of evidence point to the same broad genre, name it and
  state the strongest cue; if the tags conflict with the musical evidence, prefer a broader
  category instead of following the top tag; if the evidence only supports a broad genre,
  name that and stop; if it supports nothing reliable, say so rather than inventing a name.
  Never quote tag scores as the answer. Never state a genre as if it were a measurement.
- Be decisive. Lead with the most likely answer in a plain declarative sentence, then a
  confidence level (high / medium / low) and the one caveat that matters. When the evidence is
  partial, name WHICH layer is uncertain — do not hedge every sentence or retreat into raw
  numbers. A wrong guess you own is more useful than a correct non-answer.
- Guessing is allowed and expected — a multimodal model that actually hears the file guesses
  too, and gets things wrong. The cost of a clearly-labelled wrong guess is low; the cost of a
  non-answer is high. So: give ONE answer, not three maybes; attach confidence; and add one
  falsifier ("if it is X instead, you would hear …") so the user can check it in one listen.
  When the user corrects you, accept it and move on — no defending, no re-hedging.
  The hard floor never moves: never invent a measurement, never dress inference as measurement,
  never put a guessed number where the tool actually measured something.
- Talk about the MUSIC, not the machinery. Do not name the tool's internal sections
  ([判断]/[FIT]/[依据]), the model behind sound-source hints, or confidence
  decimals, unless the user asks how it works. Say "人声像在垫底", not "Choir 0.021".
- 回答纪律（红线，中文场景尤其容易踩）:
  · 段落名称 ≠ 段落角色：说「这段人声消失、器乐主导」，不说「这是副歌」。
  · 测量与推断分离：每个结论都要能回溯到一项测量。说「听着像 X」（带依据），
    不说「检测到 X」。找不到依据就不写。
  · 听不到的维度不描述（气声、真假音、压缩、声场）：用可测事实替代，
    如人声能量占比、起音密度。
  · 不把归因推给音源质量：ASR 置信度低可能是自语/吟唱，不一定是录音差。
  · 文件名和元数据不是测量：标题、标签、封面都可能过时或错，要判断里面有什么，去测。`

const DESCRIPTION = `Analyze a whole local audio file and return a short, human-readable verdict.

Use this when the user asks "how is this song", "what kind of track is this", or about its
structure, tempo, key, loudness, or where it changes. The tool measures the mix, then writes
the verdict in plain language: duration, tempo (and whether two metrical levels exist), key,
how hot it is, where the arrangement is fullest, the biggest contrast, and how it ends.

Text mode returns:
  [判断]   the verdict (read this first, and relay its substance rather than the raw numbers)
  [依据]   the few measurements behind the verdict
  [可信]   how far each method applies (0-1)
  [听不到] what the tool genuinely cannot hear

The [时间线] section (when present) may end an event with band deltas like "B1 -9dB/-43pp".
B1..B6 = 30–120 / 120–300 / 300–800 / 800–2000 / 2000–5000 / 5000–12000 Hz.
dB = that band's absolute energy change; pp = its share of the mix in percentage points.
Negative pp with positive dB means the band did NOT drop — other layers grew faster.
Never turn a band id into an instrument name: this measures spectral distribution, not who plays.

Options:
- preview=true returns a FAST OVERVIEW (6–40 s): auto-picks the loudest+densest segment
  from the audio data (length determined by the song, not a fixed window), reports
  duration/BPM/key/loudness for that segment. ~8 s, no stems and no ASR. Add
  preview_asr='tiny'|'base'|'small' to also get a short lyric snippet from that
  segment (may be absent if it is instrumental) — that costs ~+17 s. Use it when the
  user wants a quick sense of a song before committing to a full run.
- detail=true returns the full measurement dump ([FIT], per-block energy, chord skeleton,
  sound sources). Use it only when the user asks for the raw data or wants a deep dive.
- format=json returns everything structured (~2.6k tokens); brief=true returns only
  conclusion-level fields (~130 tokens).
- stems='4s' or '6s' (optional component) splits the mix and adds a [分轨] line:
  each stem's share plus a vocals verdict (有人声 / 纯器乐 / 人声很轻).
  '4s' = vocals/drums/bass/other; '6s' also separates guitar and piano.
  Use it when the user asks about specific parts ("does it have vocals", "how is
  the guitar", "what instruments"), not for pure parameter questions (duration,
  tempo, key). First run on a song takes ~60-90 s on CPU; results are cached,
  so later questions about the same file return instantly. If the component is
  missing the tool returns STEMS_NOT_INSTALLED and states the download size —
  never install it without telling the user.
- asr='tiny'|'base'|'small' (optional component) transcribes lyrics: it separates the
  vocals, runs Whisper, and adds a [听写] section listing only the lines it is fairly
  sure about. Treat every line as a CANDIDATE — wrong characters are normal. If it
  says "本曲没有可用的歌词段", say you could not make out the lyrics; do not fill them
  in. Use it when the user asks what a song is about or what the lyrics say. First
  run ~90–120 s plus a 75–460 MB model download.

Setup: if the engine is not installed, the user can run /audio-install in DSH (one command,
~570 MB) and check /audio-status. You can also call audio_provision for them when asked.

${READ_RULES}`

const SECTION_DESCRIPTION = `Analyze ONE time range of a local audio file (e.g. 1:30–2:10).

Use this when the user points at a specific moment — "what happens at 2:40", "compare
the intro and the chorus", "is this part louder/denser" — instead of re-analyzing the
whole track. It runs the same measurements as audio_analyze but only on [start, end],
and reports all timestamps as absolute positions in the file.

Returns the same verdict format as audio_analyze ([判断]/[依据]/[可信]/[听不到]) with a
[SCOPE] header naming the range. detail=true gives the full measurement dump.

Caveats specific to sections:
- All statistics come from the slice only, so short ranges are noisier.
- BPM on a range under ~90 s can lock onto a different metrical level. The output warns
  about this; never present it as the tempo of the whole track.

${READ_RULES}`

const BANDS_DESCRIPTION = `Internal diagnostic: split the spectrum into 5 bands and measure, per band,
how much energy it carries, how often it hits, and how regular those hits are
(cv = coefficient of variation of the gaps; < 0.5 = steady, > 1 = scattered).

Use it when the normal verdict is not enough to decide — e.g. "is this music or a field
recording", "is there really a shaker/hi-hat layer", "which band dominates". BPM/key/chords
are meaningless on non-music audio; this tool helps you tell that apart.

Output is for YOUR reasoning, not for the user: relay the conclusion ("there is a steady
high-frequency tick layer"), not the raw numbers.`

/** 校验公共前置条件;返回 null 表示可以继续。 */
function preflight(paths, audio) {
  if (!audio) return 'ERROR: path is required'
  if (!existsSync(audio)) return `ERROR: file not found: ${audio}`
  if (!existsSync(paths.python) || !existsSync(paths.script)) {
    // 不在分析时偷偷装依赖 —— 装什么、装多大,得让用户先知道。
    return 'ENGINE_NOT_INSTALLED: 本地分析引擎还没装。'
      + '请用户在 DSH 里运行 /audio-install(约 570 MB、几分钟),'
      + '或者明确让助手帮他装;装好后再重试这次分析。'
  }
  return null
}

/** 引擎是否可用(有 python 也有脚本)。 */
function engineReady(paths) {
  return existsSync(paths.python) && existsSync(paths.script)
}

/** /audio-status 用:把当前安装状态说清楚。 */
function statusText(paths) {
  const mark = join(paths.root, '.provisioned')
  const lines = [
    engineReady(paths) ? '引擎: 已安装 ✅' : '引擎: 未安装 ❌',
    `  目录: ${paths.root}`,
    `  Python 环境: ${existsSync(paths.python) ? '有' : '没有'}`,
    `  分析脚本: ${existsSync(paths.script) ? '有' : '没有'}`,
    `  声源模型: ${existsSync(paths.model) ? '已下载 (16 MB)' : '未下载 (安装时自动下)'}`,
    `  标签模型: ${existsSync(paths.musicnn) ? '有 (3 MB)' : '缺失 — 不影响主功能,但 [标签] 行不会出现'}`,
    `  节拍模型: ${existsSync(join(paths.dataRoot, 'models', 'beat_this', 'beat-this-final0.onnx'))
      ? '有 (78 MB) — BPM 用新算法'
      : '缺失 — BPM 会回退到 librosa(旧算法,可能锁错倍频);重跑 /audio-install 可补'}`,
  ]
  if (existsSync(mark)) {
    try {
      lines.push(`  安装标记: ${String(readFileSync(mark, 'utf8')).slice(0, 24)}`)
    } catch { /* 读不到就算了 */ }
  }
  if (!engineReady(paths)) {
    lines.push('', '运行 /audio-install 安装(约 570 MB,几分钟;走国内镜像)。')
  }
  return lines.join('\n')
}

/** 装引擎:找 Python → 建 venv → 装依赖(三级镜像) → 下模型 → 写标记。幂等。 */
async function provision(paths, opts = {}) {
  const mark = join(paths.root, '.provisioned')
  if (existsSync(mark) && existsSync(paths.python)) {
    // 已装过,但可能是老版本 —— 补上后来才加的东西
    const notes = []
    if (!existsSync(paths.musicnn)) {
      const dl = await downloadMusicnn(paths.musicnn)
      notes.push(dl.ok
        ? `补下了标签模型(${(dl.bytes / 1048576).toFixed(1)} MB)`
        : `标签模型下载失败:${String(dl.error).slice(0, 100)}(不影响分析,只是没有 [标签] 行)`)
    }
    const beatDir = join(paths.dataRoot, 'models', 'beat_this')
    if (!existsSync(join(beatDir, 'beat-this-final0.onnx'))) {
      const dl = await downloadBeatThis(beatDir)
      if (dl.ok) {
        notes.push(`补下了节拍模型(${(dl.bytes / 1048576).toFixed(1)} MB)`)
      } else {
        return {
          ok: false,
          text: `引擎已安装,但 节拍模型下载失败:${String(dl.error).slice(0, 120)}\n`
            + 'BPM 会回退到 librosa(旧算法,可能锁错倍频)。可以稍后重跑 /audio-install 再试。',
        }
      }
    }
    if (notes.length) return { ok: true, text: '引擎已安装;' + notes.join(';') }
    return { ok: true, text: '引擎已安装(.provisioned 标记存在),无需重复。' }
  }

  const steps = []

  // 1. 找 Python
  const py = await findPython()
  if (!py) {
    return { ok: false, text: 'ERROR: 找不到 Python 3.10+。请先安装 Python 并确保它在 PATH 里。' }
  }
  steps.push(`Python ${py.version}(${py.cmd})`)

  // 2. 建 venv
  if (!existsSync(paths.python)) {
    const r = await run(py.cmd, ['-m', 'venv', join(paths.root, 'venv')], 300_000)
    if (!r.ok) {
      return { ok: false, text: `ERROR: 建 venv 失败\n${(r.stderr || r.error).slice(0, 800)}` }
    }
    steps.push('venv 已建')
  } else {
    steps.push('venv 已存在')
  }

  // 3. 装依赖:默认源 → 清华 → 阿里云,谁通用谁
  //    (国内直连 PyPI 经常超时,不给镜像兜底的话,分享出去就卡在这一步)
  const mirrors = [
    null,
    'https://pypi.tuna.tsinghua.edu.cn/simple',
    'https://mirrors.aliyun.com/pypi/simple',
  ]
  let deps = null
  let usedMirror = ''
  let depErr = ''
  for (const mirror of mirrors) {
    const args = ['-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
      'librosa', 'onnxruntime']
    if (mirror) args.push('-i', mirror)
    const r = await run(paths.python, args, 1_500_000)
    if (r.ok) {
      deps = r
      usedMirror = mirror ? `(走 ${mirror.split('/')[2]})` : '(默认源)'
      break
    }
    depErr = r.stderr || r.error || `exit ${r.code}`
  }
  if (!deps) {
    return {
      ok: false,
      text: `ERROR: 装依赖失败(默认源和国内镜像都试过了)。可手动执行:\n  "${paths.python}" -m pip install librosa onnxruntime -i https://mirrors.aliyun.com/pypi/simple\n\n${depErr.slice(0, 1200)}`,
    }
  }
  steps.push(`依赖已装(librosa + onnxruntime)${usedMirror}`)

  // 4. 下模型
  const modelPath = paths.model
  if (!existsSync(modelPath)) {
    const dl = await downloadModel(modelPath)
    if (!dl.ok) {
      return { ok: false, text: `ERROR: 下模型失败:${dl.error}` }
    }
    steps.push(`模型已下(${Math.round(dl.bytes / 1048576)} MB)`)
  } else {
    steps.push('模型已存在')
  }

  // 4b. 标签模型(可缺失,不影响主功能)
  if (!existsSync(paths.musicnn)) {
    const dl2 = await downloadMusicnn(paths.musicnn)
    if (dl2.ok) {
      steps.push(`标签模型已下(${(dl2.bytes / 1024 / 1024).toFixed(1)} MB)`)
    } else {
      steps.push(`标签模型未下(${String(dl2.error).slice(0, 60)})—— [标签] 行不可用,其余正常`)
    }
  } else {
    steps.push('标签模型已存在')
  }

  // 4c. 节拍模型(默认装,不是可选 —— 缺了 BPM 静默降级)
  const beatDir = join(paths.dataRoot, 'models', 'beat_this')
  if (existsSync(join(beatDir, 'beat-this-final0.onnx'))) {
    steps.push('节拍模型已存在')
  } else {
    const dl3 = await downloadBeatThis(beatDir)
    if (dl3.ok) {
      steps.push(`节拍模型已下(${(dl3.bytes / 1048576).toFixed(1)} MB)`)
    } else {
      steps.push(`节拍模型未下(${String(dl3.error).slice(0, 60)})—— BPM 会回退到旧算法`)
    }
  }

  // 4d. 分轨组件(可选;只有明确要求时才装 —— 默认不占这 150 MB)
  if (opts.stems) {
    const stemsDir = join(paths.root, 'models', 'stems')
    const sixs = opts.stems === '6s'
    const modelFile = join(stemsDir, sixs
      ? 'htdemucs_6s_fp16weights.onnx'
      : 'htdemucs_fp16weights.onnx')

    let stemsDeps = null
    for (const mirror of mirrors) {
      const args = ['-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
        'demucs-onnx', 'soundfile', 'soxr']
      if (mirror) args.push('-i', mirror)
      const r = await run(paths.python, args, 900_000)
      if (r.ok) { stemsDeps = r; break }
    }
    if (!stemsDeps) {
      steps.push('分轨组件安装失败(demucs-onnx)—— [分轨] 不可用')
    } else {
      steps.push('分轨组件已装(demucs-onnx + soundfile + soxr)')
      if (!existsSync(modelFile)) {
        const dl = await downloadStemModel(modelFile, sixs)
        if (dl.ok) {
          steps.push(`分轨模型已下(${(dl.bytes / 1048576).toFixed(0)} MB${sixs ? '(6-stem)' : '(4-stem)'})`)
        } else {
          steps.push(`分轨模型下载失败:${String(dl.error).slice(0, 80)}`)
        }
      } else {
        steps.push('分轨模型已存在')
      }
    }
  }

  // 4d. 歌词听写组件(可选;只有明确要求时才装)
  if (opts.asr) {
    const asrModel = typeof opts.asr === 'string' ? opts.asr : 'base'
    let asrDeps = null
    for (const mirror of mirrors) {
      const args = ['-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
        'faster-whisper']
      if (mirror) args.push('-i', mirror)
      const r = await run(paths.python, args, 900_000)
      if (r.ok) { asrDeps = r; break }
    }
    if (!asrDeps) {
      steps.push('歌词听写组件安装失败(faster-whisper)—— [听写] 不可用')
    } else {
      steps.push('歌词听写组件已装(faster-whisper)')
      // 预下模型,免得用户第一次 --asr 时卡在下载上
      const whisperDir = join(paths.dataRoot, 'whisper').replace(/\\/g, '/')
      const asrPy = 'from faster_whisper import WhisperModel; '
        + `WhisperModel(${JSON.stringify(asrModel)}, download_root=${JSON.stringify(whisperDir)})`
      const dlAsr = await run(paths.python, ['-c', asrPy], 900_000, {
        ...paths.env,
        HF_ENDPOINT: 'https://hf-mirror.com',
        HF_HUB_DISABLE_XET: '1',   // hf-mirror 不支持 xet CAS
      })
      if (dlAsr.ok) {
        steps.push(`Whisper ${asrModel} 模型已下`)
      } else {
        steps.push(`Whisper 模型未下(${String(dlAsr.stderr || dlAsr.error || '').slice(0, 80)})—— 首次 --asr 时会再试`)
      }
    }
  }

  // 5. 写标记
  try { writeFileSync(mark, new Date().toISOString()) } catch { /* 标记失败不影响使用 */ }

  return {
    ok: true,
    text: '安装完成:\n- ' + steps.join('\n- ') + '\n\n现在可以分析音频了。',
  }
}

const TEXT_OUTPUT = {
  schema: {
    type: 'object',
    additionalProperties: false,
    required: ['ok', 'text'],
    properties: {
      ok: { type: 'boolean', description: 'Whether analysis succeeded.' },
      text: { type: 'string', description: 'Analyzer output (text or JSON) or an error message.' },
    },
  },
  render: (_args, value) => [{ type: 'text', text: String(value?.text ?? '') }],
}

export function apply(root, config) {
  const paths = resolvePaths(config)

  root.inject(['tools'], ctx => {
    ctx.tools.register({
      name: 'audio_analyze',
      description: DESCRIPTION,
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'Absolute path to a local audio file (mp3/flac/wav/m4a/ogg).',
          },
          format: {
            type: 'string',
            enum: ['text', 'json'],
            description: 'text (default, human/model readable) or json (structured, for programmatic use).',
          },
          brief: {
            type: 'boolean',
            description: 'With json: return a compact summary instead of the full evidence tree (~130 vs ~2600 tokens).',
          },
          detail: {
            type: 'boolean',
            description: 'Return the full measurement dump instead of the short verdict.',
          },
          blocks: {
            type: 'number',
            description: 'Energy block length in seconds for the [FORM] timeline. Default 16.',
          },
          stems: {
            type: 'string',
            enum: ['4s', '6s'],
            description: 'Optional stem separation. 4s = vocals/drums/bass/other; 6s = adds guitar/piano. Adds a [分轨] section. First run ~60-90 s (cached afterwards).',
          },
          preview: {
            type: 'boolean',
            description: 'Fast overview (6–40 s): auto-picks the loudest+densest segment from the audio data, reports duration/BPM/key/loudness. ~8 s, runs no stems and no ASR. Prefer this when the user just wants a quick sense of a song; run without it for a full analysis.',
          },
          preview_asr: {
            type: 'string',
            enum: ['tiny', 'base', 'small'],
            description: 'Only with preview=true: also transcribe a lyric snippet from the picked segment (separates that segment, then Whisper; adds ~17 s, tiny recommended). Default off. Lines are CANDIDATES — wrong characters are normal.',
          },
          asr: {
            type: 'string',
            enum: ['tiny', 'base', 'small'],
            description: 'Optional lyrics transcription (ASR). Separates vocals, runs Whisper, adds a [听写] section with only the lines it is fairly sure about. Lines are CANDIDATES — wrong characters are normal; never present them as exact lyrics. First run ~90–120 s plus a 75–460 MB model download.',
          },
          async: {
            type: 'boolean',
            description: 'Run as a background job and return immediately with a job id (recommended for stems/asr, which take 1–4 minutes). DSH notifies you when it finishes; collect the result with job_output. Default false = block until the analysis is done.',
          },
        },
        required: ['path'],
      },
      execute: async (args, exec) => {
        const audio = String(args?.path ?? '')
        const bad = preflight(paths, audio)
        if (bad) return { ok: false, text: bad }

        const argv = [paths.script, audio]
        if (args?.format === 'json') argv.push('--json')
        if (args?.brief) argv.push('--brief')
        if (args?.detail) argv.push('--detail')
        if (typeof args?.blocks === 'number' && args.blocks > 0) {
          argv.push('--blocks', String(args.blocks))
        }
        if (args?.stems === '4s' || args?.stems === '6s') argv.push('--stems', args.stems)
        if (args?.preview === true) argv.push('--preview')
        if (args?.preview_asr === 'tiny' || args?.preview_asr === 'base' || args?.preview_asr === 'small') {
          argv.push('--preview-asr', args.preview_asr)
        }
        if (args?.asr === 'tiny' || args?.asr === 'base' || args?.asr === 'small') {
          argv.push('--asr', args.asr)
        }

        const timeout = args?.asr ? 420_000
          : (args?.stems === '4s' || args?.stems === '6s') ? 300_000 : 180_000

        // ── 后台模式(默认关)──────────────────────────────────────────
        // 分轨/听写要 1–4 分钟,阻塞调用会让用户干等。传 async: true 时把工作
        // 交给 DSH 的后台任务(ctx.jobs),立刻返回 job id;跑完 DSH 会通知模型,
        // 模型再用 job_output 收结果。不传 async 时行为与旧版完全一致。
        if (args?.async === true) {
          const jobs = typeof ctx.get === 'function' ? ctx.get('jobs') : undefined
          if (!jobs || typeof jobs.start !== 'function') {
            return {
              ok: false,
              text: 'BACKGROUND_UNAVAILABLE: 当前宿主没有加载后台任务服务(dsh-jobs)。'
                + '去掉 async 参数改为同步调用即可。',
            }
          }
          const label = `分析 ${audio.split(/[\\/]/).pop()}`
          let jobId = ''
          try {
            jobId = jobs.start({
              kind: 'audio',
              label,
              owner: exec?.agent?.id,
              run: (handle) => {
                const started = Date.now()
                const tick = setInterval(() => {
                  handle.updateProgress(`分析中… ${Math.round((Date.now() - started) / 1000)}s`)
                }, 1000)
                const done = (async () => {
                  const r = await run(paths.python, argv, timeout, paths.env)
                  clearInterval(tick)
                  if (!r.ok) {
                    const detail = (r.stderr || r.error || '').slice(0, 800)
                    handle.append(`ERROR: ${r.error ?? `exit ${r.code}`}\n${detail}`)
                    return { status: 'failed', detail: r.error ?? `exit ${r.code}` }
                  }
                  handle.append(r.stdout.trim() || '(analyzer produced no output)')
                  return { status: 'completed', detail: 'done' }
                })()
                return { done, cancel: () => clearInterval(tick) }
              },
            })
          } catch (e) {
            return { ok: false, text: `BACKGROUND_FAILED: ${String(e?.message ?? e).slice(0, 300)}` }
          }
          return {
            ok: true,
            text: `已提交后台分析(job ${jobId})。预计 ${args?.asr ? '2–4' : '1–2'} 分钟,`
              + '跑完会通知你;到时候用 job_output 取结果,别重复提交。',
          }
        }

        const r = await run(paths.python, argv, timeout, paths.env)

        if (!r.ok) {
          const detail = (r.stderr || r.error || '').slice(0, 1200)
          return { ok: false, text: `ERROR: analysis failed (${r.error ?? `exit ${r.code}`})\n${detail}` }
        }

        const text = r.stdout.trim()
        if (!text) return { ok: false, text: 'ERROR: analyzer produced no output' }
        return { ok: true, text }
      },
      output: TEXT_OUTPUT,
    })

    // ── 局部分析:用户追问「这一分钟怎么回事」时用 ──
    ctx.tools.register({
      name: 'audio_section',
      description: SECTION_DESCRIPTION,
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'Absolute path to a local audio file (mp3/flac/wav/m4a/ogg).',
          },
          start: {
            description: 'Start of the range: seconds (number) or "m:ss" (e.g. "1:30").',
          },
          end: {
            description: 'End of the range: seconds (number) or "m:ss" (e.g. "2:10").',
          },
          format: {
            type: 'string',
            enum: ['text', 'json'],
            description: 'text (default) or json (structured).',
          },
          brief: {
            type: 'boolean',
            description: 'With json: return only conclusion-level fields (much fewer tokens).',
          },
          detail: {
            type: 'boolean',
            description: 'Return the full measurement dump instead of the short verdict.',
          },
          blocks: {
            type: 'number',
            description: 'Energy block length in seconds for this range. Default 16; use 4–8 for short ranges.',
          },
        },
        required: ['path', 'start', 'end'],
      },
      execute: async (args) => {
        const audio = String(args?.path ?? '')
        const bad = preflight(paths, audio)
        if (bad) return { ok: false, text: bad }

        const start = parseTime(args?.start)
        const end = parseTime(args?.end)
        if (!Number.isFinite(start) || !Number.isFinite(end)) {
          return { ok: false, text: 'ERROR: start/end must be seconds (90) or "m:ss" ("1:30")' }
        }
        if (end <= start) {
          return { ok: false, text: `ERROR: end (${end}s) must be greater than start (${start}s)` }
        }
        if (end - start < 2) {
          return { ok: false, text: 'ERROR: range is too short to analyze (need at least 2 s)' }
        }

        const argv = [paths.script, audio, '--start', String(start), '--end', String(end)]
        if (args?.format === 'json') argv.push('--json')
        if (args?.brief) argv.push('--brief')
        if (args?.detail) argv.push('--detail')
        if (typeof args?.blocks === 'number' && args.blocks > 0) {
          argv.push('--blocks', String(args.blocks))
        }

        const r = await run(paths.python, argv, 180_000, paths.env)

        if (!r.ok) {
          const detail = (r.stderr || r.error || '').slice(0, 1200)
          return { ok: false, text: `ERROR: section analysis failed (${r.error ?? `exit ${r.code}`})\n${detail}` }
        }

        const text = r.stdout.trim()
        if (!text) return { ok: false, text: 'ERROR: analyzer produced no output' }
        return { ok: true, text }
      },
      output: TEXT_OUTPUT,
    })

    // ── 分频段诊断:内部用,判断「音乐还是录音」/验证某层打击 ──
    ctx.tools.register({
      name: 'audio_bands',
      description: BANDS_DESCRIPTION,
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'Absolute path to a local audio file (mp3/flac/wav/m4a/ogg).',
          },
          start: {
            description: 'Optional start: seconds or "m:ss".',
          },
          end: {
            description: 'Optional end: seconds or "m:ss".',
          },
        },
        required: ['path'],
      },
      execute: async (args) => {
        const audio = String(args?.path ?? '')
        const bad = preflight(paths, audio)
        if (bad) return { ok: false, text: bad }

        const argv = [paths.script, audio, '--bands']
        if (args?.start !== undefined || args?.end !== undefined) {
          const start = parseTime(args?.start ?? 0)
          const end = parseTime(args?.end)
          if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
            return { ok: false, text: 'ERROR: start/end must be seconds (90) or "m:ss" ("1:30"), end > start' }
          }
          argv.push('--start', String(start), '--end', String(end))
        }

        const r = await run(paths.python, argv, 120_000, paths.env)
        if (!r.ok) {
          const detail = (r.stderr || r.error || '').slice(0, 1200)
          return { ok: false, text: `ERROR: band analysis failed (${r.error ?? `exit ${r.code}`})\n${detail}` }
        }
        const text = r.stdout.trim()
        if (!text) return { ok: false, text: 'ERROR: analyzer produced no output' }
        return { ok: true, text }
      },
      output: TEXT_OUTPUT,
    })

    // ── 安装引擎:建 venv + 装依赖 + 下模型 ──
    // 幂等:已装就直接返回。这是「点一下就装前置」背后真正干活的那个工具。
    ctx.tools.register({
      name: 'audio_provision',
      description: `Install the local audio engine (Python venv + librosa + onnxruntime + sound-source model).

Call this once if audio_analyze reports the engine is missing. It needs a Python 3.10+
available on PATH. It will:
  1. locate a usable Python
  2. create a venv under the engine directory
  3. pip install librosa + onnxruntime
  4. download the 16 MB sound-source model (~16 MB) from HuggingFace mirrors
  5. (optional) install faster-whisper + a Whisper model for lyrics transcription
  6. write a .provisioned marker so later calls are instant

Idempotent. Takes several minutes on first run (total download ~570 MB: deps + the sound-source model + the label model + the 78 MB beat model).`,
      parameters: {
        type: 'object',
        properties: {
          stems: {
            type: 'string',
            enum: ['4s', '6s'],
            description: 'Also install optional stem separation (demucs-onnx + model): 4s ~150 MB, 6s ~300 MB. Needed for the [分轨] output.',
          },
          asr: {
            type: 'string',
            enum: ['tiny', 'base', 'small'],
            description: 'Also install optional lyrics transcription (faster-whisper + Whisper model): tiny ~75 MB, base ~145 MB, small ~460 MB. Needed for the [听写] output.',
          },
        },
        required: [],
      },
      execute: async (args) => provision(paths, { stems: args?.stems, asr: args?.asr }),
      output: TEXT_OUTPUT,
    })
  })

  // ── 命令:DSH 的 / 菜单里会有这两个「按钮」 ──
  // /audio-status  看装没装
  // /audio-install 一键装(会告诉用户要下多少)
  try {
    root.inject(['commands'], ctx => {
      const offs = []
      offs.push(ctx.commands.register({
        name: 'audio-status',
        description: '查看 dsh-soundcheck 引擎状态:Python 环境、分析脚本、声源模型是否就绪',
        handler: () => ({ kind: 'success', text: statusText(paths) }),
      }))
      offs.push(ctx.commands.register({
        name: 'audio-install',
        description: '安装/修复 dsh-soundcheck 引擎(建 Python 环境、装 librosa + onnxruntime、下声源模型 16 MB / 标签模型 3 MB / 节拍模型 78 MB;约 570 MB,几分钟)',
        handler: async () => {
          const r = await provision(paths)
          return { kind: r.ok ? 'success' : 'error', text: r.text }
        },
      }))
      return () => { for (const off of offs) { try { off?.() } catch { /* 已卸载 */ } } }
    })
  } catch {
    // 命令服务不可用时,插件其余功能照常 —— 工具仍然能用
  }
}
