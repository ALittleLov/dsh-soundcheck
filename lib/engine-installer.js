/**
 * s8a1akuc
 *
 * 099g0oxa
 * y8dxmxkc
 * ia97l63t
 * 4o5i4hec
 * 1a9k9vm0
 * j0af7pl4
 * 0w4ux76a
 */
import { createHash } from 'node:crypto'
import { spawn } from 'node:child_process'
import {
  closeSync, copyFileSync, existsSync, mkdirSync, openSync, readFileSync,
  readdirSync, renameSync, rmSync, statSync, unlinkSync, writeFileSync,
} from 'node:fs'
import { basename, dirname, isAbsolute, join } from 'node:path'

export const ENGINE_VERSION = '0.1.1'
const PYTHON_TARGET = '3.12'
const PYTHON_INSTALLER_URL = 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe'
// 3tb9plqk
// a8u5pou4
// 0bgo0t6q
const PYTHON_SHA256 = '67b5635e80ea51072b87941312d00ec8927c4db9ba18938f7ad2d27b328b95fb'

export const CHECKSUMS = {
  'yamnet.onnx': '1510041dce24a2e9e84ec546807ac408ae496da6d1ed41bc3ccba649623f8e19',
  'yamnet_class_map.csv': 'cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2',
  'beat-this-final0.onnx': 'e5e37b7d1802895e42559c5a1ced7619b1601d257a5c8e167c9d45de37c1b078',
  'frontend.json': '7b2ba71db92b79cb33f45feb21854161de329b86a7e6ea686c3fec51ce85eb9f',
  'htdemucs_fp16weights.onnx': 'd05c269d0178d2a72ad484b10b11dd370193fc923201c3b27a99f848745db70a',
  'htdemucs_6s_fp16weights.onnx': '7ce55792e2231c93fbf92de95f5fd5b3a5e6c89f7db690dfd693e8f1dce56869',
}

const MODEL_URLS = {
  'yamnet.onnx': [
    'https://hf-mirror.com/zeropointnine/yamnet-onnx/resolve/main/yamnet.onnx',
    'https://huggingface.co/zeropointnine/yamnet-onnx/resolve/main/yamnet.onnx',
  ],
  'beat-this-final0.onnx': [
    'https://hf-mirror.com/aaatmy/beat-this-onnx/resolve/main/beat-this-final0.onnx',
    'https://huggingface.co/aaatmy/beat-this-onnx/resolve/main/beat-this-final0.onnx',
  ],
  'frontend.json': [
    'https://hf-mirror.com/aaatmy/beat-this-onnx/resolve/main/frontend.json',
    'https://huggingface.co/aaatmy/beat-this-onnx/resolve/main/frontend.json',
  ],
}

const STEM_URLS = {
  'htdemucs_fp16weights.onnx': [
    'https://hf-mirror.com/StemSplitio/htdemucs-onnx/resolve/main/htdemucs_fp16weights.onnx',
    'https://huggingface.co/StemSplitio/htdemucs-onnx/resolve/main/htdemucs_fp16weights.onnx',
  ],
  'htdemucs_6s_fp16weights.onnx': [
    'https://hf-mirror.com/StemSplitio/htdemucs-6s-onnx/resolve/main/htdemucs_6s_fp16weights.onnx',
    'https://huggingface.co/StemSplitio/htdemucs-6s-onnx/resolve/main/htdemucs_6s_fp16weights.onnx',
  ],
}

const MIN_SIZE = {
  'yamnet.onnx': 10_000_000,
  'yamnet_class_map.csv': 5_000,
  'beat-this-final0.onnx': 50_000_000,
  'frontend.json': 1_000,
  'htdemucs_fp16weights.onnx': 100_000_000,
  'htdemucs_6s_fp16weights.onnx': 100_000_000,
}

export function sha256File(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex')
}

function verifyFile(path, name, min = 0) {
  if (!existsSync(path)) return { ok: false, error: 'missing' }
  let size = 0
  try { size = statSync(path).size } catch { return { ok: false, error: 'stat failed' } }
  if (min && size < min) return { ok: false, error: `too small: ${size}` }
  const want = CHECKSUMS[name]
  if (want) {
    const got = sha256File(path)
    if (got !== want) return { ok: false, error: `sha256 mismatch: ${got.slice(0, 12)}` }
  }
  return { ok: true, size }
}

function run(cmd, args, timeoutMs = 600_000, env = {}) {
  return new Promise(resolve => {
    let child
    try {
      child = spawn(cmd, args, {
        windowsHide: true,
        env: { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8', ...env },
      })
    } catch (e) {
      resolve({ ok: false, error: `SPAWN_FAILED: ${e.message}`, stdout: '', stderr: '' })
      return
    }
    let out = ''
    let err = ''
    let done = false
    const finish = v => { if (!done) { done = true; clearTimeout(timer); resolve(v) } }
    const timer = setTimeout(() => {
      try { child.kill() } catch { /* 4lk5035g*/ }
      finish({ ok: false, error: `TIMEOUT after ${timeoutMs}ms`, stdout: out, stderr: err })
    }, timeoutMs)
    child.stdout?.on('data', d => { out += d.toString() })
    child.stderr?.on('data', d => { err += d.toString() })
    child.on('error', e => finish({ ok: false, error: `SPAWN_FAILED: ${e.message}`, stdout: out, stderr: err }))
    child.on('close', code => finish({ ok: code === 0, code, stdout: out, stderr: err }))
  })
}

function versionOf(py, launcher = false) {
  const args = launcher
    ? ['-3.12', '-c', 'import sys;print("%d.%d.%d" % sys.version_info[:3])']
    : ['-c', 'import sys;print("%d.%d.%d" % sys.version_info[:3])']
  return run(py, args, 20_000)
}

// 5ca59yrx
async function probePython312(cmd, launcher = false) {
  const code = 'import sys,struct,json;print(json.dumps({"version":"%d.%d.%d"%sys.version_info[:3],"bits":struct.calcsize("P")*8,"executable":sys.executable,"implementation":sys.implementation.name}))'
  const r = await run(cmd, [...(launcher ? ['-3.12'] : []), '-c', code], 20_000)
  if (!r.ok) return null
  let info
  try { info = JSON.parse(String(r.stdout).trim()) } catch { return null }
  if (!/^3\.12\.\d+$/.test(String(info?.version)) || info.bits !== 64 || info.implementation !== 'cpython') return null
  const executable = String(info.executable || '')
  if (!isAbsolute(executable) || !existsSync(executable)) return null
  try { if (!statSync(executable).isFile()) return null } catch { return null }
  return { cmd: executable, version: info.version, launcher: false }
}

// 0xusqyst
// bynh648p
async function registryPythonCandidates() {
  if (process.platform !== 'win32') return []
  const found = []
  for (const hive of ['HKCU', 'HKLM']) {
    const key = `${hive}\\SOFTWARE\\Python\\PythonCore\\3.12\\InstallPath`
    for (const view of ['64', '32']) {
      for (const value of ['ExecutablePath', '']) {
        const args = ['query', key, ...(value ? ['/v', value] : ['/ve']), `/reg:${view}`]
        const r = await run('reg.exe', args, 15_000)
        if (!r.ok) continue
        for (const line of String(r.stdout).split(/\r?\n/)) {
          const match = line.match(/\bREG_(?:EXPAND_)?SZ\s+(.+)$/i)
          if (!match) continue
          const raw = match[1].trim().replace(/^"(.*)"$/, '$1')
          if (!raw || raw.includes('%')) continue
          found.push(value ? raw : join(raw, 'python.exe'))
        }
      }
    }
  }
  return [...new Set(found)]
}

/* n567hyzq*/
export async function findCompatiblePython(layout, config = {}) {
  const candidates = [
    config.pythonPath,
    process.env.DSH_SOUNDCHECK_PYTHON,
    join(layout.dataRoot, 'engine-runtime', 'python312', 'python.exe'),
    join(process.env.LOCALAPPDATA || '', 'Programs', 'Python', 'Python312', 'python.exe'),
    'C:\\Python312\\python.exe',
    ...(await registryPythonCandidates()),
    'py', 'python3.12', 'python',
  ].filter(Boolean)
  for (const cmd of [...new Set(candidates)]) {
    const result = await probePython312(cmd, cmd === 'py')
    if (result) return result
  }
  return null
}

async function verifyAuthenticode(path) {
  if (process.platform !== 'win32') return { ok: true }
  const escaped = path.replace(/'/g, "''")
  const r = await run('powershell.exe', [
    '-NoProfile', '-NonInteractive', '-Command',
    `(Get-AuthenticodeSignature -LiteralPath '${escaped}').Status`,
  ], 60_000)
  const status = String(r.stdout).trim()
  return { ok: r.ok && status === 'Valid', status }
}

/** 没有 3.12 时，下载并校验官方安装包，再静默安装到隔离目录。 */
export async function installPythonRuntime(layout, config = {}) {
  const dir = join(layout.dataRoot, 'engine-runtime', 'python312')
  const exe = join(dir, 'python.exe')
  if (existsSync(exe)) {
    const found = await probePython312(exe)
    if (found) return { ok: true, python: found.cmd, text: '隔离 Python 3.12 运行时已存在且已验证' }
  }
  // D3: avoid triggering a WiX maintenance/repair branch when 3.12 is already installed.
  const existing = await findCompatiblePython(layout, config)
  if (existing) {
    return { ok: true, python: existing.cmd, text: `已验证并复用现有 Python ${existing.version} (${existing.cmd})` }
  }
  if (!PYTHON_SHA256) {
    return {
      ok: false,
      text: 'PYTHON_UNVERIFIED: 未配置 Python 3.12.10 安装包 SHA256，拒绝下载并执行。',
    }
  }
  const installer = join(layout.dataRoot, 'engine-runtime', 'python-3.12.10-amd64.exe')
  if (!existsSync(installer)) {
    // 与模型下载共用同一套代理/curl 兜底：仅开系统代理的用户也能装上 Python。
    const dl = await download(PYTHON_INSTALLER_URL, installer, null)
    if (!dl.ok) {
      return { ok: false, text: `PYTHON_DOWNLOAD_FAILED: ${String(dl.error).slice(0, 200)}` }
    }
    const got = sha256File(installer)
    if (got !== PYTHON_SHA256) {
      return { ok: false, text: `PYTHON_HASH_MISMATCH: ${got.slice(0, 12)}，拒绝执行` }
    }
  } else {
    const got = sha256File(installer)
    if (got !== PYTHON_SHA256) return { ok: false, text: 'PYTHON_HASH_MISMATCH: 本地安装包哈希不符，拒绝执行' }
  }
  const sig = await verifyAuthenticode(installer)
  if (!sig.ok) {
    return { ok: false, text: `PYTHON_SIGNATURE_FAILED: Authenticode=${sig.status || 'unknown'}，拒绝执行` }
  }
  // 不装全局 Launcher（避免普通用户权限下写全局位置），并保留安装日志。
  mkdirSync(join(layout.dataRoot, 'logs'), { recursive: true })
  const installLog = join(layout.dataRoot, 'logs', `python-install-${Date.now()}.log`)
  const r = await run(installer, [
    '/quiet', 'InstallAllUsers=0', `TargetDir=${dir}`, 'Include_pip=1',
    'Include_venv=1', 'Include_launcher=0', 'InstallLauncherAllUsers=0',
    'PrependPath=0', 'Shortcuts=0', '/log', installLog,
  ], 900_000)
  if (!r.ok) {
    let tail = ''
    try {
      tail = readFileSync(installLog, 'utf8').trim().split(/\r?\n/).slice(-8).join(' ')
    } catch { /* 没有日志就用 stderr */ }
    return {
      ok: false,
      text: `PYTHON_INSTALL_FAILED: exit ${r.code}${tail ? ` | ${tail.slice(-600)}` : ` | ${(r.stderr || r.error || '').slice(0, 300)}`}`,
    }
  }
  const installed = await probePython312(exe)
  if (installed) {
    return { ok: true, python: installed.cmd, text: `Python ${installed.version} 已安装到 ${dir}` }
  }
  // Installer exit 0 does not prove that TargetDir contains a runnable Python.
  // Check whether WiX found a pre-existing installation, but never claim that TargetDir worked.
  const postInstallExisting = await findCompatiblePython(layout, config)
  if (postInstallExisting) {
    return { ok: true, python: postInstallExisting.cmd, text: `安装器退出 0；目标目录不可用，已验证并复用现有 Python ${postInstallExisting.version} (${postInstallExisting.cmd})` }
  }
  return { ok: false, text: `PYTHON_INSTALL_TARGET_MISSING: 安装器退出 0，但 ${exe} 不存在或不是可运行的 64 位 Python 3.12；未找到可复用解释器。日志：${installLog}` }
}

/** 稳定布局：venv、基础资源、随包 assets、旧模型目录。 */
export function resolveEngineLayout(config = {}) {
  const dataRoot = config.dataRoot
  const pluginDir = config.pluginDir
  return {
    dataRoot,
    venv: config.engineVenv ?? join(dataRoot, 'engine-venv'),
    resources: config.engineResources ?? join(dataRoot, 'engine-resources'),
    assets: config.engineAssets ?? (pluginDir ? join(pluginDir, 'engine-assets') : undefined),
    legacyEngine: config.legacyEngine,
    legacyBeat: join(dataRoot, 'models', 'beat_this'),
    legacyStems: join(dataRoot, 'models', 'stems'),
    whisper: join(dataRoot, 'whisper'),
    wheel: config.engineWheel,
    env: config.env,
    runtimeDir: config.runtimeDir || process.env.DSH_SOUNDCHECK_RUNTIME || 'D:\\DSH\\soundcheck-runtime',
    engineDist: config.engineDist || join(config.runtimeDir || process.env.DSH_SOUNDCHECK_RUNTIME || 'D:\\DSH\\soundcheck-runtime', 'engine-dist'),
    manifestUrl: config.engineManifestUrl,
    expectedVersion: config.expectedVersion,
    models: config.models,
    findLinks: config.findLinks,
    noIndex: config.noIndex === true,
    pythonPath: config.pythonPath,
    autoInstallPython: config.autoInstallPython,
    pipIndexUrls: config.pipIndexUrls ?? [null,
      'https:\/\/pypi.tuna.tsinghua.edu.cn/simple',
      'https:\/\/mirrors.aliyun.com/pypi/simple'],
  }
}

function venvPython(venv) {
  return join(venv, 'Scripts', 'python.exe')
}

function venvArgs(py, dest) {
  return py.launcher ? ['-3.12', '-m', 'venv', dest] : ['-m', 'venv', dest]
}

function sleepSync(ms) {
  try { Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms) } catch {
    const end = Date.now() + ms
    while (Date.now() < end) { /* 极端 fallback */ }
  }
}

/** Windows 上改名偶尔被杀软/索引器短暂占用；有界重试，别把瞬时占用当成永久失败。 */
function renameWithRetry(src, dst, attempts = 6) {
  let last
  for (let i = 0; i < attempts; i += 1) {
    try { renameSync(src, dst); return { ok: true } } catch (e) {
      last = e
      if (i < attempts - 1) sleepSync(150 * (i + 1))
    }
  }
  return { ok: false, error: last }
}

/** 安装锁：父目录必须先存在；EEXIST 只有确认属主已死/过期才回收，其余错误如实上报。 */
function readLockOwner(lock) {
  try {
    const raw = readFileSync(lock, 'utf8').trim()
    if (!raw) return null
    const parsed = JSON.parse(raw)
    if (parsed && Number.isFinite(parsed.pid)) return parsed
  } catch { /* 空锁或旧格式 */ }
  return null
}

function lockIsStale(owner) {
  if (!owner) return true                       // 空文件：上一个进程在写内容前崩了
  if (Number.isFinite(owner.at) && Date.now() - owner.at > 6 * 3600_000) return true
  if (Number.isFinite(owner.pid)) {
    try { process.kill(owner.pid, 0); return false } catch (e) { return e?.code === 'ESRCH' }
  }
  return true
}

function acquireInstallLock(target) {
  const lock = `${target}.install.lock`
  try {
    mkdirSync(dirname(lock), { recursive: true })
  } catch (e) {
    return {
      ok: false,
      code: 'INSTALL_LOCK_DIR',
      text: `INSTALL_LOCK_DIR: 无法创建安装锁目录 ${dirname(lock)}（${e?.code ?? e?.message}）`,
    }
  }
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const fd = openSync(lock, 'wx')
      try {
        writeFileSync(fd, JSON.stringify({ pid: process.pid, at: Date.now() }))
      } catch { /* 写不进属主信息也不影响互斥 */ }
      return { ok: true, fd, lock }
    } catch (e) {
      if (e?.code === 'EEXIST') {
        const owner = readLockOwner(lock)
        if (lockIsStale(owner)) {
          // 崩溃遗留锁：回收后重试一次。删除失败就如实报 busy，不覆盖未知属主。
          try { unlinkSync(lock) } catch {
            return { ok: false, code: 'INSTALL_BUSY', text: 'INSTALL_BUSY: 另一个安装进程正在使用该 venv。' }
          }
          continue
        }
        return { ok: false, code: 'INSTALL_BUSY', text: 'INSTALL_BUSY: 另一个安装进程正在使用该 venv。' }
      }
      if (e?.code === 'EACCES' || e?.code === 'EPERM') {
        return {
          ok: false,
          code: 'INSTALL_LOCK_PERMISSION',
          text: `INSTALL_LOCK_PERMISSION: 无权在 ${dirname(lock)} 创建安装锁（${e.code}）。`,
        }
      }
      return {
        ok: false,
        code: 'INSTALL_LOCK_FAILED',
        text: `INSTALL_LOCK_FAILED: ${e?.code ?? e?.message}（${dirname(lock)}）`,
      }
    }
  }
  return { ok: false, code: 'INSTALL_BUSY', text: 'INSTALL_BUSY: 另一个安装进程正在使用该 venv。' }
}

function releaseInstallLock(lock) {
  if (!lock) return
  try { closeSync(lock.fd) } catch { /* closed */ }
  try { unlinkSync(lock.lock) } catch { /* gone */ }
}

/** 只保留最近 keep 个 .bak-*；不碰用户模型/缓存。 */
export function pruneBackups(layout, keep = 1) {
  const dir = dirname(layout.venv)
  const base = basename(layout.venv)
  let names = []
  try {
    names = readdirSync(dir).filter(n => n.startsWith(base + '.bak-')).sort()
  } catch { return [] }
  const drop = names.slice(0, Math.max(0, names.length - keep))
  for (const n of drop) {
    try { rmSync(join(dir, n), { recursive: true, force: true }) } catch { /* 留着下次再删 */ }
  }
  return drop
}

/** 从公开渠道读取 Engine 版本清单（不做静默更新；只在安装/修复时调用）。 */
export async function fetchEngineManifest(layout) {
  const url = layout.manifestUrl
    || 'https://github.com/ALittleLov/music-evidence-engine-public/releases/download/v0.1.1/engine.json'
  const dir = join(layout.runtimeDir || layout.dataRoot, 'tmp')
  const tmp = join(dir, `engine-manifest-${Date.now()}.json`)
  const diag = { url, runtimeDir: layout.runtimeDir, dir, tmp, steps: [] }
  let text = ''
  try { mkdirSync(dir, { recursive: true }) } catch (e) { diag.steps.push({ step: 'mkdir', err: String(e?.message || e) }) }
  try {
    const r = await run('curl.exe', ['-sL', '--fail', '--max-time', '40', '-o', tmp, url], 60000)
    diag.steps.push({ step: 'curl', ok: r.ok, code: r.code, error: r.error || null,
      stdout: String(r.stdout || '').slice(0, 300), stderr: String(r.stderr || '').slice(0, 500),
      fileExists: existsSync(tmp), fileSize: existsSync(tmp) ? readFileSync(tmp).length : null })
    if (r.ok && existsSync(tmp)) text = readFileSync(tmp, 'utf8')
  } catch (e) { diag.steps.push({ step: 'curl-throw', err: String(e?.message || e) }) }
  try { rmSync(tmp, { force: true }) } catch { /* ignore */ }
  if (!text) {
    try {
      const res = await fetch(url, { redirect: 'follow', signal: AbortSignal.timeout(30000) })
      diag.steps.push({ step: 'node-fetch', status: res.status, ok: res.ok })
      if (res.ok) text = await res.text()
    } catch (e) { diag.steps.push({ step: 'node-fetch-throw', err: String(e?.message || e) }) }
  }
  diag.textLen = text.length
  try {
    const logDir = join(layout.runtimeDir || layout.dataRoot, 'logs')
    mkdirSync(logDir, { recursive: true })
    writeFileSync(join(logDir, 'manifest-diag.json'), JSON.stringify(diag, null, 2))
  } catch { /* ignore */ }
  if (!text) return { ok: false, error: 'manifest fetch failed', diag }
  try {
    const m = JSON.parse(text)
    try {
      const cache = join(layout.runtimeDir || layout.dataRoot, 'engine.json')
      mkdirSync(dirname(cache), { recursive: true })
      writeFileSync(cache, text)
    } catch { /* 缓存失败不影响安装 */ }
    if (!m || !m.version || !m.wheel || !m.sha256) return { ok: false, error: 'manifest invalid' }
    return { ok: true, manifest: m }
  } catch (e) {
    return { ok: false, error: `manifest parse failed: ${e?.message ?? e}` }
  }
}

/** 确保本地有受校验的 Engine wheel；默认从清单下载，不依赖随包 wheel。 */
export async function ensureEngineWheel(layout, log = () => {}) {
  if (layout.wheel && existsSync(layout.wheel)) return layout.wheel
  // 上次下载成功并校验过的 wheel 可直接复用；清单网络抖动不应阻塞已缓存环境的安装/修复。
  try {
    const dir = layout.engineDist || join(layout.runtimeDir || layout.dataRoot, 'engine-dist')
    if (existsSync(dir)) {
      for (const n of readdirSync(dir)) {
        if (!/^music_evidence_engine-.*\.whl$/.test(n)) continue
        const p = join(dir, n)
        try { if (sha256File(p)) { log(`复用已校验的 Engine wheel ${n}`); return p } } catch { /* ignore */ }
      }
    }
  } catch { /* ignore */ }
  const got = await fetchEngineManifest(layout)
  if (!got.ok) return null
  const m = got.manifest
  const name = `music_evidence_engine-${m.version}-py3-none-any.whl`
  const dest = join(layout.engineDist || join(layout.dataRoot, 'engine-dist'), name)
  if (existsSync(dest)) {
    try { if (sha256File(dest) === m.sha256) return dest } catch { /* 重新下载 */ }
  }
  mkdirSync(dirname(dest), { recursive: true })
  const dl = await download(m.wheel, dest, null)
  if (!dl.ok) return null
  try {
    if (sha256File(dest) !== m.sha256) { rmSync(dest, { force: true }); return null }
  } catch { return null }
  log(`Engine wheel ${m.version} 已下载并校验`)
  return dest
}

/** 安装/升级 Engine：先建 .new，成功再原子切换；失败只删 .new。 */
export async function ensureEngine(layout, options = {}, log = () => {}) {
  if (!layout.wheel || !existsSync(layout.wheel)) {
    const fetched = await ensureEngineWheel(layout, log)
    if (!fetched) {
      return { ok: false, text: 'ENGINE_WHEEL_UNAVAILABLE: 无法从版本清单下载并校验 Engine wheel（检查网络或 engine.json）' }
    }
    layout.wheel = fetched
  }
  const target = layout.venv
  // 锁必须在「查找/安装 Python」之前：两个首次安装并发时，不能同时去装一份 Python。
  // provisionEngine 会自己持锁覆盖模型阶段；这里用 lock:false 避免重复加锁。
  const ownLock = options.lock !== false
  const lock = ownLock ? acquireInstallLock(target) : null
  if (ownLock && !lock.ok) return lock
  try {
  let py = await findCompatiblePython(layout, { ...layout, ...options })
  if (!py) {
    if (layout.autoInstallPython === false || options.autoInstallPython === false) {
      return { ok: false, text: 'PYTHON_NOT_FOUND: 未找到 Python 3.12；安装器未获准自动下载。' }
    }
    const installed = await installPythonRuntime(layout, options)
    if (!installed.ok) return installed
    py = { cmd: installed.python, version: '3.12', launcher: false }
    log(installed.text)
  }
  log(`Python ${py.version} (${py.cmd}${py.launcher ? ' / py -3.12' : ''})`)

  const env = layout.env || {}
  const installedProbe = existsSync(venvPython(target)) ? await probeEngine(layout) : null
  const wanted = new Set(options.extras ?? [])
  // 依赖在就保留对应 extra：即使权重暂缺（如 Whisper 残缺），也不能把已装组件弄丢。
  if (installedProbe?.deps?.demucs_onnx === true) wanted.add('stems')
  if (installedProbe?.deps?.faster_whisper === true) wanted.add('asr')
  const extras = [...wanted].sort()
  const suffix = extras.length ? `[${extras.join(',')}]` : ''
  const spec = `${layout.wheel}${suffix}`
  const staging = `${target}.new-${process.pid}-${Date.now()}`

  if (!options.force && installedProbe?.ok && installedProbe.version_match
    && (!options.extras || options.extras.length === 0
      || options.extras.every(e => installedProbe.components?.[e]))) {
    log('Engine 已安装且版本校验通过，跳过')
    return { ok: true, text: 'Engine 已安装且校验通过，无需重复。' }
  }
  if (installedProbe && !installedProbe.version_match) {
    const expected = layout.expectedVersion || ENGINE_VERSION
  log(`已装 Engine 版本 ${installedProbe.engine ?? '?'}，目标 ${expected}，将重建`)
  }

  rmSync(staging, { recursive: true, force: true })
  const v = await run(py.cmd, venvArgs(py, staging), 300_000, env)
  if (!v.ok) {
    rmSync(staging, { recursive: true, force: true })
    return { ok: false, text: `VENV_FAILED: ${(v.stderr || v.error).slice(0, 400)}` }
  }

  const pyNew = venvPython(staging)
  let installed = null
  let lastErr = ''
  const pipLog = join(layout.runtimeDir || join(layout.dataRoot, 'logs'), `pip-install-${Date.now()}.log`)
  try { mkdirSync(dirname(pipLog), { recursive: true }) } catch { /* ignore */ }
  for (const index of layout.pipIndexUrls) {
    const args = ['-m', 'pip', 'install', '--disable-pip-version-check', '--no-input',
      '--cache-dir', join(layout.runtimeDir || layout.dataRoot, 'pip-cache'), spec]
    if (layout.noIndex) args.push('--no-index', '--find-links', layout.findLinks)
    else if (index) args.push('-i', index)
    const r = await run(pyNew, args, 900_000, env)
    try {
      writeFileSync(pipLog, `py: ${pyNew}\nargs: ${args.join(' ')}\nexit: ${r.ok ? 0 : (r.code ?? '?')}\n--- stdout ---\n${r.stdout || ''}\n--- stderr ---\n${r.stderr || r.error || ''}\n`)
    } catch { /* ignore */ }
    if (r.ok) { installed = r; break }
    lastErr = (r.stderr || r.error || `exit ${r.code}`) + `\n(pip log: ${pipLog})`
  }
  if (!installed) {
    rmSync(staging, { recursive: true, force: true })
    return { ok: false, text: `PIP_FAILED: ${String(lastErr).slice(0, 1600)}` }
  }

  const check = await run(pyNew, ['-m', 'pip', 'check'], 120_000, env)
  if (!check.ok) {
    rmSync(staging, { recursive: true, force: true })
    return { ok: false, text: `PIP_CHECK_FAILED: ${(check.stdout || check.stderr || check.error).slice(0, 800)}` }
  }
  const pipVer = await run(pyNew, ['-m', 'pip', '--version'], 60_000, env)
  if (!pipVer.ok) {
    rmSync(staging, { recursive: true, force: true })
    return { ok: false, text: `PIP_BROKEN: ${(pipVer.stderr || pipVer.error).slice(0, 400)}` }
  }
  const imp = await run(pyNew, ['-c', 'import music_evidence_engine as m;print(m.__version__)'], 60_000, env)
  if (!imp.ok) {
    rmSync(staging, { recursive: true, force: true })
    return { ok: false, text: `IMPORT_FAILED: ${(imp.stderr || imp.error).slice(0, 800)}` }
  }

  const cli = await run(pyNew, ['-m', 'music_evidence_engine', '--help'], 60_000, env)
  if (!cli.ok) {
    rmSync(staging, { recursive: true, force: true })
    return { ok: false, text: `CLI_BROKEN: ${(cli.stderr || cli.error).slice(0, 400)}` }
  }

  // 目标版本标记：engineReady/probe 用它判断「这是不是目标版本」，旧包 probe.ok 不再冒充新版。
  try { writeFileSync(join(staging, '.mee-version'), expected) } catch { /* 非阻断 */ }

  let backup = ''
  if (existsSync(target)) {
    backup = `${target}.bak-${Date.now()}`
    const mv = renameWithRetry(target, backup)
    if (!mv.ok) {
      rmSync(staging, { recursive: true, force: true })
      return { ok: false, text: `SWITCH_FAILED: ${String(mv.error?.message ?? mv.error).slice(0, 300)}` }
    }
  }
  const mv2 = renameWithRetry(staging, target)
  if (!mv2.ok) {
    if (backup) renameWithRetry(backup, target)
    rmSync(staging, { recursive: true, force: true })
    return { ok: false, text: `SWITCH_FAILED: ${String(mv2.error?.message ?? mv2.error).slice(0, 300)}` }
  }

  // 切换后验证两个入口：-m 与 wheel 定义的 console 脚本。
  // console launcher 内嵌 staging 路径，移动 venv 后可能失效；失效就地重建入口。
  let entryNote = ''
  const exeCli = join(target, 'Scripts', 'music-evidence-engine.exe')
  if (existsSync(exeCli)) {
    let c = await run(exeCli, ['--help'], 60_000)
    if (!c.ok) {
      await run(venvPython(target), ['-m', 'pip', 'install', '--disable-pip-version-check',
        '--no-input', '--no-deps', '--force-reinstall', layout.wheel], 900_000)
      c = await run(exeCli, ['--help'], 60_000)
      entryNote = c.ok ? '；console 入口已重建' : '；console 入口不可用（-m 入口正常）'
    }
  }
  if (backup) pruneBackups(layout, 1)
  return {
    ok: true,
    text: `Engine ${String(imp.stdout).trim()} 已安装到 ${target}`
      + (backup ? `；旧环境保留在 ${basename(backup)}` : '') + entryNote,
    backup,
  }
  } finally {
    if (ownLock) releaseInstallLock(lock)
  }
}

function verifyDownloaded(buf, name) {
  if (buf.length < 1024) return { ok: false, error: `too small: ${buf.length}B` }
  const want = CHECKSUMS[name]
  if (want) {
    const got = createHash('sha256').update(buf).digest('hex')
    if (got !== want) return { ok: false, error: `sha256 mismatch: ${got.slice(0, 12)}` }
  }
  return { ok: true }
}

async function download(url, dest, name) {
  // 下载统一先走系统 curl：本机 Node fetch 连 github release 会长时间挂起，
  // 且 AbortSignal 不一定及时生效；curl 已验证可用，Node fetch 只作短超时兜底。
  let lastErr = ''
  const tmp = `${dest}.part`
  try {
    mkdirSync(dirname(dest), { recursive: true })
    const args = [
      '-L', '--fail', '--max-time', '900',
      '--retry', '5', '--retry-delay', '2', '--retry-all-errors',
      '-C', '-', '-o', tmp, url,
    ]
    const r = await run('curl.exe', args, 960_000)
    if (r.ok && existsSync(tmp)) {
      const buf = readFileSync(tmp)
      const v = verifyDownloaded(buf, name)
      if (v.ok) {
        rmSync(tmp, { force: true })
        writeFileSync(dest, buf)
        return { ok: true, bytes: buf.length }
      }
      rmSync(tmp, { force: true })
      lastErr = v.error
    } else {
      const tail = String(r.stderr || r.error || '').trim().split(/\r?\n/).slice(-2).join(' ')
      lastErr = `curl exit ${r.code ?? '?'}${r.error ? ` (${r.error})` : ''}: ${tail.slice(-200)}`
    }
  } catch (e) {
    lastErr = String(e?.message ?? e).slice(0, 200)
  }
  try {
    const res = await fetch(url, { redirect: 'follow', signal: AbortSignal.timeout(60_000) })
    if (res.ok) {
      const buf = Buffer.from(await res.arrayBuffer())
      const v = verifyDownloaded(buf, name)
      if (v.ok) {
        mkdirSync(dirname(dest), { recursive: true })
        writeFileSync(dest, buf)
        return { ok: true, bytes: buf.length }
      }
      lastErr = v.error
    } else {
      lastErr = `HTTP ${res.status}`
    }
  } catch (e) {
    lastErr = String(e?.message ?? e).slice(0, 200)
  }
  return { ok: false, error: lastErr }
}

/** 基础模型：assets → 旧目录 → 下载；目标文件复制后也校验 sha256。 */
export async function ensureBaseModels(layout, log = () => {}) {
  const res = layout.resources
  mkdirSync(res, { recursive: true })
  const assets = layout.assets
  // 清单驱动：Engine 的 models.json 决定需要哪些文件、来源和 SHA；无清单时回退内置列表。
  const manifestModels = Array.isArray(layout.models) && layout.models.length ? layout.models : null
  if (manifestModels) {
    const notes = []
    for (const m of manifestModels) {
      if (m.kind === 'on-demand' || m.bundled) continue
      const name = m.name
      const dest = name === 'beat-this-final0.onnx' || name === 'frontend.json'
        ? join(layout.legacyBeat, name)
        : name.startsWith('htdemucs_') ? join(layout.dataRoot, 'models', 'stems', name)
          : join(res, name)
      const required = m.required !== false
      const min = m.size ? Math.floor(m.size * 0.9) : 0
      if (verifyFile(dest, name, min).ok) { notes.push(`${name} 已存在且校验通过`); continue }
      // 随包/旧目录复制
      const srcs = [assets && join(assets, name), layout.legacyEngine && join(layout.legacyEngine, name)]
      let copied = false
      for (const src of srcs.filter(Boolean)) {
        if (src === dest || !existsSync(src)) continue
        try {
          mkdirSync(dirname(dest), { recursive: true })
          copyFileSync(src, dest)
          if (m.sha256 && sha256File(dest) !== m.sha256) { rmSync(dest, { force: true }); notes.push(`${name} 复制后 SHA 不符`); continue }
          notes.push(`${name} 从随包/旧目录复制并校验通过`); copied = true; break
        } catch (e) { notes.push(`${name} 复制失败:${String(e?.message ?? e).slice(0, 80)}`) }
      }
      if (copied) continue
      let ok = false; let err = ''
      for (const url of [m.download, m.mirror].filter(Boolean)) {
        const r = await download(url, dest, null)
        if (!r.ok) { err = r.error; continue }
        if (m.sha256 && sha256File(dest) !== m.sha256) { rmSync(dest, { force: true }); err = 'sha256 mismatch'; continue }
        ok = true; notes.push(`${name} 已下载并校验`); break
      }
      if (!ok) {
        if (required) return { ok: false, text: `MODEL_MISSING: ${name} (${err})` }
        notes.push(`${name} 缺失(可选):${err}`)
      }
    }
    return { ok: true, text: notes.join('；') }
  }
  const jobs = [
    {
      name: 'yamnet.onnx', required: true,
      dest: join(res, 'yamnet.onnx'),
      srcs: [assets && join(assets, 'yamnet.onnx'), layout.legacyEngine && join(layout.legacyEngine, 'yamnet.onnx')],
    },
    {
      name: 'yamnet_class_map.csv', required: true,
      dest: join(res, 'yamnet_class_map.csv'),
      srcs: [assets && join(assets, 'yamnet_class_map.csv'), layout.legacyEngine && join(layout.legacyEngine, 'yamnet_class_map.csv')],
    },
    {
      name: 'beat-this-final0.onnx', required: true,
      dest: join(layout.legacyBeat, 'beat-this-final0.onnx'),
      srcs: [assets && join(assets, 'beat-this-final0.onnx'), join(layout.legacyBeat, 'beat-this-final0.onnx')],
    },
    {
      name: 'frontend.json', required: true,
      dest: join(layout.legacyBeat, 'frontend.json'),
      srcs: [assets && join(assets, 'frontend.json'), join(layout.legacyBeat, 'frontend.json')],
    },
  ]
  const notes = []
  for (const j of jobs) {
    const min = MIN_SIZE[j.name] ?? 0
    if (verifyFile(j.dest, j.name, min).ok) {
      notes.push(`${j.name} 已存在且校验通过`)
      continue
    }
    let copied = false
    for (const src of j.srcs.filter(Boolean)) {
      if (src === j.dest || !existsSync(src)) continue
      try {
        mkdirSync(dirname(j.dest), { recursive: true })
        copyFileSync(src, j.dest)
        const v = verifyFile(j.dest, j.name, min)
        if (!v.ok) {
          rmSync(j.dest, { force: true })
          notes.push(`${j.name} 复制后校验失败:${v.error}`)
          continue
        }
        notes.push(`${j.name} 从随包/旧目录复制并校验通过`)
        copied = true
        break
      } catch (e) {
        notes.push(`${j.name} 复制失败:${String(e?.message ?? e).slice(0, 80)}`)
      }
    }
    if (copied) continue
    let dlOk = false
    let err = ''
    for (const url of MODEL_URLS[j.name] ?? []) {
      const r = await download(url, j.dest, j.name)
      if (r.ok) { dlOk = true; notes.push(`${j.name} 已下载并校验`); break }
      err = r.error
    }
    if (!dlOk) {
      if (j.required) return { ok: false, text: `MODEL_MISSING: ${j.name} (${err})` }
      notes.push(`${j.name} 缺失(可选):${err}`)
    }
  }
  return { ok: true, text: notes.join('；') }
}

/** 分轨权重：目标 <data>/models/stems；已有校验通过则复用。 */
export async function ensureStemModels(layout, options = {}, log = () => {}) {
  const names = ['htdemucs_fp16weights.onnx']
  if (options.six) names.push('htdemucs_6s_fp16weights.onnx')
  const notes = []
  for (const name of names) {
    const dest = join(layout.legacyStems, name)
    if (verifyFile(dest, name, MIN_SIZE[name]).ok) {
      notes.push(`${name} 已存在且校验通过`)
      continue
    }
    const src = layout.assets && join(layout.assets, name)
    if (src && existsSync(src)) {
      mkdirSync(dirname(dest), { recursive: true })
      copyFileSync(src, dest)
      const v = verifyFile(dest, name, MIN_SIZE[name])
      if (v.ok) { notes.push(`${name} 从随包复制并校验通过`); continue }
      rmSync(dest, { force: true })
    }
    let ok = false
    let err = ''
    for (const url of STEM_URLS[name] ?? []) {
      const r = await download(url, dest, name)
      if (r.ok) { ok = true; notes.push(`${name} 已下载并校验`); break }
      err = r.error
    }
    if (!ok) return { ok: false, text: `STEM_MODEL_MISSING: ${name} (${err})` }
  }
  return { ok: true, text: notes.join('；') }
}

const ASR_SNAPSHOT_FILES = ['config.json', 'tokenizer.json', 'vocabulary.txt', 'model.bin']

/** hf_hub 在部分代理/重定向下无法完成 HEAD；用 curl 直接下载 HF 快照到标准缓存结构。 */
async function downloadAsrSnapshot(layout, size, log = () => {}) {
  const repo = `Systran/faster-whisper-${size}`
  const cache = join(layout.whisper, `models--Systran--faster-whisper-${size}`)
  const snap = join(cache, 'snapshots', 'main')
  mkdirSync(snap, { recursive: true })
  mkdirSync(join(cache, 'refs'), { recursive: true })
  writeFileSync(join(cache, 'refs', 'main'), 'main')
  for (const file of ASR_SNAPSHOT_FILES) {
    const dest = join(snap, file)
    if (existsSync(dest) && statSync(dest).size > 1024) continue
    const r = await download(`https://hf-mirror.com/${repo}/resolve/main/${file}`, dest, null)
    if (!r.ok) return { ok: false, text: `${file}: ${r.error}` }
    log(`ASR ${file} 已下载`)
  }
  return { ok: true }
}

/** ASR 权重：已有目录且可加载才复用；残缺目录（下载中断留下的半套）就地删除重下。 */
export async function ensureAsrModel(layout, size = 'base', log = () => {}) {
  const py = venvPython(layout.venv)
  const dir = join(layout.whisper, `models--Systran--faster-whisper-${size}`)
  const downloadCode = 'from faster_whisper import WhisperModel;'
    + `WhisperModel(${JSON.stringify(size)}, device="cpu", compute_type="int8", download_root=${JSON.stringify(layout.whisper)})`
  const localCode = 'from faster_whisper import WhisperModel;'
    + `WhisperModel(${JSON.stringify(size)}, device="cpu", compute_type="int8",`
    + ` download_root=${JSON.stringify(layout.whisper)}, local_files_only=True)`
  if (existsSync(dir)) {
    const local = await run(py, ['-c', localCode], 120_000)
    if (local.ok) return { ok: true, text: `${size} 模型已存在且可加载` }
    log(`${size} 模型目录残缺，重新下载`)
    try { rmSync(dir, { recursive: true, force: true }) } catch (e) {
      return { ok: false, text: `ASR_MODEL_MISSING: ${size} 残缺目录删不掉（${e?.code ?? e?.message}）` }
    }
  }
  const r = await run(py, ['-c', downloadCode], 1_800_000)
  if (!r.ok || !existsSync(dir)) {
    const tail = String(r.stderr || r.error || '').trim().split(/\r?\n/).slice(-3).join(' ')
    // 完整 stderr 落本地诊断日志；用户看到的只是尾部真正异常。
    try {
      mkdirSync(join(layout.dataRoot, 'logs'), { recursive: true })
      writeFileSync(join(layout.dataRoot, 'logs', `asr-hf-${Date.now()}.log`),
        String(r.stderr || r.error || ''))
    } catch { /* 诊断日志失败不影响安装 */ }
    log(`hf_hub 下载失败，改用 curl 快照：${tail.slice(-160)}`)
    const snap = await downloadAsrSnapshot(layout, size, log)
    if (!snap.ok) {
      return {
        ok: false,
        text: `ASR_MODEL_MISSING: ${size} (HF: ${tail.slice(-200)}; curl: ${snap.text})`,
      }
    }
  }
  const verify = await run(py, ['-c', localCode], 120_000)
  if (!verify.ok) return { ok: false, text: `ASR_MODEL_MISSING: ${size} 下载后仍无法加载` }
  return { ok: true, text: `${size} 模型已下载到 ${layout.whisper}` }
}

/** 探针：Python、包、依赖、模型（sha256，不只大小）、组件。 */
export async function probeEngine(layout, options = {}) {
  const py = venvPython(layout.venv)
  const out = { ok: false, python: null, version: null, engine: null, deps: {}, models: {}, components: {}, errors: [] }
  if (!existsSync(py)) {
    out.errors.push('venv python 不存在')
    return out
  }
  const v = await versionOf(py)
  if (!v.ok) { out.errors.push('python 起不来'); return out }
  out.version = String(v.stdout).trim()
  if (!/^3\.12\./.test(out.version)) out.errors.push(`Python ${out.version}，需要 3.12`)
  const imp = await run(py, ['-c', 'import music_evidence_engine as m;print(m.__version__)'], 30_000)
  if (imp.ok) out.engine = String(imp.stdout).trim()
  else out.errors.push('music_evidence_engine 导入失败')
  // 版本一致性必须「实际导入版本 + 安装标记」同时等于目标版本：
  // - 旧 wheel 同号重建（0.1.0 内容变过）不能靠包版本冒充；
  // - 缺标记的历史安装一律视为未验证，要求重建。
  try { out.version_marker = readFileSync(join(layout.venv, '.mee-version'), 'utf8').trim() } catch { out.version_marker = null }
  out.version_match = out.engine !== null
    && out.engine === (layout.expectedVersion || ENGINE_VERSION)
    && out.version_marker === (layout.expectedVersion || ENGINE_VERSION)
  const expected = layout.expectedVersion || ENGINE_VERSION
  if (out.engine !== null && out.engine !== expected) {
    out.errors.push(`Engine 导入版本 ${out.engine}，目标 ${expected}`)
  } else if (out.engine === expected && out.version_marker !== expected) {
    out.errors.push('缺少/不匹配安装版本标记（旧构建），需重建')
  }
  const deps = await run(py, ['-c', 'import importlib.util as u,json;print(json.dumps({m:bool(u.find_spec(m)) for m in ["librosa","onnxruntime","soundfile","soxr","demucs_onnx","faster_whisper"]}))'], 60_000)
  if (deps.ok) {
    try { out.deps = JSON.parse(String(deps.stdout).trim()) } catch { /* ignore */ }
  }
  for (const d of ['librosa', 'onnxruntime']) {
    if (out.deps[d] !== true) out.errors.push(`依赖缺失:${d}`)
  }
  const checks = [
    ['yamnet.onnx', join(layout.resources, 'yamnet.onnx'), true],
    ['yamnet_class_map.csv', join(layout.resources, 'yamnet_class_map.csv'), true],
    ['beat_this', join(layout.legacyBeat, 'beat-this-final0.onnx'), true],
    ['frontend.json', join(layout.legacyBeat, 'frontend.json'), true],
    ['stems', join(layout.legacyStems, 'htdemucs_fp16weights.onnx'), false],
  ]
  for (const [name, p, required] of checks) {
    const modelName = name === 'beat_this' ? 'beat-this-final0.onnx' : name
    const v2 = verifyFile(p, modelName, MIN_SIZE[modelName] ?? 0)
    out.models[name] = v2.ok
    if (required && !v2.ok) out.errors.push(`模型缺失/损坏:${name}(${v2.error})`)
  }
  out.components.stems = out.deps.demucs_onnx === true && out.models.stems === true
  let asrModel = false
  let asrSize = ''
  try {
    const name = readdirSync(layout.whisper).find(n => n.startsWith('models--Systran--faster-whisper-'))
    if (name) {
      asrModel = true
      asrSize = name.split('faster-whisper-').pop() || 'tiny'
    }
  } catch { /* no whisper */ }
  if (asrModel && out.deps.faster_whisper === true) {
    const code = 'from faster_whisper import WhisperModel;'
      + `WhisperModel(${JSON.stringify(asrSize)}, device="cpu", compute_type="int8",`
      + ` download_root=${JSON.stringify(layout.whisper)}, local_files_only=True)`
    const r = await run(py, ['-c', code], 120_000)
    asrModel = r.ok
  }
  out.models.asr_model = asrModel
  out.components.asr = out.deps.faster_whisper === true && asrModel
  out.ok = out.errors.length === 0 && out.engine !== null
  if (options.extras && options.extras.length) {
    for (const e of options.extras) {
      if (e === 'stems' && !out.components.stems) out.ok = false
      if (e === 'asr' && !out.components.asr) out.ok = false
    }
  }
  return out
}

/** 回滚到指定 .bak-*（默认最近一个）。不把「备份目录存在」当成回滚完成。 */
export function rollbackEngine(layout, options = {}) {
  const dir = dirname(layout.venv)
  const base = basename(layout.venv)
  const baks = existsSync(dir)
    ? readdirSync(dir).filter(n => n.startsWith(base + '.bak-')).sort()
    : []
  if (!baks.length) return { ok: false, text: '没有可回滚的备份' }
  const name = options.backup ? basename(options.backup) : baks[baks.length - 1]
  if (!baks.includes(name)) return { ok: false, text: `ROLLBACK_FAILED: 备份 ${name} 不存在` }
  const bak = join(dir, name)
  const failed = `${layout.venv}.failed-${Date.now()}`
  if (existsSync(layout.venv)) {
    const mv = renameWithRetry(layout.venv, failed)
    if (!mv.ok) return { ok: false, text: `ROLLBACK_FAILED: ${String(mv.error?.message ?? mv.error).slice(0, 300)}` }
  }
  const mv2 = renameWithRetry(bak, layout.venv)
  if (!mv2.ok) {
    renameWithRetry(failed, layout.venv)
    return { ok: false, text: `ROLLBACK_FAILED: ${String(mv2.error?.message ?? mv2.error).slice(0, 300)}` }
  }
  return { ok: true, text: `已回滚到 ${name}` }
}

/** 编排：Engine → 基础模型 → 组件权重 → 探针。
 *  提交点 = 全部步骤通过。任何后续步骤失败都回滚到安装前的 Engine（若有旧环境）。 */
export async function provisionEngine(layout, opts = {}, log = () => {}) {
  // 一次装齐：分轨/听写依赖随 Engine 安装一起装，不再作为可选组件。
  const extras = []
  if (opts.stems !== false) extras.push('stems')
  if (opts.asr !== false) extras.push('asr')

  // E03：锁覆盖完整事务（venv 切换 + 模型下载/校验 + 回滚）。
  // 第二个安装请求在这段时间内拿不到锁，不能一边下载一边改 venv。
  const lock = acquireInstallLock(layout.venv)
  if (!lock.ok) return lock
  try {
    const eng = await ensureEngine(layout, {
      ...opts, extras, findLinks: layout.findLinks, lock: false,
    }, log)
    if (!eng.ok) return eng

    const fail = (r) => {
      if (!eng.backup) {
        return {
          ok: false,
          text: `${r.text}\n（首次安装：新 Engine 环境保留，旧环境未受影响；修好上面的问题后重跑 /audio-install 即可继续。）`,
        }
      }
      const rb = rollbackEngine(layout, { backup: eng.backup })
      return {
        ok: false,
        text: `${r.text}\n${rb.ok
          ? '已自动回滚到安装前的 Engine，旧环境仍可分析。'
          : `回滚失败：${rb.text}（安装前备份仍在 ${basename(eng.backup)}）`}`,
      }
    }

    try {
      const models = await ensureBaseModels(layout, log)
      if (!models.ok) return fail(models)
      if (opts.stems) {
        const s = await ensureStemModels(layout, { six: opts.stems === '6s' }, log)
        if (!s.ok) return fail(s)
      }
      if (opts.asr) {
        const size = typeof opts.asr === 'string' ? opts.asr : 'base'
        const a = await ensureAsrModel(layout, size, log)
        if (!a.ok) return fail(a)
      }
      const probe = await probeEngine(layout, { extras })
      if (!probe.ok) {
        return fail({ text: `VERIFY_FAILED: ${probe.errors.join('; ')}` })
      }
      return { ok: true, text: `${eng.text}；${models.text}；校验通过` }
    } catch (e) {
      // 模型阶段的异常也必须回滚，不能把「新 venv 已生效、模型半套」留在用户机器上。
      return fail({ text: `MODEL_STAGE_FAILED: ${String(e?.stack || e).slice(0, 800)}` })
    }
  } finally {
    releaseInstallLock(lock)
  }
}
