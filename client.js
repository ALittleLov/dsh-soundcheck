/* dsh-soundcheck 设置面板：状态控制台。 */
window.__ModuleLoader__.load({
  id: 'dsh-soundcheck',
  factory: (require) => {
    const React = require('react')
    const { useState, useEffect, useCallback, useRef } = React
    const h = React.createElement
    const NS = 'dsh-soundcheck'

    const zh = {
      nav: '音频分析', title: 'dsh-soundcheck', subtitle: '本地音频分析引擎',
      core: '核心分析', diag: '诊断与日志', optional: '按需功能',
      statusNotInstalled: '引擎尚未安装', statusRepair: '引擎需要修复',
      statusPartial: '部分分析能力可用', statusReady: '核心分析引擎已全面就绪',
      hintNotInstalled: '首次安装会准备 Python 环境、核心依赖与必需模型。',
      hintRepair: '检测到核心组件缺失或版本不一致，修复后恢复正常。',
      hintPartial: '核心分析部分受损，补齐缺失组件后恢复。',
      hintReady: '核心依赖与必需模型齐全，可以直接分析音频。',
      btnInstall: '安装引擎', btnRepair: '修复引擎', btnFill: '补齐缺失组件',
      btnDoctor: '运行自检', running: '正在执行…', refresh: '刷新', refreshDone: '刷新',
      env: '运行环境', deps: '核心依赖', models: '必需模型', dirs: '本地目录',
      detail: '查看组件与目录详情', optionalSummary: '分轨 · 歌词听写',
      stems: '音轨分离', asr: '歌词听写', installOne: '安装', repairOne: '修复',
      installed: '已安装', missing: '缺失', partial: '部分', notInstalled: '未安装',
      copy: '复制日志', copied: '已复制', logTitle: '任务日志', errorSummary: '执行失败',
      loading: '加载中…', loadFailed: '状态读取失败', retry: '重试',
      venv: 'venv', data: '数据目录', temp: '临时目录', writable: '可写', notWritable: '不可写',
      checkedAt: '检查于',
    }
    const en = {
      nav: 'Audio', title: 'dsh-soundcheck', subtitle: 'Local audio engine',
      core: 'Core analysis', diag: 'Diagnostics', optional: 'Optional features',
      statusNotInstalled: 'Engine not installed', statusRepair: 'Engine needs repair',
      statusPartial: 'Partially available', statusReady: 'Core analysis ready',
      hintNotInstalled: 'First install prepares Python, core deps and required models.',
      hintRepair: 'Core components missing or version mismatch.',
      hintPartial: 'Part of core analysis is unavailable.',
      hintReady: 'Core deps and required models are ready.',
      btnInstall: 'Install engine', btnRepair: 'Repair engine', btnFill: 'Fill missing parts',
      btnDoctor: 'Run check-up', running: 'Running…', refresh: 'Refresh', refreshDone: 'Refresh',
      env: 'Runtime', deps: 'Core deps', models: 'Required models', dirs: 'Local dirs',
      detail: 'Show components and paths', optionalSummary: 'Stems · Lyrics',
      stems: 'Stem separation', asr: 'Lyrics transcription', installOne: 'Install', repairOne: 'Repair',
      installed: 'installed', missing: 'missing', partial: 'partial', notInstalled: 'not installed',
      copy: 'Copy log', copied: 'Copied', logTitle: 'Task log', errorSummary: 'Failed',
      loading: 'Loading…', loadFailed: 'Failed to load status', retry: 'Retry',
      venv: 'venv', data: 'Data dir', temp: 'Temp dir', writable: 'writable', notWritable: 'not writable',
      checkedAt: 'checked at',
    }
    let t = (k) => k

    const C = {
      wrap: { padding: '4px 0', display: 'flex', flexDirection: 'column', gap: '22px', fontSize: '13px' },
      head: { display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: '10px' },
      title: { fontSize: '15px', fontWeight: 600, color: 'var(--dsw-alias-label-primary, #eee)' },
      sub: { fontSize: '11px', color: 'var(--dsw-alias-label-tertiary, #888)' },
      hero: { border: '1px solid var(--dsw-alias-border-l1, rgba(128,128,128,.25))', borderRadius: '12px', padding: '18px', display: 'flex', flexDirection: 'column', gap: '10px' },
      heroTop: { display: 'flex', alignItems: 'center', gap: '10px' },
      dot: { width: '8px', height: '8px', borderRadius: '50%', flex: 'none' },
      heroTitle: { fontSize: '21px', fontWeight: 600, color: 'var(--dsw-alias-label-primary, #eee)', flex: '1 1 auto' },
      hint: { fontSize: '12px', color: 'var(--dsw-alias-label-tertiary, #888)', lineHeight: '1.6' },
      section: { display: 'flex', flexDirection: 'column' },
      sectionTitle: { fontSize: '13px', fontWeight: 600, color: 'var(--dsw-alias-label-secondary, #aaa)', marginBottom: '2px' },
      row: { display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: '12px', padding: '10px 0', borderBottom: '1px solid var(--dsw-alias-border-l1, rgba(128,128,128,.18))' },
      name: { fontSize: '12px', color: 'var(--dsw-alias-label-secondary, #999)' },
      val: { fontSize: '12px', fontFamily: 'ui-monospace, monospace', color: 'var(--dsw-alias-label-primary, #ddd)', textAlign: 'right', overflowWrap: 'anywhere' },
      ok: { color: 'var(--dsw-alias-state-success-primary, #3a3)' },
      bad: { color: 'var(--dsw-alias-state-error-primary, #d55)' },
      warn: { color: 'var(--dsw-alias-state-warn-primary, #d90)' },
      muted: { color: 'var(--dsw-alias-label-tertiary, #888)' },
      btnRow: { display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' },
      btn: { padding: '6px 12px', fontSize: '12px', lineHeight: '20px', borderRadius: '8px', cursor: 'pointer', textAlign: 'center',
        border: '1px solid var(--dsw-alias-border-l2, rgba(128,128,128,.3))',
        background: 'transparent', color: 'var(--dsw-alias-label-primary, #eee)' },
      primary: { width: '120px', height: '32px', padding: '0 8px', boxSizing: 'border-box', fontSize: '12px', fontWeight: 600, lineHeight: '20px', borderRadius: '8px', cursor: 'pointer', textAlign: 'center', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', whiteSpace: 'nowrap',
        border: '1px solid transparent', background: 'var(--dsw-alias-button-primary-fill, #4a7)',
        color: 'var(--dsw-alias-label-primary-foreground, #fff)' },
      details: { marginTop: '4px' },
      summary: { fontSize: '12px', color: 'var(--dsw-alias-label-secondary, #999)', cursor: 'pointer', padding: '8px 0' },
      log: { margin: '6px 0 0', padding: '10px 12px', borderRadius: '8px', fontSize: '11px', lineHeight: '1.55',
        fontFamily: 'ui-monospace, monospace', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', maxHeight: '240px', overflow: 'auto',
        background: 'var(--dsw-alias-markdown-code-block, rgba(0,0,0,.25))', color: 'var(--dsw-alias-label-primary, #ddd)' },
      detailBox: { display: 'flex', flexDirection: 'column', gap: '4px' },
      detailLine: { display: 'flex', justifyContent: 'space-between', gap: '12px', fontSize: '12px' },
      detailName: { fontFamily: 'ui-monospace, monospace', color: 'var(--dsw-alias-label-secondary, #999)' },
      detailVal: { color: 'var(--dsw-alias-label-primary, #ddd)', textAlign: 'right' },
      errBox: { padding: '10px 12px', borderRadius: '8px', fontSize: '12px',
        border: '1px solid var(--dsw-alias-state-error-primary, #d55)', color: 'var(--dsw-alias-state-error-primary, #d55)', overflowWrap: 'anywhere' },
      tool: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '10px', fontSize: '11px', color: 'var(--dsw-alias-label-tertiary, #888)' },
    }

    const REQUIRED_MODELS = ['yamnet.onnx', 'yamnet_class_map.csv', 'beat_this', 'frontend.json']
    const CORE_DEPS = ['librosa', 'onnxruntime', 'soundfile', 'soxr']

    function Row({ name, value, ok }) {
      return h('div', { style: C.row },
        h('span', { style: C.name }, name),
        h('span', { style: { ...C.val, ...(ok === true ? C.ok : ok === false ? C.bad : {}) } }, value))
    }

    function Item({ title, desc, tone, onClick, open, children, action }) {
      return h('div', null,
        h('div', { style: { display: 'flex', alignItems: 'center', gap: '10px', padding: '12px 0' },
          onClick },
          h('div', { style: { flex: '1 1 auto', minWidth: 0 } },
            h('div', { style: { fontSize: '13px', fontWeight: 500, color: 'var(--dsw-alias-label-primary, #eee)' } }, title),
            desc ? h('div', { style: { fontSize: '12px', color: tone || 'var(--dsw-alias-label-tertiary, #888)', marginTop: '2px', overflowWrap: 'anywhere' } }, desc) : null),
          onClick ? h('span', { style: { fontSize: '12px', color: 'var(--dsw-alias-label-tertiary, #888)', flex: 'none' } }, open ? '收起' : '查看') : null,
          action || null),
        open && children
          ? h('div', { style: { padding: '0 0 12px 0', display: 'flex', flexDirection: 'column', gap: '6px' } }, children)
          : null)
    }

    function Panel() {
      const [state, setState] = useState({ status: 'loading' })
      const [busy, setBusy] = useState('')
      const [out, setOut] = useState('')
      const [outErr, setOutErr] = useState('')
      const [openLog, setOpenLog] = useState(false)
      const [openKey, setOpenKey] = useState('')
      const [copied, setCopied] = useState(false)
      const [checkedAt, setCheckedAt] = useState('')
      const [elapsed, setElapsed] = useState(0)
      const [checking, setChecking] = useState(false)
      const [protectedMsg, setProtectedMsg] = useState(false)
      const userClosedLog = useRef(false)

      const load = useCallback(() => {
        fetch('/dsh-soundcheck/api/overview', { headers: { accept: 'application/json' }, cache: 'no-store' })
          .then((r) => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json() })
          .then((j) => {
            setState({ status: 'ok', data: j.data })
            setCheckedAt(new Date().toLocaleTimeString())
            const saved = (() => { try { return localStorage.getItem('dsc_active_job') } catch { return null } })()
            const active = (j.data && j.data.activeJob) || (saved ? { id: saved, status: 'running' } : null)
            if (active && active.status === 'running') {
              setBusy('engine')
              poll(active.id)
            }
          })
          .catch((e) => setState({ status: 'error', error: String((e && e.message) || e) }))
      }, [])

      useEffect(() => { load() }, [load])

      // 后端任务是唯一事实来源：挂载即订阅，卸载即取消；不依赖组件本地 busy。
      const [snap, setSnap] = useState(null)
      useEffect(() => {
        let alive = true
        let timer = null
        const ctl = new AbortController()
        const tick = async () => {
          try {
            const r = await fetch('/dsh-soundcheck/api/overview', { cache: 'no-store', signal: ctl.signal })
            if (!r.ok) throw new Error('HTTP ' + r.status)
            const j = await r.json()
            if (alive) setSnap(j.data || null)
          } catch { /* 保留上一次快照 */ }
          finally { if (alive) timer = setTimeout(tick, 1500) }
        }
        tick()
        return () => { alive = false; if (timer) clearTimeout(timer); ctl.abort() }
      }, [])

      useEffect(() => {
        if (!busy) return
        setElapsed(0)
        const timer = setInterval(() => setElapsed((v) => v + 1), 1000)
        return () => clearInterval(timer)
      }, [busy])

      const poll = useCallback((id) => {
        const timer = setInterval(() => {
          fetch('/dsh-soundcheck/api/job?id=' + encodeURIComponent(id), { cache: 'no-store' })
            .then((r) => r.json())
            .then((j) => {
              if (!j.ok) {
                clearInterval(timer); setBusy(''); setOutErr(String(j.error || 'job 丢失')); setOpenLog(true); userClosedLog.current = false
                try { localStorage.removeItem('dsc_active_job') } catch { /* ignore */ }
                return
              }
              if (j.job.status !== 'running') {
                clearInterval(timer); setBusy(''); setChecking(false); try { localStorage.removeItem('dsc_active_job') } catch { /* ignore */ }
                const failed = j.job.status === 'error'
                if (failed) { setOutErr(j.job.text || '执行失败'); if (!userClosedLog.current) setOpenLog(true) }
                else { setOutErr(''); setOut(j.job.text || '') }
                load()
              }
            })
            .catch(() => { /* 网络抖动，继续轮询 */ })
        }, 1200)
      }, [load])

      const checkUpdate = useCallback(() => {
        setChecking(true); setOut(''); setOutErr('')
        fetch('/dsh-soundcheck/api/action?kind=check-update', { method: 'POST' })
          .then((r) => r.json())
          .then((j) => {
            if (!j.ok) { setChecking(false); setOutErr(String(j.error || '失败')); setOpenLog(true); return }
            poll(j.job)
          })
          .catch((e) => { setChecking(false); setOutErr(String((e && e.message) || e)); setOpenLog(true) })
      }, [poll])

      const act = useCallback((kind) => {
        setBusy(kind); setOut(''); setOutErr(''); userClosedLog.current = false
        fetch('/dsh-soundcheck/api/action?kind=' + encodeURIComponent(kind), { method: 'POST' })
          .then((r) => r.json())
          .then((j) => {
            if (!j.ok) { setBusy(''); setOutErr(String(j.error || '失败')); setOpenLog(true); return }
            try { localStorage.setItem('dsc_active_job', j.job) } catch { /* ignore */ }
            poll(j.job)
          })
          .catch((e) => { setBusy(''); setOutErr(String((e && e.message) || e)); setOpenLog(true) })
      }, [poll])

      const copyLog = useCallback(() => {
        const text = [outErr, out].filter(Boolean).join('\n')
        try {
          navigator.clipboard.writeText(text).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500) })
        } catch { /* 剪贴板不可用时忽略 */ }
      }, [out, outErr])

      if (state.status === 'loading') return h('div', { style: C.wrap }, h('div', { style: C.hint }, t('loading')))
      if (state.status === 'error') {
        return h('div', { style: C.wrap },
          h('div', { style: C.errBox }, t('loadFailed') + '：' + state.error),
          h('div', { style: C.btnRow }, h('button', { style: C.btn, onClick: load }, t('retry'))))
      }

      const d = state.data
      const deps = d.deps || {}
      const models = d.models || []
      const modelOk = (n) => { const m = models.find((x) => x.name === n); return !!(m && m.ok) }
      const coreDepsOk = d.groups && d.groups.deps ? d.groups.deps.every((x) => x.ok) : CORE_DEPS.every((m) => !!deps[m])
      const coreModelsOk = d.groups && d.groups.models ? d.groups.models.every((x) => x.ok) : REQUIRED_MODELS.every(modelOk)
      const versionOk = d.versionMatch !== false && d.engineVersion === d.expectedVersion
      const optionalStems = !!(d.components && d.components.stems)
      const optionalAsr = !!(d.components && d.components.asr)
      const componentsOk = !!(d.components && d.components.stems)
      const engineOk = !!d.engine

      const hasUpdate = !!(d.engineVersion && d.latestVersion && d.engineVersion !== d.latestVersion)
      const engineJob = snap && snap.activeJobs && snap.activeJobs.engine
      const checkJob = snap && snap.activeJobs && snap.activeJobs.check
      const activeRun = (engineJob && engineJob.status === 'running') ? engineJob : null
      const checkingNow = !!checking || !!(checkJob && checkJob.status === 'running')
      const elapsedShown = (activeRun && activeRun.startedAt)
        ? Math.max(0, Math.floor((Date.now() - activeRun.startedAt) / 1000))
        : elapsed
      const overall = d.engineState === 'not_installed' ? 'notInstalled'
        : d.engineState === 'broken' ? 'repair'
          : d.engineState === 'update_available' ? 'update'
            : (!engineOk ? 'repair' : 'ready')
      const meta = {
        notInstalled: { tone: C.bad, title: t('statusNotInstalled'), hint: t('hintNotInstalled'), btn: t('btnInstall'), primary: true },
        repair: { tone: C.warn, title: t('statusRepair'), hint: t('hintRepair'), btn: t('btnRepair'), primary: true },
        partial: { tone: C.warn, title: t('statusPartial'), hint: t('hintPartial'), btn: t('btnFill'), primary: true },
        ready: { tone: C.ok, title: t('statusReady'), hint: t('hintReady'), btn: t('btnDoctor'), primary: false },
        update: { tone: C.warn, title: '有可用更新', hint: d.engineVersion && d.latestVersion ? `当前 Engine ${d.engineVersion}，可更新到 ${d.latestVersion}；确认后才会升级。` : '', btn: '更新 Engine', primary: true },
      }[overall]
      const shown = overall === 'update'
        ? { tone: C.warn, title: '有可用更新', hint: `当前 Engine ${d.engineVersion}，可更新到 ${d.latestVersion}；确认后才会升级。`, btn: '更新 Engine', primary: true }
        : meta
      const doMain = () => act(overall === 'ready' ? 'doctor' : 'engine')

      const depsCount = CORE_DEPS.filter((m) => !!deps[m]).length
      const modelsCount = d.groups && d.groups.models ? d.groups.models.filter((x) => x.ok).length : REQUIRED_MODELS.filter(modelOk).length
      const failing = !!outErr
      const missingDeps = CORE_DEPS.filter((m) => !deps[m])
      const missingModels = REQUIRED_MODELS.filter((n) => !modelOk(n))
      const clickRow = (key, name, value, ok, body) => h('div', {
        style: { ...C.row, cursor: 'pointer', flexDirection: 'column', alignItems: 'stretch', gap: '0' },
        onClick: () => setOpenKey((k) => (k === key ? '' : key)),
      },
        h('div', { style: { display: 'flex', justifyContent: 'space-between', gap: '12px' } },
          h('span', { style: C.name }, (openKey === key ? '▾ ' : '▸ ') + name),
          h('span', { style: { ...C.val, ...(ok === true ? C.ok : ok === false ? C.bad : {}) } }, value)),
        openKey === key
          ? h('div', { style: { padding: '8px 0 2px 14px' } }, body)
          : null)

      return h('div', { style: C.wrap },
        h('div', { style: C.head },
          h('div', null,
            h('div', { style: C.title }, t('title')),
            h('div', { style: { ...C.sub, marginTop: '2px' } }, t('subtitle'))),
          h('div', { style: C.sub }, 'v' + d.version)),

        h('div', { style: C.hero },
          h('div', { style: C.heroTop },
          h('span', { style: { ...C.dot, background: shown.tone.color } }),
          h('span', {
            style: overall === 'ready' && !protectedMsg ? { ...C.heroTitle, cursor: 'pointer' } : C.heroTitle,
            onClick: overall === 'ready' && !protectedMsg ? () => setProtectedMsg(true) : undefined,
          }, overall === 'ready' && protectedMsg ? '您已处于大肥鱼强大的保护之中 (｡•̀ᴗ-)✧' : shown.title),
          (busy === 'engine' || busy === 'stems' || busy === 'stems6s' || busy === 'asr' || activeRun)
            ? h('div', { style: {
                position: 'relative', overflow: 'hidden', width: '120px', height: '32px', padding: '0 8px', boxSizing: 'border-box',
                fontSize: '12px', fontWeight: 600, lineHeight: '20px', borderRadius: '8px',
                display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                background: 'var(--dsw-alias-button-primary-fill, #4a7)',
                color: 'var(--dsw-alias-label-primary-foreground, #fff)',
              } },
                h('style', null, '@keyframes dsc-move{from{background-position:0 0}to{background-position:48px 0}}'),
                h('div', { style: {
                  position: 'absolute', inset: 0,
                  background: 'repeating-linear-gradient(90deg, rgba(255,255,255,0) 0 18px, rgba(255,255,255,.28) 18px 36px)',
                  backgroundSize: '48px 100%', animation: 'dsc-move 1s linear infinite',
                } }),
                h('span', { style: { position: 'relative' } }, '安装中 · ' + elapsedShown + 's'))
            : (overall !== 'ready' ? h('button', {
                style: shown.primary ? C.primary : C.btn,
                disabled: !!busy,
                onClick: doMain,
              }, shown.btn) : null)),
          h('div', { style: C.hint }, shown.hint),
        (overall === 'ready' && d.engineVersion && d.latestVersion && d.engineVersion !== d.latestVersion)
          ? h('div', { style: { ...C.hint, ...C.muted } }, `本机 Engine ${d.engineVersion}，发布版 ${d.latestVersion}（比发布版新，不影响使用）`)
          : null,
          (missingDeps.length || missingModels.length)
            ? h('div', { style: { ...C.hint, ...C.bad } }, '缺少：' + [...missingDeps, ...missingModels].join('、'))
            : null),

        h('div', { style: C.section },
          h('div', { style: C.sectionTitle }, t('core')),
          h(Item, {
            title: t('env'),
            desc: d.python ? 'Python ' + d.python + ' · Engine ' + (d.engineVersion || '?') : t('notInstalled'),
            onClick: () => setOpenKey((k) => (k === 'env' ? '' : 'env')),
            open: openKey === 'env',
          }, h('div', { style: C.detailBox },
            h(Row, { name: 'venv', value: d.venvDir || '—' }),
            h(Row, { name: t('data'), value: d.dataRoot || '—' }))),
          h(Item, {
            title: t('deps'),
            desc: coreDepsOk ? CORE_DEPS.length + ' 项检查通过' : depsCount + '/' + CORE_DEPS.length + ' · 缺少 ' + missingDeps.join('、'),
            tone: coreDepsOk ? null : C.bad.color,
            onClick: () => setOpenKey((k) => (k === 'deps' ? '' : 'deps')),
            open: openKey === 'deps',
          }, h('div', null, CORE_DEPS.map((m) => h('div', { key: m, style: C.detailLine },
            h('span', { style: C.detailName }, m),
            h('span', { style: { ...C.detailVal, ...(deps[m] ? {} : C.bad) } }, deps[m] ? t('installed') : t('missing')))))),
          h(Item, {
            title: t('models'),
            desc: coreModelsOk ? REQUIRED_MODELS.length + ' 项齐全' : modelsCount + '/' + REQUIRED_MODELS.length + ' · 缺少 ' + missingModels.join('、'),
            tone: coreModelsOk ? null : C.bad.color,
            onClick: () => setOpenKey((k) => (k === 'models' ? '' : 'models')),
            open: openKey === 'models',
          }, h('div', null, REQUIRED_MODELS.map((n) => {
            const m = models.find((x) => x.name === n)
            return h('div', { key: n, style: C.detailLine },
              h('span', { style: C.detailName }, n),
              h('span', { style: { ...C.detailVal, ...(m && m.ok ? {} : C.bad) } }, m && m.ok ? (m.size_mb ? m.size_mb + ' MB' : t('installed')) : t('missing')))
          }))),
          h(Item, {
            title: '音轨分离（4 轨）',
            desc: optionalStems ? '已就绪：人声 / 鼓 / 贝斯 / 其他' : '未安装：人声 / 鼓 / 贝斯 / 其他',
            tone: optionalStems ? null : C.bad.color,
          }, null),
          h(Item, {
            title: t('dirs'),
            desc: d.temp ? '检查通过' : '临时目录不可写',
            tone: d.temp ? null : C.bad.color,
            onClick: () => setOpenKey((k) => (k === 'dirs' ? '' : 'dirs')),
            open: openKey === 'dirs',
          }, h('div', { style: C.detailBox },
            h(Row, { name: t('venv'), value: d.venvDir || '—' }),
            h(Row, { name: t('data'), value: d.dataRoot || '—' }),
            h(Row, { name: t('temp'), value: d.temp ? t('writable') : t('notWritable') }))),

          h(Item, {
            title: t('optional'),
            desc: '音轨分离 · 歌词听写',
            onClick: () => setOpenKey((k) => (k === 'opt' ? '' : 'opt')),
            open: openKey === 'opt',
          }, h('div', null,
            h(Item, {
              title: '音轨分离（6 轨）',
              desc: '在 4 轨基础上再分吉他 / 钢琴',
              action: h('span', { style: C.btnRow },
                h('button', { style: C.btn, disabled: !!busy, onClick: (e) => { e.stopPropagation(); act('stems6s') } }, '安装 6 轨')),
            }),
            h(Item, {
              title: t('asr'),
              desc: '人声转文字',
              action: h('span', { style: C.btnRow },
                h('button', { style: C.btn, disabled: !!busy, onClick: (e) => { e.stopPropagation(); act('asr') } }, '安装')),
            })))),

        h('div', { style: C.section },
          h('div', { style: C.sectionTitle }, t('diag')),
          failing
            ? h('div', { style: C.errBox }, outErr.slice(0, 300))
            : h('div', { style: { fontSize: '12px', color: 'var(--dsw-alias-label-tertiary, #888)', padding: '6px 0 10px' } }, '最近任务：无异常'),
          failing || openLog
            ? (out || outErr
              ? h('div', null,
                  h('pre', { style: C.log }, [outErr, out].filter(Boolean).join('\n\n')),
                  h('div', { style: { marginTop: '6px' } }, h('button', { style: C.btn, onClick: copyLog }, copied ? t('copied') : t('copy'))))
              : null)
            : (out || outErr
              ? h('button', { style: { ...C.btn, marginTop: '6px' }, onClick: () => setOpenLog(true) }, '查看日志')
              : null)),

        h('div', { style: C.tool },
          h('span', null, d.version ? t('checkedAt') + ' ' + (checkedAt || '—') : ''),
          h('span', { style: C.btnRow },
            h('button', { style: C.btn, disabled: !!busy, onClick: () => act('doctor') }, busy === 'doctor' ? t('running') : t('btnDoctor')),
            h('button', { style: C.btn, disabled: !!busy || checking, onClick: checkUpdate }, checkingNow ? '检查中…' : '检查更新'),
            h('button', { style: C.btn, disabled: !!busy, onClick: load }, t('refresh')))))
    }

    return {
      inject: ['slots', 'locale'],
      apply(ctx) {
        ctx.effect(() => ctx.locale.register(NS, { zh, en }), 'dicts')
        const bound = ctx.locale.bind(NS)
        t = bound
        ctx.effect(() => ctx.slots.inject('settings.section', () => ctx.slots.register({
          name: 'settings.section',
          id: NS,
          order: 70,
          label: () => bound('nav'),
          locale: NS,
        }, Panel)), 'settings section')
      },
    }
  },
})
