/* dsh-soundcheck 设置页面板：状态总览 + 安装/自检按钮。 */
window.__ModuleLoader__.load({
  id: 'dsh-soundcheck',
  factory: (require) => {
    const React = require('react')
    const { useState, useEffect, useCallback } = React
    const h = React.createElement
    const NS = 'dsh-soundcheck'

    const zh = {
      nav: '音频分析',
      title: 'dsh-soundcheck',
      subtitle: '本地音频分析引擎的状态与安装',
      engine: '引擎',
      runtime: '运行环境',
      python: 'Python',
      temp: '临时目录',
      deps: '功能组件',
      models: '模型',
      components: '可选功能',
      dataRoot: '数据目录',
      install: '安装 / 修复引擎',
      installOne: '安装',
      stems: '安装分轨',
      asr: '安装听写',
      doctor: '自检',
      running: '正在执行…',
      done: '完成',
      failed: '失败',
      retry: '刷新',
      missing: '缺失',
      ok: '已安装',
      bad: '异常',
      partial: '部分安装',
      notInstalled: '未安装',
      writable: '可写',
      notWritable: '不可写',
      optionalTag: '可选',
      loading: '加载中…',
      loadFailed: '状态读取失败',
      gCore: '基础分析',
      gCoreDesc: '解码音频、算节拍/调性/能量',
      gStems: '分轨',
      gStemsDesc: '把人声、鼓、贝斯、其他分开',
      gAsr: '歌词听写',
      gAsrDesc: '人声转文字',
      gStemComp: '分轨（人声 / 鼓 / 贝斯 / 其他）',
      gAsrComp: '歌词听写（人声转文字）',
    }
    const en = {
      nav: 'Audio',
      title: 'dsh-soundcheck',
      subtitle: 'Local audio engine status',
      engine: 'Engine',
      runtime: 'Runtime',
      python: 'Python',
      temp: 'Temp dir',
      deps: 'Features',
      models: 'Models',
      components: 'Optional features',
      dataRoot: 'Data dir',
      install: 'Install / repair engine',
      installOne: 'Install',
      stems: 'Install stems',
      asr: 'Install ASR',
      doctor: 'Check-up',
      running: 'Running…',
      done: 'Done',
      failed: 'Failed',
      retry: 'Refresh',
      missing: 'missing',
      ok: 'installed',
      bad: 'error',
      partial: 'partial',
      notInstalled: 'not installed',
      writable: 'writable',
      notWritable: 'not writable',
      optionalTag: 'optional',
      loading: 'Loading…',
      loadFailed: 'Failed to load status',
      gCore: 'Core analysis',
      gCoreDesc: 'decode, tempo/key/energy',
      gStems: 'Stem separation',
      gStemsDesc: 'vocals, drums, bass, other',
      gAsr: 'Lyrics transcription',
      gAsrDesc: 'vocals to text',
      gStemComp: 'Stems (vocals / drums / bass / other)',
      gAsrComp: 'Lyrics (vocals to text)',
    }
    let t = (k) => k

    const C = {
      wrap: { padding: '4px 0', display: 'flex', flexDirection: 'column', gap: '12px' },
      head: { display: 'flex', alignItems: 'baseline', gap: '10px', flexWrap: 'wrap' },
      title: { fontSize: '15px', fontWeight: 600, color: 'var(--dsw-alias-label-primary, #eee)' },
      sub: { fontSize: '12px', color: 'var(--dsw-alias-label-tertiary, #999)' },
      card: {
        border: '1px solid var(--dsw-alias-border-l1, rgba(128,128,128,.25))',
        borderRadius: '10px',
        background: 'var(--dsw-alias-bg-layer-1, rgba(128,128,128,.06))',
        padding: '10px 12px',
      },
      cardTitle: { fontSize: '12px', fontWeight: 600, color: 'var(--dsw-alias-label-secondary, #aaa)', marginBottom: '6px' },
      row: { display: 'flex', justifyContent: 'space-between', gap: '12px', padding: '3px 0', fontSize: '13px' },
      name: { color: 'var(--dsw-alias-label-secondary, #999)' },
      val: { color: 'var(--dsw-alias-label-primary, #ddd)', fontFamily: 'ui-monospace, monospace', fontSize: '12px' },
      ok: { color: 'var(--dsw-alias-state-success-primary, #3a3)' },
      bad: { color: 'var(--dsw-alias-state-error-primary, #d55)' },
      warn: { color: 'var(--dsw-alias-state-warn-primary, #d90)' },
      hint: { fontSize: '11px', color: 'var(--dsw-alias-label-tertiary, #888)' },
      btnRow: { display: 'flex', gap: '8px', flexWrap: 'wrap' },
      btn: {
        padding: '6px 12px', fontSize: '12px', borderRadius: '8px', cursor: 'pointer',
        border: '1px solid var(--dsw-alias-border-l2, rgba(128,128,128,.3))',
        background: 'var(--dsw-alias-bg-layer-2, rgba(128,128,128,.1))',
        color: 'var(--dsw-alias-label-primary, #eee)',
      },
      primary: {
        padding: '6px 12px', fontSize: '12px', borderRadius: '8px', cursor: 'pointer',
        border: '1px solid transparent',
        background: 'var(--dsw-alias-button-primary-fill, #4a7)',
        color: 'var(--dsw-alias-label-primary-foreground, #fff)',
      },
      out: {
        marginTop: '8px', padding: '8px 10px', borderRadius: '8px', fontSize: '12px',
        fontFamily: 'ui-monospace, monospace', whiteSpace: 'pre-wrap', maxHeight: '240px', overflow: 'auto',
        background: 'var(--dsw-alias-markdown-code-block, rgba(0,0,0,.25))',
        color: 'var(--dsw-alias-label-primary, #ddd)',
      },
    }

    function Row({ name, value, ok }) {
      return h('div', { style: C.row },
        h('span', { style: C.name }, name),
        h('span', { style: ok === true ? { ...C.val, ...C.ok } : ok === false ? { ...C.val, ...C.bad } : C.val }, value))
    }

    function Panel() {
      const [state, setState] = useState({ status: 'loading' })
      const [busy, setBusy] = useState('')
      const [out, setOut] = useState('')
      const [open, setOpen] = useState({})

      const load = useCallback(() => {
        fetch('/dsh-soundcheck/api/overview', { headers: { accept: 'application/json' }, cache: 'no-store' })
          .then((r) => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json() })
          .then((j) => setState({ status: 'ok', data: j.data }))
          .catch((e) => setState({ status: 'error', error: String((e && e.message) || e) }))
      }, [])

      useEffect(() => { load() }, [load])

      const poll = useCallback((id) => {
        const timer = setInterval(() => {
          fetch('/dsh-soundcheck/api/job?id=' + encodeURIComponent(id), { cache: 'no-store' })
            .then((r) => r.json())
            .then((j) => {
              if (!j.ok) { clearInterval(timer); setBusy(''); setOut(String(j.error || 'job 丢失')); return }
              if (j.job.status !== 'running') {
                clearInterval(timer); setBusy(''); setOut(j.job.text || '')
                load()
              }
            })
            .catch(() => { /* 网络抖动，继续轮询 */ })
        }, 1200)
      }, [load])

      const act = useCallback((kind) => {
        setBusy(kind); setOut('')
        fetch('/dsh-soundcheck/api/action?kind=' + encodeURIComponent(kind), { method: 'POST' })
          .then((r) => r.json())
          .then((j) => {
            if (!j.ok) { setBusy(''); setOut(String(j.error || '失败')); return }
            poll(j.job)
          })
          .catch((e) => { setBusy(''); setOut(String((e && e.message) || e)) })
      }, [poll])

      if (state.status === 'loading') return h('div', { style: C.wrap }, h('div', { style: C.sub }, t('loading')))
      if (state.status === 'error') {
        return h('div', { style: C.wrap },
          h('div', { style: C.bad }, t('loadFailed') + '：' + state.error),
          h('div', { style: C.btnRow }, h('button', { style: C.btn, onClick: load }, t('retry'))))
      }

      const d = state.data
      const deps = d.deps || {}
      const groups = [
        { key: 'core', title: t('gCore'), desc: t('gCoreDesc'), modules: ['librosa', 'onnxruntime', 'soundfile'] },
        { key: 'stems', title: t('gStems'), desc: t('gStemsDesc'), optional: true, modules: ['demucs_onnx', 'soxr'] },
        { key: 'asr', title: t('gAsr'), desc: t('gAsrDesc'), optional: true, modules: ['faster_whisper'] },
      ]
      const depGroup = (g) => {
        const states = g.modules.map((m) => !!deps[m])
        const all = states.every(Boolean)
        const none = states.every((s) => !s)
        const label = all ? t('ok') : none ? t('notInstalled') : t('partial')
        const tone = all ? C.ok : none ? C.bad : C.warn
        const expanded = !!open[g.key]
        return h('div', { key: g.key },
          h('div', {
            style: { ...C.row, cursor: 'pointer', alignItems: 'baseline' },
            onClick: () => setOpen((o) => ({ ...o, [g.key]: !o[g.key] })),
          },
            h('span', { style: C.name },
              (expanded ? '▾ ' : '▸ ') + g.title,
              g.optional ? h('span', { style: { ...C.hint, marginLeft: '6px' } }, '[' + t('optionalTag') + ']') : null,
              h('span', { style: { ...C.hint, marginLeft: '8px' } }, g.desc)),
            h('span', { style: { ...C.val, ...tone } }, label)),
          expanded
            ? h('div', { style: { padding: '2px 0 6px 16px' } },
              g.modules.map((m) => h(Row, {
                key: m, name: m,
                value: deps[m] ? t('ok') : t('notInstalled'),
                ok: !!deps[m],
              })))
            : null)
      }

      const installed = groups.filter((g) => g.modules.every((m) => !!deps[m]))
      const missing = groups.filter((g) => !g.modules.every((m) => !!deps[m]))

      return h('div', { style: C.wrap },
        h('div', { style: C.head },
          h('div', { style: C.title }, t('title')),
          h('div', { style: C.sub }, 'v' + d.version + ' · ' + t('subtitle'))),

        h('div', { style: C.card },
          h('div', { style: C.cardTitle }, t('runtime')),
          h(Row, { name: t('engine'), value: d.engine ? d.root : t('missing'), ok: d.engine }),
          h(Row, { name: t('python'), value: d.python || t('missing'), ok: !!d.python }),
          h(Row, { name: t('temp'), value: d.temp ? t('writable') : t('notWritable'), ok: d.temp }),
          h(Row, { name: t('dataRoot'), value: d.dataRoot })),

        h('div', { style: C.card }, h('div', { style: C.cardTitle }, t('deps')), installed.map(depGroup)),

        h('div', { style: C.card },
          h('div', { style: C.cardTitle }, t('models')),
          (d.models || []).map((m) => h(Row, {
            key: m.name, name: m.name,
            value: m.ok ? t('ok') + ' · ' + m.size_mb + ' MB' : t('notInstalled'),
            ok: m.ok,
          }))),

        missing.length
          ? h('div', { style: C.card },
            h('div', { style: C.cardTitle }, t('components')),
            missing.map((g) => h('div', {
              key: g.key,
              style: { ...C.row, alignItems: 'baseline' },
            },
              h('span', { style: C.name },
                g.title,
                h('span', { style: { ...C.hint, marginLeft: '8px' } }, g.desc)),
              h('button', {
                style: C.btn, disabled: !!busy,
                onClick: () => act(g.key),
              }, busy === g.key ? t('running') : t('installOne')))))
          : null,

        h('div', { style: C.btnRow },
          h('button', { style: C.primary, disabled: !!busy, onClick: () => act('engine') }, busy === 'engine' ? t('running') : t('install')),
          h('button', { style: C.btn, disabled: !!busy, onClick: () => act('doctor') }, busy === 'doctor' ? t('running') : t('doctor')),
          h('button', { style: C.btn, disabled: !!busy, onClick: load }, t('retry'))),

        out ? h('div', { style: C.out }, out) : null)
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
