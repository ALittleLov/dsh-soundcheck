import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import test from 'node:test'

import { ensureEngineWheel, resolveEngineLayout } from '../lib/engine-installer.js'

function makeWheel(dir, version) {
  const name = `music_evidence_engine-${version}-py3-none-any.whl`
  const path = join(dir, name)
  writeFileSync(path, Buffer.alloc(2048, 7))
  return { path, name, sha256: createHash('sha256').update(readFileSync(path)).digest('hex') }
}

test('ensureEngineWheel writes the manifest version into layout', async () => {
  const root = mkdtempSync(join(tmpdir(), 'soundcheck-wheel-'))
  try {
    const dist = join(root, 'engine-dist')
    mkdirSync(dist, { recursive: true })
    const wheel = makeWheel(dist, '9.9.9')
    const manifest = join(root, 'engine.json')
    writeFileSync(manifest, JSON.stringify({
      version: '9.9.9', wheel: pathToFileURL(wheel.path).href, sha256: wheel.sha256,
    }))
    const layout = resolveEngineLayout({
      dataRoot: root, runtimeDir: root, engineDist: dist,
      manifestUrl: pathToFileURL(manifest).href,
    })
    const got = await ensureEngineWheel(layout)
    assert.equal(got, wheel.path)
    assert.equal(layout.expectedVersion, '9.9.9')
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})

test('explicit wheel also supplies the expected version', async () => {
  const root = mkdtempSync(join(tmpdir(), 'soundcheck-wheel-'))
  try {
    const wheel = makeWheel(root, '9.9.8')
    const layout = resolveEngineLayout({ dataRoot: root, runtimeDir: root, engineDist: join(root, 'engine-dist') })
    layout.wheel = wheel.path
    const got = await ensureEngineWheel(layout)
    assert.equal(got, wheel.path)
    assert.equal(layout.expectedVersion, '9.9.8')
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})
