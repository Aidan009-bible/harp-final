import test from 'node:test'
import assert from 'node:assert/strict'

import {
  buildGridRows,
  buildPerStringStats,
  calculateDetectionSummary,
  formatTime,
  validateVideoFile,
} from './detection.js'

test('formatTime handles invalid and minute-spanning values', () => {
  assert.equal(formatTime(Number.NaN), '00:00.00')
  assert.equal(formatTime(61.257), '01:01.26')
})

test('validateVideoFile checks type and size', () => {
  assert.equal(validateVideoFile({ name: 'take.mp4', size: 1024 }), null)
  assert.match(validateVideoFile({ name: 'take.exe', size: 1024 }), /MP4/)
  assert.match(validateVideoFile({ name: 'take.mov', size: 251 * 1024 * 1024 }), /250 MB/)
})

test('agreement is calculated only from comparable events', () => {
  const logs = [
    { type: 'audio', time: 1, string: 'S1', confidence: 0.9 },
    { type: 'hand', time: 1, string: 'S1', confidence: 0.8 },
    { type: 'audio', time: 2, string: 'S2', confidence: 0.7 },
    { type: 'audio', time: 3, string: 'S3', confidence: 0.6 },
    { type: 'hand', time: 3, string: 'S4', confidence: 0.7 },
  ]
  const rows = buildGridRows(logs)
  const summary = calculateDetectionSummary(rows, logs, true)

  assert.equal(summary.totalEvents, 3)
  assert.equal(summary.comparableEvents, 2)
  assert.equal(summary.matches, 1)
  assert.equal(summary.agreement, 50)
})

test('per-string matches do not credit unrelated notes in a chord', () => {
  const rows = buildGridRows([
    { type: 'audio', time: 1, string: 'S1' },
    { type: 'audio', time: 1, string: 'S2' },
    { type: 'hand', time: 1, string: 'S1' },
  ])
  const stats = buildPerStringStats(rows)

  assert.deepEqual(stats.S1, { total: 1, matches: 1 })
  assert.deepEqual(stats.S2, { total: 1, matches: 0 })
})
