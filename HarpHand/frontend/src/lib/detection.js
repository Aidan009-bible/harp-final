export const LOG_TIME_WINDOW = 0.25
export const NOTE_COLUMNS = 8
export const MAX_VIDEO_SIZE_BYTES = 250 * 1024 * 1024

export const STRING_TO_NOTE = Object.freeze({
  1: 'G5', 2: 'E5', 3: 'D5', 4: 'C5',
  5: 'A4', 6: 'G4', 7: 'E4', 8: 'D4',
  9: 'C4', 10: 'A3', 11: 'G3', 12: 'E3',
  13: 'D3', 14: 'C3', 15: 'A2', 16: 'G2',
})

const VIDEO_EXTENSIONS = new Set(['mp4', 'mov', 'mkv', 'avi', 'webm'])

export function validateVideoFile(file) {
  if (!file) return 'Choose a performance video first.'

  const extension = file.name?.split('.').pop()?.toLowerCase()
  if (!extension || !VIDEO_EXTENSIONS.has(extension)) {
    return 'Use an MP4, MOV, MKV, AVI, or WebM video.'
  }

  if (file.size > MAX_VIDEO_SIZE_BYTES) {
    return 'Video must be 250 MB or smaller.'
  }

  return null
}

export function formatTime(seconds) {
  const safeSeconds = Number.isFinite(Number(seconds)) ? Math.max(0, Number(seconds)) : 0
  const mins = Math.floor(safeSeconds / 60)
  const secs = (safeSeconds % 60).toFixed(2)
  return `${String(mins).padStart(2, '0')}:${secs.padStart(5, '0')}`
}

export function buildGridRows(logs, currentTime = 0) {
  const byTime = new Map()

  for (const event of logs) {
    const time = Number(event.time ?? 0)
    const key = Number.isFinite(time) ? time : 0
    if (!byTime.has(key)) byTime.set(key, { time: key, audio: [], hand: [] })

    if (event.type === 'audio') byTime.get(key).audio.push(event)
    if (event.type === 'hand') byTime.get(key).hand.push(event)
  }

  return [...byTime.values()]
    .sort((a, b) => a.time - b.time)
    .map((row, index) => {
      const audioStrings = [...new Set(row.audio.map((event) => event.string).filter(Boolean))]
      const handStrings = [...new Set(row.hand.map((event) => event.string).filter(Boolean))]
      const matchedStrings = audioStrings.filter((string) => handStrings.includes(string))
      const match = matchedStrings.length > 0
      const scores = row.audio
        .map((event) => event.confidence)
        .filter((score) => Number.isFinite(score))

      let annotation = ''
      if (scores.length > 0) {
        annotation = scores.map((score) => `${(score * 100).toFixed(0)}%`).join(', ')
      } else if (row.hand[0]) {
        const handScore = Number.isFinite(row.hand[0].confidence)
          ? ` ${(row.hand[0].confidence * 100).toFixed(0)}%`
          : ''
        annotation = `${row.hand[0].finger || ''}${handScore}`.trim()
      }

      return {
        index: index + 1,
        time: row.time,
        stringMain: audioStrings.length ? audioStrings.join(', ') : '-',
        handMain: handStrings.length ? handStrings.join(', ') : '-',
        audioStrings,
        handStrings,
        matchedStrings,
        match,
        note: annotation,
        inWindow: Math.abs(row.time - currentTime) <= LOG_TIME_WINDOW,
        audio: row.audio,
        hand: row.hand,
      }
    })
}

export function calculateDetectionSummary(rows, logs, isCombined) {
  const totalEvents = rows.length
  const comparableEvents = rows.filter((row) => row.audio.length > 0 && row.hand.length > 0).length
  const matches = rows.filter((row) => row.match).length
  const agreement = comparableEvents > 0 ? (matches / comparableEvents) * 100 : null
  const handCoverage = totalEvents > 0 ? (comparableEvents / totalEvents) * 100 : null
  const scores = logs
    .map((event) => event.confidence)
    .filter((score) => Number.isFinite(score))
  const averageScore = scores.length > 0
    ? (scores.reduce((sum, score) => sum + score, 0) / scores.length) * 100
    : null

  return {
    totalEvents,
    comparableEvents,
    matches,
    agreement: isCombined ? agreement : null,
    handCoverage: isCombined ? handCoverage : null,
    averageScore: isCombined ? null : averageScore,
  }
}

export function buildPerStringStats(rows) {
  const stats = Object.fromEntries(
    Array.from({ length: 16 }, (_, index) => [`S${index + 1}`, { total: 0, matches: 0 }]),
  )

  for (const row of rows) {
    for (const string of row.audioStrings) {
      if (!stats[string]) continue
      stats[string].total += 1
      if (row.matchedStrings.includes(string)) stats[string].matches += 1
    }
  }

  return stats
}

export function buildAgreementMatrix(rows) {
  const matrix = {}
  for (let audio = 1; audio <= 16; audio += 1) {
    matrix[`S${audio}`] = {}
    for (let hand = 1; hand <= 16; hand += 1) {
      matrix[`S${audio}`][`S${hand}`] = 0
    }
  }

  for (const row of rows) {
    for (const audioString of row.audioStrings) {
      for (const handString of row.handStrings) {
        if (matrix[audioString]?.[handString] != null) {
          matrix[audioString][handString] += 1
        }
      }
    }
  }

  return matrix
}

export function buildNoteRows(rows) {
  const cells = rows.map((row) => {
    const baseStrings = row.audioStrings.length > 0 ? row.audioStrings : row.handStrings
    const thumbStrings = new Set(
      row.hand
        .filter((event) => String(event.finger || '').toLowerCase() === 'thumb')
        .map((event) => String(event.string || '').trim()),
    )

    const parts = baseStrings
      .map((string) => {
        const number = String(string).replace(/^S\s*/i, '').replace(/\D/g, '')
        if (!number) return null
        return { num: number, thumb: thumbStrings.has(`S${number}`) }
      })
      .filter(Boolean)

    return { parts, together: parts.length > 1 }
  })

  const rowsOfNotes = []
  for (let index = 0; index < cells.length; index += NOTE_COLUMNS) {
    rowsOfNotes.push(cells.slice(index, index + NOTE_COLUMNS))
  }
  return rowsOfNotes
}
