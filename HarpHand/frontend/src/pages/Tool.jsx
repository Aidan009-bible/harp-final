import { useState, useRef, useCallback, useEffect, useMemo } from 'react'
import { Link } from 'react-router-dom'
import '../App.css'
import harpImage from '../assets/myanmar_harp.jpg'
import {
  NOTE_COLUMNS,
  STRING_TO_NOTE,
  buildAgreementMatrix,
  buildGridRows,
  buildNoteRows,
  buildPerStringStats,
  calculateDetectionSummary,
  formatTime,
  validateVideoFile,
} from '../lib/detection.js'

const API = import.meta.env.VITE_API_URL || '/api'

const DETECTION_METHODS = [
  {
    value: 'audio',
    label: 'Audio',
    eyebrow: 'Fastest',
    description: 'Detect plucks from the soundtrack using the trained 16-string model.',
  },
  {
    value: 'hand',
    label: 'Hand tracking',
    eyebrow: 'Visual',
    description: 'Track fingertips and their proximity to detected harp strings.',
  },
  {
    value: 'both',
    label: 'Audio + hand',
    eyebrow: 'Recommended',
    description: 'Compare both signals and review where their string labels agree.',
  },
]

export default function App() {
  const [modelFile, setModelFile] = useState(null)
  const [videoFile, setVideoFile] = useState(null)
  const [jobId, setJobId] = useState(null)
  const [status, setStatus] = useState(null)
  const [error, setError] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [method, setMethod] = useState('both')  // 'audio' | 'hand' | 'both'
  const [mode, setMode] = useState('hybrid')   // 'default' | 'hybrid' (audio only)
  const [weightsFile, setWeightsFile] = useState(null)
  const [logs, setLogs] = useState([])
  const [videoUrl, setVideoUrl] = useState(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [logViewMode, setLogViewMode] = useState('list')  // 'list' | 'grid'
  const [noteFormat, setNoteFormat] = useState('number')  // 'number' | 'note'
  const [videoDuration, setVideoDuration] = useState(0)
  const [useDefaultModel, setUseDefaultModel] = useState(false)
  const [useDefaultWeights, setUseDefaultWeights] = useState(true)
  const [defaultsAvailable, setDefaultsAvailable] = useState({
    default_model: false,
    default_weights: false,
    allow_custom_model_uploads: false,
    calibrated_thresholds: false,
    ffmpeg_available: false,
    hand_pre_onset_ms: 150,
  })
  const [apiState, setApiState] = useState('checking')
  const [isDraggingVideo, setIsDraggingVideo] = useState(false)
  const videoRef = useRef(null)
  const generatedNoteRef = useRef(null)
  const logPanelRef = useRef(null)
  const modelInput = useRef(null)
  const videoInput = useRef(null)
  const weightsInput = useRef(null)
  const pollTimerRef = useRef(null)
  const activeJobRef = useRef(null)

  useEffect(() => {
    const controller = new AbortController()

    fetch(`${API}/defaults`, { signal: controller.signal })
      .then((r) => {
        if (!r.ok) throw new Error('API not ok')
        return r.json()
      })
      .then((d) => {
        setDefaultsAvailable(d)
        setUseDefaultModel(!!d.default_model)
        setUseDefaultWeights(!!d.default_weights)
        setApiState('ready')
      })
      .catch((fetchError) => {
        if (fetchError.name === 'AbortError') return
        setDefaultsAvailable({
          default_model: false,
          default_weights: false,
          allow_custom_model_uploads: false,
          calibrated_thresholds: false,
          ffmpeg_available: false,
          hand_pre_onset_ms: 150,
        })
        setUseDefaultModel(false)
        setUseDefaultWeights(false)
        setApiState('unavailable')
      })

    return () => controller.abort()
  }, [])

  useEffect(() => () => {
    if (pollTimerRef.current) window.clearTimeout(pollTimerRef.current)
    activeJobRef.current = null
  }, [])

  const visibleLogs = useMemo(
    () => logs.filter((event) => Math.abs((event.time || 0) - currentTime) <= 0.25),
    [logs, currentTime],
  )
  const gridRows = useMemo(() => buildGridRows(logs, currentTime), [logs, currentTime])
  const perStringStats = useMemo(() => buildPerStringStats(gridRows), [gridRows])
  const agreementMatrix = useMemo(() => buildAgreementMatrix(gridRows), [gridRows])
  const isCombinedResult = Boolean(status?.audio && status?.hand)
  const summary = useMemo(
    () => calculateDetectionSummary(gridRows, logs, isCombinedResult),
    [gridRows, logs, isCombinedResult],
  )

  const pollStatus = useCallback(async (id) => {
    if (activeJobRef.current !== id) return

    try {
      const res = await fetch(`${API}/status/${id}`)
      if (!res.ok) throw new Error('Could not check the analysis status.')
      const data = await res.json()
      if (activeJobRef.current !== id) return

      setStatus(data)
      if (data.status === 'done') {
        const logsRes = await fetch(`${API}/logs/${id}`)
        if (!logsRes.ok) throw new Error('Analysis finished, but the event log could not be loaded.')

        const logsData = await logsRes.json()
        setLogs(logsData.events || [])

        if (data.audio && data.hand) {
          const type = data.combined ? 'combined' : 'audio'
          setVideoUrl(`${API}/video-stream/${id}?type=${type}`)
        } else if (data.video_path) {
          setVideoUrl(`${API}/video-stream/${id}`)
        }
        activeJobRef.current = null
        return
      }

      if (data.status === 'error') {
        setError(data.message || 'The analysis could not be completed.')
        activeJobRef.current = null
        return
      }

      pollTimerRef.current = window.setTimeout(() => pollStatus(id), 1500)
    } catch (pollError) {
      setError(pollError.message || 'Lost contact with the analysis service.')
      setStatus({ status: 'error', message: pollError.message })
      activeJobRef.current = null
    }
  }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()
    const videoError = validateVideoFile(videoFile)
    if (videoError) {
      setError(videoError)
      return
    }
    if (apiState !== 'ready') {
      setError('The analysis service is unavailable. Start the backend and try again.')
      return
    }
    if ((method === 'audio' || method === 'both') && !useDefaultModel && !modelFile) {
      setError('Use default model or select a .keras model file.')
      return
    }
    if ((method === 'audio' || method === 'both') && useDefaultModel && !defaultsAvailable.default_model) {
      setError('Default model is not available. Place default.keras in backend/models/ or upload your own.')
      return
    }
    if ((method === 'hand' || method === 'both') && !useDefaultWeights && !weightsFile) {
      setError('Use default weights or select a .pt weights file.')
      return
    }
    if ((method === 'hand' || method === 'both') && useDefaultWeights && !defaultsAvailable.default_weights) {
      setError('Default weights are not available. Place best.pt in backend/weights/ or upload your own.')
      return
    }
    setError(null)
    setStatus(null)
    setLogs([])
    setVideoUrl(null)
    setCurrentTime(0)
    setVideoDuration(0)
    setUploading(true)
    try {
      const form = new FormData()
      form.append('method', method)
      form.append('video', videoFile)
      if (method === 'audio' || method === 'both') {
        const sendDefaultModel = defaultsAvailable.default_model ? useDefaultModel : false;
        form.append('use_default_model', sendDefaultModel ? 'true' : 'false')
        form.append('mode', mode)
        if (!sendDefaultModel && modelFile) form.append('model', modelFile)
      }
      if (method === 'hand' || method === 'both') {
        const sendDefaultWeights = defaultsAvailable.default_weights ? useDefaultWeights : false;
        form.append('use_default_weights', sendDefaultWeights ? 'true' : 'false')
        if (!sendDefaultWeights && weightsFile) form.append('weights', weightsFile)
      }
      const res = await fetch(`${API}/upload`, {
        method: 'POST',
        body: form,
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || res.statusText)
      }
      const { job_id } = await res.json()
      setJobId(job_id)
      activeJobRef.current = job_id
      pollStatus(job_id)
    } catch (err) {
      setError(err.message || 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  const reset = () => {
    if (pollTimerRef.current) window.clearTimeout(pollTimerRef.current)
    pollTimerRef.current = null
    activeJobRef.current = null
    setModelFile(null)
    setVideoFile(null)
    setWeightsFile(null)
    setJobId(null)
    setStatus(null)
    setError(null)
    setLogs([])
    setVideoUrl(null)
    setCurrentTime(0)
    setVideoDuration(0)
    if (modelInput.current) modelInput.current.value = ''
    if (videoInput.current) videoInput.current.value = ''
    if (weightsInput.current) weightsInput.current.value = ''
  }

  const handleVideoSelection = (file) => {
    const validationError = validateVideoFile(file)
    setVideoFile(validationError ? null : file)
    setError(validationError)
    if (validationError && videoInput.current) videoInput.current.value = ''
  }

  const handleDownloadNotePdf = async () => {
    if (!generatedNoteRef.current || !noteRows.length) return
    try {
      const [{ default: jsPDF }, { default: html2canvas }] = await Promise.all([
        import('jspdf'),
        import('html2canvas'),
      ])
      const node = generatedNoteRef.current
      // Temporarily expand the container so html2canvas captures the full grid
      const origMaxHeight = node.style.maxHeight
      const origOverflow = node.style.overflow
      node.style.maxHeight = 'none'
      node.style.overflow = 'visible'

      const canvas = await html2canvas(node, {
        scale: window.devicePixelRatio || 2,
        useCORS: true,
        backgroundColor: '#f5f0e1',
        scrollY: -window.scrollY,
        windowHeight: node.scrollHeight + 200,
      })

      // Restore original styles
      node.style.maxHeight = origMaxHeight
      node.style.overflow = origOverflow

      const imgData = canvas.toDataURL('image/png')
      const pdf = new jsPDF({
        orientation: 'landscape',
        unit: 'pt',
        format: 'a4',
      })
      const pageWidth = pdf.internal.pageSize.getWidth()
      const pageHeight = pdf.internal.pageSize.getHeight()

      // Mode label
      const modeLabel = method === 'both' ? 'Both (Audio + Hand)' : method === 'audio' ? 'Audio Detection' : 'Hand Detection'

      // Title
      pdf.setFontSize(16)
      pdf.text(modeLabel, pageWidth / 2, 30, { align: 'center' })

      // Stats line
      pdf.setFontSize(10)
      let statsText = `${summary.totalEvents} events`
      if (isCombinedResult && summary.agreement != null) {
        statsText += ` · ${summary.matches}/${summary.comparableEvents} matching labels · ${summary.agreement.toFixed(1)}% agreement`
      } else if (summary.averageScore != null) {
        const scoreName = method === 'hand' ? 'avg proximity score' : 'avg model confidence'
        statsText += ` · ${summary.averageScore.toFixed(1)}% ${scoreName}`
      }
      pdf.text(statsText, pageWidth / 2, 48, { align: 'center' })

      // Calculate image size to fit the page
      const imgWidth = pageWidth - 80
      const imgHeight = (canvas.height * imgWidth) / canvas.width

      // If image is too tall for one page, scale it down or split across pages
      const availableHeight = pageHeight - 70
      if (imgHeight <= availableHeight) {
        const offsetY = 60
        pdf.addImage(imgData, 'PNG', 40, offsetY, imgWidth, imgHeight, undefined, 'FAST')
      } else {
        // Scale to fit width, then paginate
        const scaledWidth = imgWidth
        const scaledHeight = imgHeight
        let yOffset = 0
        let pageNum = 0
        while (yOffset < scaledHeight) {
          if (pageNum > 0) pdf.addPage()
          const sourceY = (yOffset / scaledHeight) * canvas.height
          const sourceH = Math.min((availableHeight / scaledHeight) * canvas.height, canvas.height - sourceY)
          const sliceCanvas = document.createElement('canvas')
          sliceCanvas.width = canvas.width
          sliceCanvas.height = sourceH
          const ctx = sliceCanvas.getContext('2d')
          ctx.drawImage(canvas, 0, sourceY, canvas.width, sourceH, 0, 0, canvas.width, sourceH)
          const sliceData = sliceCanvas.toDataURL('image/png')
          const sliceH = (sourceH * scaledWidth) / canvas.width
          pdf.addImage(sliceData, 'PNG', 40, pageNum === 0 ? 60 : 30, scaledWidth, sliceH, undefined, 'FAST')
          yOffset += availableHeight
          pageNum++
        }
      }
      pdf.save(`${method}-detection-note.pdf`)
    } catch (err) {
      console.error('Failed to generate PDF', err)
    }
  }

  const downloadLog = () => {
    const headers = ['#', 'Time', 'Type', 'String', 'Finger', 'Confidence%', 'Method', 'Distance(px)', 'Status']
    const rows = logs.map((e, i) => {
      const num = e.entry_number ?? String(i + 1).padStart(4, '0')
      const time = formatTime(e.time ?? 0)
      const type = e.type === 'audio' ? 'string' : 'hand'
      const str = e.string ?? ''
      const finger = e.finger ?? ''
      const conf = e.confidence != null ? (e.confidence * 100).toFixed(1) : ''
      const method = e.method ?? ''
      const dist = e.distance != null ? e.distance.toFixed(1) : ''
      const status = e.status ?? ''
      return [num, time, type, str, finger, conf, method, dist, status]
    })
    const csv = [headers.join(','), ...rows.map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(','))].join('\n')
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = jobId ? `detection_log_${jobId}.csv` : 'detection_log.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  const downloadCsv = (type) => {
    if (!jobId) return
    const url = type ? `${API}/download/csv/${jobId}?type=${type}` : `${API}/download/csv/${jobId}`
    window.open(url, '_blank')
  }

  const downloadVideo = (type) => {
    if (!jobId) return
    const url = type ? `${API}/download/video/${jobId}?type=${type}` : `${API}/download/video/${jobId}`
    window.open(url, '_blank')
  }

  const downloadManifest = () => {
    if (!jobId) return
    window.open(`${API}/download/manifest/${jobId}`, '_blank')
  }

  const seekToTime = (timeInSeconds) => {
    if (videoRef.current != null && !isNaN(timeInSeconds)) {
      videoRef.current.currentTime = timeInSeconds
      setCurrentTime(timeInSeconds)
    }
  }

  const goToNextPluck = () => {
    const next = gridRows.find((r) => r.time > currentTime)
    if (next) seekToTime(next.time)
  }

  const goToPrevPluck = () => {
    const prev = [...gridRows].reverse().find((r) => r.time < currentTime)
    if (prev) seekToTime(prev.time)
  }

  const hasBothAudioHand = summary.comparableEvents > 0
  const noteRows = useMemo(() => buildNoteRows(gridRows), [gridRows])

  // User profile from localStorage
  const userName = localStorage.getItem('user_name') || ''
  const userAvatar = localStorage.getItem('user_avatar') || ''
  const userEmail = localStorage.getItem('user_email') || ''
  const userInitials = userName
    ? userName.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)
    : (userEmail ? userEmail[0].toUpperCase() : 'U')
  const needsAudioModel = method === 'audio' || method === 'both'
  const needsHandWeights = method === 'hand' || method === 'both'
  const audioResourceReady = !needsAudioModel
    || (useDefaultModel && defaultsAvailable.default_model)
    || (!useDefaultModel && defaultsAvailable.allow_custom_model_uploads && modelFile)
  const handResourceReady = !needsHandWeights
    || (useDefaultWeights && defaultsAvailable.default_weights)
    || (!useDefaultWeights && defaultsAvailable.allow_custom_model_uploads && weightsFile)
  const runtimeReady = !needsAudioModel || defaultsAvailable.ffmpeg_available
  const canRun = apiState === 'ready' && Boolean(videoFile) && audioResourceReady && handResourceReady && runtimeReady && !uploading

  return (
    <div className="app app-showcase">
      <header className="tool-topbar">
        <Link to="/" className="tool-brand">
          <span className="tool-brand-mark" aria-hidden="true">NSN</span>
          <span>
            <strong>Nat Shin Naung</strong>
            <small>Saung analysis studio</small>
          </span>
        </Link>
        <div className="tool-topbar-actions">
          <Link to="/" className="tool-back-link">Project overview</Link>
          {userName && (
            <div className="user-profile-badge" title={userName + (userEmail ? `\n${userEmail}` : '')}>
              {userAvatar ? (
                <img src={userAvatar} alt={userName} className="user-profile-avatar" referrerPolicy="no-referrer" />
              ) : (
                <span className="user-profile-initials">{userInitials}</span>
              )}
              <span className="user-profile-name">{userName}</span>
            </div>
          )}
        </div>
      </header>

      <section className="tool-hero" aria-labelledby="tool-title">
        <div>
          <p className="tool-eyebrow">Myanmar harp · 16-string detection</p>
          <h1 id="tool-title">Turn a performance into a reviewable string timeline.</h1>
          <p className="tool-hero-copy">
            Upload one clear video. HarpHand finds pluck moments, labels likely strings, and creates an annotated result you can inspect, teach from, or export.
          </p>
        </div>
        <div className="tool-hero-facts" aria-label="Analysis capabilities">
          <span><strong>16</strong> strings</span>
          <span><strong>3</strong> analysis modes</span>
          <span><strong>1</strong> synced timeline</span>
        </div>
      </section>

      <main className="main tool-main">
        <div className="tool-workspace">
          <section className="card upload-card workflow-card">
            <div className="workflow-card-heading">
              <div>
                <p className="section-kicker">New analysis</p>
                <h2>Prepare your performance</h2>
              </div>
              <span className={`service-pill service-pill-${apiState}`}>
                {apiState === 'ready' ? 'Service ready' : apiState === 'checking' ? 'Connecting' : 'Service offline'}
              </span>
            </div>

            <form onSubmit={handleSubmit} className="upload-form showcase-form">
              <fieldset className="workflow-step">
                <legend>
                  <span className="step-number">01</span>
                  <span><strong>Choose the signal</strong><small>What should the analysis listen to or watch?</small></span>
                </legend>
                <div className="method-grid">
                  {DETECTION_METHODS.map((option) => (
                    <label key={option.value} className={`method-card ${method === option.value ? 'method-card-selected' : ''}`}>
                      <input
                        type="radio"
                        name="method"
                        value={option.value}
                        checked={method === option.value}
                        onChange={() => setMethod(option.value)}
                      />
                      <span className="method-card-topline">
                        <strong>{option.label}</strong>
                        <small>{option.eyebrow}</small>
                      </span>
                      <span className="method-card-copy">{option.description}</span>
                    </label>
                  ))}
                </div>
              </fieldset>

              <fieldset className="workflow-step">
                <legend>
                  <span className="step-number">02</span>
                  <span><strong>Add a video</strong><small>Keep the harp visible and the audio free of heavy background noise.</small></span>
                </legend>
                <label
                  className={`video-dropzone ${isDraggingVideo ? 'video-dropzone-active' : ''} ${videoFile ? 'video-dropzone-ready' : ''}`}
                  onDragEnter={(event) => { event.preventDefault(); setIsDraggingVideo(true) }}
                  onDragOver={(event) => event.preventDefault()}
                  onDragLeave={() => setIsDraggingVideo(false)}
                  onDrop={(event) => {
                    event.preventDefault()
                    setIsDraggingVideo(false)
                    handleVideoSelection(event.dataTransfer.files?.[0] ?? null)
                  }}
                >
                  <input
                    id="video"
                    ref={videoInput}
                    type="file"
                    accept=".mp4,.mov,.mkv,.avi,.webm"
                    onChange={(event) => handleVideoSelection(event.target.files?.[0] ?? null)}
                  />
                  <span className="dropzone-badge" aria-hidden="true">VIDEO</span>
                  {videoFile ? (
                    <span className="dropzone-copy">
                      <strong>{videoFile.name}</strong>
                      <small>{(videoFile.size / (1024 * 1024)).toFixed(1)} MB · ready to analyze</small>
                    </span>
                  ) : (
                    <span className="dropzone-copy">
                      <strong>Drop a performance here or browse</strong>
                      <small>MP4, MOV, MKV, AVI, or WebM · up to 250 MB</small>
                    </span>
                  )}
                </label>
              </fieldset>

              <details className="advanced-settings">
                <summary>
                  <span><strong>Analysis settings</strong><small>Bundled models are selected automatically.</small></span>
                  <span className="summary-action">Adjust</span>
                </summary>
                <div className="advanced-settings-body">
                  {(method === 'audio' || method === 'both') && (
                    <div className="settings-block">
                      <div>
                        <strong>Audio detection</strong>
                        <p>Hybrid mode uses pitch estimation when model confidence is low.</p>
                      </div>
                      <div className="segmented-control" aria-label="Audio detection mode">
                        <label className={mode === 'hybrid' ? 'selected' : ''}>
                          <input type="radio" name="mode" value="hybrid" checked={mode === 'hybrid'} onChange={() => setMode('hybrid')} />
                          Hybrid
                        </label>
                        <label className={mode === 'default' ? 'selected' : ''}>
                          <input type="radio" name="mode" value="default" checked={mode === 'default'} onChange={() => setMode('default')} />
                          Model only
                        </label>
                      </div>
                    </div>
                  )}

                  <div className="model-readiness-grid">
                    {needsAudioModel && (
                      <div className={`model-readiness ${defaultsAvailable.default_model ? 'ready' : 'missing'}`}>
                        <span>Audio model</span>
                        <strong>{defaultsAvailable.default_model ? 'Bundled and ready' : 'Not installed'}</strong>
                      </div>
                    )}
                    {needsAudioModel && (
                      <div className={`model-readiness ${defaultsAvailable.calibrated_thresholds ? 'ready' : ''}`}>
                        <span>Threshold profile</span>
                        <strong>
                          {defaultsAvailable.calibrated_thresholds
                            ? 'Validation calibrated'
                            : 'Built-in baseline'}
                        </strong>
                      </div>
                    )}
                    {needsAudioModel && (
                      <div className={`model-readiness ${defaultsAvailable.ffmpeg_available ? 'ready' : 'missing'}`}>
                        <span>Audio runtime</span>
                        <strong>{defaultsAvailable.ffmpeg_available ? 'FFmpeg ready' : 'FFmpeg missing'}</strong>
                      </div>
                    )}
                    {needsHandWeights && (
                      <div className={`model-readiness ${defaultsAvailable.default_weights ? 'ready' : 'missing'}`}>
                        <span>Hand model</span>
                        <strong>{defaultsAvailable.default_weights ? 'Bundled and ready' : 'Not installed'}</strong>
                      </div>
                    )}
                  </div>

                  {defaultsAvailable.allow_custom_model_uploads && (
                    <div className="custom-model-settings">
                      <p className="field-hint">Developer mode is enabled. Custom model files are loaded by the backend; only use files you trust.</p>
                      {needsAudioModel && (
                        <div className="field">
                          <label>Audio model source</label>
                          <div className="source-choice-row">
                            <label><input type="radio" name="modelSource" checked={useDefaultModel} onChange={() => { setUseDefaultModel(true); setModelFile(null) }} /> Bundled</label>
                            <label><input type="radio" name="modelSource" checked={!useDefaultModel} onChange={() => setUseDefaultModel(false)} /> Custom</label>
                          </div>
                          {!useDefaultModel && <input id="model" ref={modelInput} type="file" accept=".keras" onChange={(event) => setModelFile(event.target.files?.[0] ?? null)} />}
                        </div>
                      )}
                      {needsHandWeights && (
                        <div className="field">
                          <label>Hand model source</label>
                          <div className="source-choice-row">
                            <label><input type="radio" name="weightsSource" checked={useDefaultWeights} onChange={() => { setUseDefaultWeights(true); setWeightsFile(null) }} /> Bundled</label>
                            <label><input type="radio" name="weightsSource" checked={!useDefaultWeights} onChange={() => setUseDefaultWeights(false)} /> Custom</label>
                          </div>
                          {!useDefaultWeights && <input id="weights" ref={weightsInput} type="file" accept=".pt" onChange={(event) => setWeightsFile(event.target.files?.[0] ?? null)} />}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </details>

              {apiState === 'unavailable' && (
                <p className="service-message" role="status">
                  The interface is ready, but the analysis service did not respond. Start the FastAPI backend to run a video.
                </p>
              )}
              {error && <p className="error" role="alert">{error}</p>}
              <button type="submit" className="btn primary run-analysis-button" disabled={!canRun}>
                {uploading ? 'Uploading performance…' : videoFile ? 'Analyze this performance' : 'Choose a video to continue'}
              </button>
              <p className="privacy-note">Your video is processed for this analysis and is not used to retrain the model.</p>
            </form>
          </section>

          <aside className="showcase-sidebar" aria-label="Recording guidance">
            <div className="sidebar-visual">
              <img src={harpImage} alt="Traditional Myanmar harp" />
              <div className="sidebar-visual-copy">
                <span>Saung</span>
                <strong>16-string<br />analysis</strong>
              </div>
            </div>
            <div className="sidebar-panel">
              <p className="section-kicker">For a cleaner result</p>
              <ol className="capture-tips">
                <li><span>01</span><p><strong>Frame the full string area.</strong> Keep hands and strings visible together.</p></li>
                <li><span>02</span><p><strong>Use steady light.</strong> Reflections and motion blur weaken hand tracking.</p></li>
                <li><span>03</span><p><strong>Record clean audio.</strong> Reduce speech and other instruments where possible.</p></li>
              </ol>
            </div>
            <div className="sidebar-disclaimer">
              <strong>About the result</strong>
              <p>Agreement compares the audio and hand labels. It is useful diagnostic evidence, not a measured accuracy score.</p>
            </div>
          </aside>
        </div>

        {status && (
          <section className="card status-card">
            <h3>Analysis status</h3>
            {status.status === 'queued' && <p className="running">{status.message || 'Waiting to start analysis…'}</p>}
            {status.status === 'running' && (
              <p className="running">{status.message || 'Processing…'}</p>
            )}
            {(status.status === 'queued' || status.status === 'running') && (
              <div
                className="status-progress"
                role="progressbar"
                aria-label="Analysis progress"
                aria-valuemin="0"
                aria-valuemax="100"
                aria-valuenow={status.progress ?? 0}
              >
                <span style={{ width: `${status.progress ?? 8}%` }} />
              </div>
            )}
            {status.status === 'error' && (
              <p className="error">{status.message}</p>
            )}
            {status.status === 'done' && (
              <div className="done">
                {status.audio && status.hand ? (
                  <>
                    <p className="success">
                      Done. Audio: {status.audio.rows ?? 0} onset(s). Hand: {status.hand.rows ?? 0} touch(es).
                    </p>
                    {status.combined ? (
                      <div className="actions" style={{ marginBottom: '0.5rem' }}>
                        <button type="button" className="btn primary" onClick={() => downloadVideo('combined')} style={{ fontSize: '1rem', padding: '0.75rem 1.5rem' }}>
                          Combined annotated video
                        </button>
                      </div>
                    ) : status.combined_error && (
                      <p className="error" style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>
                        Combined video: {status.combined_error}
                      </p>
                    )}
                    <div className="actions actions-both">
                      <button type="button" className="btn primary" onClick={() => downloadCsv('audio')}>
                        Audio CSV
                      </button>
                      <button type="button" className="btn secondary" onClick={() => downloadVideo('audio')}>
                        Audio video
                      </button>
                      <button type="button" className="btn primary" onClick={() => downloadCsv('hand')}>
                        Hand CSV
                      </button>
                      <button type="button" className="btn secondary" onClick={() => downloadVideo('hand')}>
                        Hand video
                      </button>
                    </div>
                    <div className="actions" style={{ marginTop: '0.5rem' }}>
                      <button type="button" className="btn primary" onClick={downloadLog}>
                        Detection log (CSV)
                      </button>
                    </div>
                  </>
                ) : status.audio ? (
                  <>
                    <p className="success">
                      Done (audio only). {status.audio.rows ?? 0} onset(s).
                    </p>
                    {status.hand_error && (
                      <p className="error" style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>
                        Hand detection: {status.hand_error}
                      </p>
                    )}
                    <div className="actions">
                      <button type="button" className="btn primary" onClick={() => downloadCsv('audio')}>
                        Download CSV
                      </button>
                      <button type="button" className="btn secondary" onClick={() => downloadVideo('audio')}>
                        Download video
                      </button>
                    </div>
                    <div className="actions" style={{ marginTop: '0.5rem' }}>
                      <button type="button" className="btn primary" onClick={downloadLog}>
                        Detection log (CSV)
                      </button>
                    </div>
                  </>
                ) : (
                  <>
                    <p className="success">
                      Done. Detected {status.rows ?? 0} {method === 'hand' ? 'touch event(s)' : 'onset(s)'}.
                    </p>
                    <div className="actions">
                      <button type="button" className="btn primary" onClick={() => downloadCsv()}>
                        Download CSV
                      </button>
                      <button type="button" className="btn secondary" onClick={() => downloadVideo()}>
                        Download {method === 'hand' ? 'annotated' : 'labeled'} video
                      </button>
                    </div>
                    <div className="actions" style={{ marginTop: '0.5rem' }}>
                      <button type="button" className="btn primary" onClick={downloadLog}>
                        Detection log (CSV)
                      </button>
                    </div>
                  </>
                )}
                <div className="actions" style={{ marginTop: '0.5rem' }}>
                  {method !== 'hand' && (
                    <button type="button" className="btn secondary" onClick={downloadManifest}>
                      Inference manifest
                    </button>
                  )}
                  <button type="button" className="btn ghost" onClick={reset}>
                    New run
                  </button>
                </div>
              </div>
            )}
          </section>
        )}

        {status?.status === 'done' && videoUrl && (
          <section className="card preview-card">
            <h3>Preview</h3>
            {gridRows.length > 0 && (
              <div className="preview-nav">
                <button type="button" className="btn ghost" onClick={goToPrevPluck} title="Previous pluck">
                  ← Prev pluck
                </button>
                <button type="button" className="btn ghost" onClick={goToNextPluck} title="Next pluck">
                  Next pluck →
                </button>
              </div>
            )}
            <div className="video-container">
              <video
                ref={videoRef}
                controls
                src={videoUrl}
                className="preview-video"
                onLoadedMetadata={() => {
                  if (videoRef.current && isFinite(videoRef.current.duration))
                    setVideoDuration(videoRef.current.duration)
                }}
                onTimeUpdate={() => {
                  if (videoRef.current) setCurrentTime(videoRef.current.currentTime)
                }}
                onSeeked={() => {
                  if (videoRef.current) setCurrentTime(videoRef.current.currentTime)
                }}
                onPlay={() => {
                  if (videoRef.current) setCurrentTime(videoRef.current.currentTime)
                }}
              >
                Your browser does not support the video tag.
              </video>
            </div>
            {gridRows.length > 0 && videoDuration > 0 && (
              <div className="timeline-strip" title="Click to seek">
                <div
                  className="timeline-playhead"
                  style={{ left: `${(currentTime / videoDuration) * 100}%` }}
                />
                {gridRows.map((row) => (
                  <button
                    key={`tl-${row.time}-${row.index}`}
                    type="button"
                    className={`timeline-marker ${row.match ? 'timeline-marker-match' : 'timeline-marker-miss'}`}
                    style={{ left: `${(row.time / videoDuration) * 100}%` }}
                    onClick={() => seekToTime(row.time)}
                    title={`${formatTime(row.time)} ${row.match ? '✓' : ''}`}
                  />
                ))}
              </div>
            )}
          </section>
        )}

        {status?.status === 'done' && logs.length > 0 && (
          <section className="card log-card">
            <div className="log-card-header">
              <h3>Detection Log (synced to video · {formatTime(currentTime)})</h3>
              <div className="log-view-toggle">
                <button
                  type="button"
                  className={`btn ghost ${logViewMode === 'list' ? 'active' : ''}`}
                  onClick={() => setLogViewMode('list')}
                >
                  List
                </button>
                <button
                  type="button"
                  className={`btn ghost ${logViewMode === 'grid' ? 'active' : ''}`}
                  onClick={() => setLogViewMode('grid')}
                >
                  Grid
                </button>
              </div>
            </div>
            <div className="log-download-row">
              <span className="log-summary">
                {gridRows.length} events
                {isCombinedResult && summary.agreement != null ? (
                  <> · {summary.matches}/{summary.comparableEvents} matching labels · {summary.agreement.toFixed(1)}% agreement · {summary.handCoverage.toFixed(1)}% hand coverage</>
                ) : summary.averageScore != null ? (
                  <> · {summary.averageScore.toFixed(1)}% {method === 'hand' ? 'avg proximity score' : 'avg model confidence'}</>
                ) : null}
              </span>
              <button type="button" className="btn primary" onClick={downloadLog} title="Download full detection log as CSV">
                Download log (CSV)
              </button>
            </div>
            {logViewMode === 'list' ? (
              <div ref={logPanelRef} className="log-panel">
                {visibleLogs.length === 0 ? (
                  <p className="muted" style={{ padding: '1rem', textAlign: 'center' }}>
                    No events at this time. Play the video to see detections.
                  </p>
                ) : (
                  visibleLogs.map((event, idx) => (
                    <div
                      key={event.entry_number || idx}
                      role="button"
                      tabIndex={0}
                      className="log-entry log-entry-active log-entry-clickable"
                      onClick={() => seekToTime(event.time)}
                      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); seekToTime(event.time) } }}
                      title="Click to seek video to this time"
                    >
                      <span className="log-entry-number">{event.entry_number || `${(idx + 1).toString().padStart(4, '0')}`}</span>
                      <span className="log-time">{formatTime(event.time)}</span>
                      <span className={`log-type log-type-${event.type}`}>
                        {event.type === 'audio' ? 'string' : 'hand'}
                      </span>
                      <span className="log-string">{event.string}</span>
                      {event.type === 'audio' && (
                        <span style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
                          <span className="log-method">{event.method === 'yin' || event.method === 'string*' ? 'String*' : (event.method === 'model' || event.method === 'string' ? 'String' : (event.method || 'String'))}</span>
                          <span className="log-confidence">{(event.confidence * 100).toFixed(1)}%</span>
                        </span>
                      )}
                      {event.type === 'hand' && (
                        <span style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
                          <span className="log-status">{event.status || 'detected'}</span>
                          <span className="log-finger">{event.finger || '-'}</span>
                          <span className="log-distance">{event.distance ? event.distance.toFixed(1) : '0.0'}px</span>
                          <span className="log-confidence">{(event.confidence * 100).toFixed(1)}%</span>
                        </span>
                      )}
                    </div>
                  ))
                )}
              </div>
            ) : (
              <div ref={logPanelRef} className="log-panel log-grid-wrap">
                <table className="log-grid">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Time</th>
                      <th>String</th>
                      <th>Hand</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {gridRows.map((row) => (
                      <tr
                        key={`${row.time}-${row.index}`}
                        role="button"
                        tabIndex={0}
                        className={`log-grid-row-clickable ${row.inWindow ? 'log-grid-row-active' : ''}`}
                        onClick={() => seekToTime(row.time)}
                        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); seekToTime(row.time) } }}
                        title="Click to seek video to this time"
                      >
                        <td className="log-grid-num">{row.index}</td>
                        <td className="log-grid-time">{formatTime(row.time)}</td>
                        <td className="log-grid-cell">
                          <span className="log-grid-main">{row.stringMain}</span>
                          {row.note ? <span className="log-grid-annot">{row.note}</span> : null}
                        </td>
                        <td className={`log-grid-cell ${row.match ? 'log-grid-cell-match' : ''}`}>
                          <span className="log-grid-main">{row.handMain}</span>
                        </td>
                        <td className="log-grid-note">{row.match ? '✓' : ''}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}

        {status?.status === 'done' && logs.length > 0 && (
          <section className="card generated-note-card">
            <div className="generated-note-header-row">
              <div>
                <h3 style={{ margin: 0 }}>
                  {method === 'both' ? 'Both (Audio + Hand)' : method === 'audio' ? 'Audio Detection' : 'Hand Detection'}
                  {' '}
                  — Note
                </h3>
                <span style={{ fontSize: '0.85rem', color: 'var(--muted)' }}>
                  {gridRows.length} events
                  {isCombinedResult && summary.agreement != null ? (
                    <> · {summary.matches}/{summary.comparableEvents} matching labels · {summary.agreement.toFixed(1)}% agreement</>
                  ) : summary.averageScore != null ? (
                    <> · {summary.averageScore.toFixed(1)}% {method === 'hand' ? 'avg proximity score' : 'avg model confidence'}</>
                  ) : null}
                </span>
              </div>
              <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                <div className="log-view-toggle">
                  <button
                    type="button"
                    className={`btn ghost ${noteFormat === 'number' ? 'active' : ''}`}
                    onClick={() => setNoteFormat('number')}
                    title="Show string numbers (1–16)"
                  >
                    S#
                  </button>
                  <button
                    type="button"
                    className={`btn ghost ${noteFormat === 'note' ? 'active' : ''}`}
                    onClick={() => setNoteFormat('note')}
                    title="Show Western note names (G5–G2)"
                  >
                    Note
                  </button>
                </div>
                {noteRows.length > 0 && (
                  <button type="button" className="btn secondary btn-sm" onClick={handleDownloadNotePdf}>
                    Download as PDF
                  </button>
                )}
              </div>
            </div>
            <p className="generated-note-desc">
              {noteFormat === 'note'
                ? 'Each box is one sound event shown as Western note name. Strings plucked together = notes with underline.'
                : 'Each box is one sound event. Single string = number only; strings plucked together = numbers with underline.'}
            </p>
            {noteRows.length > 0 ? (
              <div className="generated-note-grid-wrap" ref={generatedNoteRef}>
                <div className="generated-note-grid" style={{ gridTemplateColumns: `repeat(${NOTE_COLUMNS}, 1fr)` }}>
                  {noteRows.map((row, ri) =>
                    row.map((cell, ci) => {
                      const flatIndex = ri * NOTE_COLUMNS + ci
                      const eventTime = gridRows[flatIndex]?.time
                      const hasParts = cell.parts && cell.parts.length > 0
                      return (
                        <div
                          key={`${ri}-${ci}`}
                          className={`generated-note-cell${hasParts ? ' generated-note-cell-clickable' : ''}`}
                          role={hasParts ? 'button' : undefined}
                          tabIndex={hasParts ? 0 : -1}
                          onClick={() => hasParts && eventTime != null && seekToTime(eventTime)}
                          onKeyDown={(e) => {
                            if (!hasParts) return
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.preventDefault()
                              if (eventTime != null) seekToTime(eventTime)
                            }
                          }}
                          title={hasParts && eventTime != null ? `Seek to ${formatTime(eventTime)}` : ''}
                        >
                          {hasParts && (
                            <span className={cell.together ? 'generated-note-together' : ''}>
                              {cell.parts.map((p, idx) => (
                                <span key={idx} className="generated-note-token">
                                  <span
                                    className={
                                      'generated-note-thumb-dot' +
                                      (p.thumb ? '' : ' generated-note-thumb-dot--placeholder')
                                    }
                                    aria-hidden
                                  >
                                    ·
                                  </span>
                                  <span>{noteFormat === 'note' ? (STRING_TO_NOTE[p.num] || p.num) : p.num}</span>
                                </span>
                              ))}
                            </span>
                          )}
                        </div>
                      )
                    })
                  )}
                </div>
              </div>
            ) : (
              <p className="muted">No events to show.</p>
            )}
          </section>
        )}

        {status?.status === 'done' && logs.length > 0 && hasBothAudioHand && (
          <section className="card analysis-card">
            <h3>Analysis</h3>
            <div className="analysis-section">
              <h4 className="analysis-subtitle">Per-string match rate</h4>
              <div className="per-string-table-wrap">
                <table className="per-string-table">
                  <thead>
                    <tr>
                      <th>String</th>
                      <th>Plucks</th>
                      <th>Matches</th>
                      <th>%</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].map((s) => {
                      const key = `S${s}`
                      const st = perStringStats[key]
                      if (!st || st.total === 0) return null
                      const pct = st.total > 0 ? ((st.matches / st.total) * 100).toFixed(0) : '0'
                      return (
                        <tr key={key}>
                          <td className="per-string-name">{key}</td>
                          <td>{st.total}</td>
                          <td>{st.matches}</td>
                          <td className="per-string-pct">{pct}%</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="analysis-section">
              <h4 className="analysis-subtitle">Audio vs hand (agreement)</h4>
              <p className="muted analysis-hint">Count of times each (audio string, hand string) pair occurred.</p>
              <div className="agreement-matrix-wrap">
                <table className="agreement-matrix">
                  <thead>
                    <tr>
                      <th></th>
                      {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].map((j) => (
                        <th key={j}>S{j}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].map((i) => {
                      const row = agreementMatrix[`S${i}`]
                      const hasAny = row && Object.values(row).some((v) => v > 0)
                      if (!hasAny) return null
                      return (
                        <tr key={`row-S${i}`}>
                          <th>S{i}</th>
                          {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].map((j) => {
                            const v = row[`S${j}`] || 0
                            return (
                              <td key={j} className={v > 0 ? 'agreement-cell' : ''} title={`Audio S${i} / Hand S${j}`}>
                                {v > 0 ? v : ''}
                              </td>
                            )
                          })}
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>NAT SHIN NAUNG · Audio · Hand · Both · For teaching / presentation</p>
      </footer>
    </div>
  )
}
