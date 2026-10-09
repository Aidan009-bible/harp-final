import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import './Home.css'
import harpHero from '../assets/harp_2.jpg'
import harpDetail from '../assets/harp_3.jpg'
import BrandLogo from '../components/BrandLogo.jsx'
import ThemeToggle from '../components/ThemeToggle.jsx'

const SIGNAL_LEVELS = [42, 70, 36, 84, 54, 92, 47, 76, 58, 88, 40, 68, 50, 80, 62, 44]

const CAPABILITIES = [
  {
    number: '01',
    title: 'Hear the attack',
    label: 'Audio inference',
    description: 'Find onset moments, score all sixteen strings, and retain the confidence evidence behind each prediction.',
  },
  {
    number: '02',
    title: 'Follow the gesture',
    label: 'Hand geometry',
    description: 'Track fingertips against the instrument’s detected strings with resolution-aware proximity measurements.',
  },
  {
    number: '03',
    title: 'Question the result',
    label: 'Synced review',
    description: 'Compare both signals on one timeline, inspect disagreements, and export the underlying events.',
  },
]

export default function Home() {
  useEffect(() => {
    const elements = document.querySelectorAll('[data-reveal]')
    if (!('IntersectionObserver' in window)) {
      elements.forEach((element) => element.classList.add('is-visible'))
      return undefined
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return
          entry.target.classList.add('is-visible')
          observer.unobserve(entry.target)
        })
      },
      { threshold: 0.14 },
    )

    elements.forEach((element) => observer.observe(element))
    return () => observer.disconnect()
  }, [])

  return (
    <div className="home-page">
      <a className="skip-link" href="#main-content">Skip to content</a>
      <div className="home-aurora home-aurora-one" aria-hidden="true" />
      <div className="home-aurora home-aurora-two" aria-hidden="true" />

      <header className="home-header" data-reveal>
        <Link to="/" className="home-brand" aria-label="Nat Shin Naung home">
          <BrandLogo />
          <span>
            <strong>Nat Shin Naung</strong>
            <small>Saung intelligence project</small>
          </span>
        </Link>
        <nav className="home-nav" aria-label="Primary navigation">
          <a href="#capabilities">System</a>
          <a href="#workflow">Workflow</a>
          <a href="#research">Research</a>
        </nav>
        <div className="home-header-actions">
          <ThemeToggle />
          <Link to="/tool" className="home-nav-cta">Enter studio <span aria-hidden="true">↗</span></Link>
        </div>
      </header>

      <main id="main-content">
        <section className="home-hero" aria-labelledby="home-title">
          <div className="home-hero-copy" data-reveal>
            <p className="home-kicker"><span aria-hidden="true" /> Myanmar harp · multimodal detection</p>
            <h1 id="home-title">Hear every string.<br /><em>See every gesture.</em></h1>
            <p className="home-lede">
              A research studio that turns recorded saung performances into inspectable string events—combining audio inference, hand tracking, and an evidence-first review timeline.
            </p>
            <div className="home-hero-actions">
              <Link to="/tool" className="action-primary">Analyze a performance <span aria-hidden="true">↗</span></Link>
              <a href="#workflow" className="action-secondary">Explore the method</a>
            </div>
            <dl className="home-proof" aria-label="Project facts">
              <div><dt>16</dt><dd>strings modeled</dd></div>
              <div><dt>03</dt><dd>analysis modes</dd></div>
              <div><dt>01</dt><dd>review timeline</dd></div>
            </dl>
          </div>

          <div className="instrument-stage" data-reveal>
            <figure className="instrument-frame">
              <img src={harpHero} alt="Traditional Myanmar saung harp viewed from the side" />
              <figcaption>Traditional form · computational reading</figcaption>
            </figure>
            <div className="stage-status" aria-hidden="true">
              <span className="status-beacon" />
              <div><small>Pipeline</small><strong>Audio + hand aligned</strong></div>
            </div>
            <div className="stage-event" aria-hidden="true">
              <span>Event 024</span>
              <strong>S07</strong>
              <small>00:12.48 · 91%</small>
            </div>
          </div>
        </section>

        <section className="signal-deck" aria-labelledby="signal-title" data-reveal>
          <div className="signal-deck-heading">
            <div>
              <p className="section-label">Live system language</p>
              <h2 id="signal-title">Sixteen strings. One readable signal.</h2>
            </div>
            <p>Each event keeps its time, label, source, and score. The interface shows evidence—not a magic accuracy number.</p>
          </div>
          <div className="signal-strings" aria-label="Decorative sixteen-string signal visualization">
            {SIGNAL_LEVELS.map((level, index) => (
              <div
                className="signal-string"
                key={index}
                style={{ '--signal-level': `${level}%`, '--signal-delay': `${index * 70}ms` }}
              >
                <span className="signal-string-track"><i /></span>
                <small>{String(index + 1).padStart(2, '0')}</small>
              </div>
            ))}
          </div>
        </section>

        <section id="capabilities" className="capability-section">
          <div className="section-heading" data-reveal>
            <p className="section-label">What the system does</p>
            <h2>A performance becomes<br /><em>reviewable evidence.</em></h2>
          </div>
          <div className="capability-grid">
            {CAPABILITIES.map((item, index) => (
              <article className="capability-card" data-reveal key={item.number} style={{ '--card-delay': `${index * 90}ms` }}>
                <div className="capability-card-top"><span>{item.number}</span><small>{item.label}</small></div>
                <h3>{item.title}</h3>
                <p>{item.description}</p>
                <div className="capability-rule" aria-hidden="true"><span /></div>
              </article>
            ))}
          </div>
        </section>

        <section id="workflow" className="workflow-story" data-reveal>
          <figure className="workflow-image">
            <img src={harpDetail} alt="Gold and red Myanmar harp on a dark blue background" />
            <figcaption>Saung · form, motion, resonance</figcaption>
          </figure>
          <div className="workflow-copy">
            <p className="section-label">The review loop</p>
            <h2>Record once.<br /><em>Inspect every layer.</em></h2>
            <ol>
              <li><span>01</span><div><strong>Frame the performance</strong><p>Keep the full string field and both hands visible with clean, unclipped audio.</p></div></li>
              <li><span>02</span><div><strong>Run the signals</strong><p>Choose audio, hand tracking, or the combined comparison workflow.</p></div></li>
              <li><span>03</span><div><strong>Challenge the output</strong><p>Seek to each event, inspect agreement, and export logs, video, notes, and the inference manifest.</p></div></li>
            </ol>
            <Link to="/tool" className="text-link">Open the analysis workspace <span aria-hidden="true">→</span></Link>
          </div>
        </section>

        <section id="research" className="research-section" data-reveal>
          <div>
            <p className="section-label">Research posture</p>
            <h2>Built to show its work.</h2>
          </div>
          <p>
            HarpHand is a research and teaching prototype. Signal agreement is diagnostic evidence, not ground-truth accuracy. Model checksums, thresholds, timing, and event-level exports keep each run traceable.
          </p>
          <div className="research-tags" aria-label="Research qualities">
            <span>Inspectable</span><span>Reproducible</span><span>Exportable</span>
          </div>
        </section>

        <section className="home-cta" data-reveal>
          <p className="section-label">The studio is ready</p>
          <h2>Bring one clear performance.<br /><em>Leave with a timeline.</em></h2>
          <Link to="/tool" className="action-primary">Begin analysis <span aria-hidden="true">↗</span></Link>
        </section>
      </main>

      <footer className="home-footer">
        <Link to="/" className="home-brand">
          <BrandLogo />
          <span><strong>Nat Shin Naung</strong><small>Myanmar harp research</small></span>
        </Link>
        <p>Audio · Hand · Evidence</p>
        <p>Research prototype</p>
      </footer>
    </div>
  )
}
