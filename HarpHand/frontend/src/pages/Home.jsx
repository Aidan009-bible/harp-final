import { Link } from 'react-router-dom';
import './Home.css';
import harpImage from '../assets/myanmar_harp.jpg';

export default function Home() {
  return (
    <div className="vintage-page">
      <div className="vintage-rivets vintage-rivets-left" aria-hidden />
      <div className="vintage-rivets vintage-rivets-right" aria-hidden />

      <header className="vintage-header">
        <div className="vintage-header-inner">
          <Link to="/" className="vintage-logo">NAT SHIN NAUNG</Link>
          <nav className="vintage-nav">
            <Link to="/">HOME</Link>
            <a href="#benefits">BENEFITS</a>
            <a href="#workflow">WORKFLOW</a>
            <a href="#contact">CONTACT</a>
          </nav>
          <Link to="/tool" className="vintage-btn vintage-btn-cta">OPEN STUDIO</Link>
        </div>
      </header>

      <main className="vintage-main">
        <section className="vintage-hero">
          <div className="vintage-hero-banner">
            <div className="vintage-hero-content">
              <p className="vintage-hero-kicker">Research prototype · Myanmar saung</p>
              <h1 className="vintage-hero-title">Harp String Detection</h1>
              <p className="vintage-hero-desc">
                Turn a recorded performance into a reviewable timeline of likely string plucks. Compare audio and hand signals, inspect every event, and export the evidence.
              </p>
              <div className="vintage-hero-actions">
                <Link to="/tool" className="vintage-btn vintage-btn-primary">ANALYZE A PERFORMANCE</Link>
                <a href="#workflow" className="vintage-btn vintage-btn-secondary">SEE THE WORKFLOW</a>
              </div>
            </div>
            <figure className="vintage-hero-media">
              <img src={harpImage} alt="Traditional Myanmar harp used for the detection project" />
              <figcaption>Audio model · string detector · hand landmarks</figcaption>
            </figure>
            <div className="vintage-chains" aria-hidden />
          </div>
        </section>

        <section id="benefits" className="vintage-benefits">
          <h2 className="vintage-section-title">Benefits</h2>
          <p className="vintage-section-subtitle">
            This tool helps musicians and teachers visualize and document harp string plucks from video—with optional hand and audio analysis.
          </p>
          <div className="vintage-benefits-grid">
            <div className="vintage-benefit-card">
              <div className="vintage-benefit-icon" aria-hidden>
                <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M24 8v8l6 6M18 14l6 6 12-12M12 28l6 6 12-12" /></svg>
              </div>
              <h3>Easy to use</h3>
              <p>Upload a video, choose audio and/or hand detection, and get timestamps and note sheets.</p>
            </div>
            <div className="vintage-benefit-card">
              <div className="vintage-benefit-icon" aria-hidden>
                <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M8 24h32M24 8v32M16 16l16 16M32 16L16 32" /></svg>
              </div>
              <h3>Audio + Hand</h3>
              <p>Compare two independent signals and see where their string labels agree or diverge.</p>
            </div>
            <div className="vintage-benefit-card">
              <div className="vintage-benefit-icon" aria-hidden>
                <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="24" cy="24" r="18" /><path d="M24 14v10l6 6" /></svg>
              </div>
              <h3>Reviewable</h3>
              <p>Jump from each detected event to its video moment and inspect the evidence yourself.</p>
            </div>
            <div className="vintage-benefit-card">
              <div className="vintage-benefit-icon" aria-hidden>
                <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M12 12h24v24H12z" /><path d="M18 24l6 6 12-12" /></svg>
              </div>
              <h3>Export</h3>
              <p>Download CSV logs, annotated video, and PDF note sheets for your records.</p>
            </div>
          </div>
          <div className="vintage-gears vintage-gears-bottom" aria-hidden />
        </section>

        <section id="workflow" className="vintage-video vintage-workflow">
          <div>
            <p className="vintage-section-kicker">A clear research workflow</p>
            <h2 className="vintage-section-title">From performance to evidence</h2>
            <p className="vintage-section-subtitle">The system keeps the original video, detected events, signal agreement, and exports connected in one review flow.</p>
          </div>
          <ol className="workflow-preview">
            <li><span>01</span><strong>Record</strong><p>Keep the strings, both hands, and audio clear.</p></li>
            <li><span>02</span><strong>Analyze</strong><p>Run audio, hand tracking, or both together.</p></li>
            <li><span>03</span><strong>Review</strong><p>Inspect the timeline before exporting or reporting results.</p></li>
          </ol>
          <Link to="/tool" className="vintage-btn vintage-btn-primary">Open the analysis studio</Link>
        </section>

        <section id="contact" className="vintage-contact">
          <h2 className="vintage-section-title">Contact</h2>
          <p className="vintage-section-subtitle">NAT SHIN NAUNG — a research and teaching prototype for Myanmar harp performance analysis.</p>
        </section>
      </main>

      <footer className="vintage-footer">
        <p>NAT SHIN NAUNG · Myanmar harp research · Audio · Hand · Review</p>
      </footer>
    </div>
  );
}
