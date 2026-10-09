import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useGoogleLogin } from '@react-oauth/google'
import './Login.css'
import harp1 from '../assets/myanmar_harp.jpg'
import harp2 from '../assets/harp_2.jpg'
import harp3 from '../assets/harp_3.jpg'
import BrandLogo from '../components/BrandLogo.jsx'
import ThemeToggle from '../components/ThemeToggle.jsx'

const CAROUSEL_IMAGES = [harp1, harp2, harp3]
const CAROUSEL_INTERVAL = 5600

export default function Login() {
  const navigate = useNavigate()
  const [authError, setAuthError] = useState(null)
  const [isAuthenticating, setIsAuthenticating] = useState(false)
  const [currentSlide, setCurrentSlide] = useState(0)
  const [firstName, setFirstName] = useState('')
  const [lastName, setLastName] = useState('')
  const googleConfigured = Boolean(import.meta.env.VITE_GOOGLE_CLIENT_ID)

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return undefined
    const timer = window.setInterval(() => {
      setCurrentSlide((previous) => (previous + 1) % CAROUSEL_IMAGES.length)
    }, CAROUSEL_INTERVAL)
    return () => window.clearInterval(timer)
  }, [])

  const handleProfile = (event) => {
    event.preventDefault()
    const displayName = `${firstName} ${lastName}`.trim()
    localStorage.setItem('user_name', displayName)
    localStorage.setItem('user_avatar', '')
    localStorage.setItem('user_email', '')
    navigate('/tool')
  }

  const loginWithGoogle = useGoogleLogin({
    flow: 'auth-code',
    onSuccess: async (codeResponse) => {
      try {
        setIsAuthenticating(true)
        setAuthError(null)
        const API = import.meta.env.VITE_API_URL || '/api'
        const response = await fetch(`${API}/auth/google`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ token: codeResponse.code }),
        })

        if (!response.ok) throw new Error('Failed to authenticate with the analysis service')
        const data = await response.json()
        const name = data.name || data.user?.name || ''
        const picture = data.picture || data.user?.picture || ''
        const email = data.email || data.user?.email || ''
        if (name) localStorage.setItem('user_name', name)
        if (picture) localStorage.setItem('user_avatar', picture)
        if (email) localStorage.setItem('user_email', email)
        navigate('/tool')
      } catch (error) {
        console.error(error)
        setAuthError('Google sign-in could not be completed. You can still continue with a local display name.')
      } finally {
        setIsAuthenticating(false)
      }
    },
    onError: () => setAuthError('Google sign-in was canceled or could not be completed.'),
  })

  return (
    <main className="profile-page">
      <a className="skip-link" href="#profile-form">Skip to profile form</a>
      <ThemeToggle className="profile-theme-toggle" />
      <section className="profile-gallery" aria-label="Myanmar harp gallery">
        {CAROUSEL_IMAGES.map((source, index) => (
          <img
            key={source}
            src={source}
            alt={index === 0 ? 'Traditional Myanmar saung harp' : ''}
            className={index === currentSlide ? 'profile-image is-active' : 'profile-image'}
          />
        ))}
        <div className="profile-gallery-shade" />
        <Link to="/" className="profile-brand">
          <BrandLogo />
          <div><strong>Nat Shin Naung</strong><small>Saung intelligence project</small></div>
        </Link>
        <div className="profile-gallery-copy">
          <p>Research studio · Myanmar harp</p>
          <h1>A quieter way to<br /><em>read performance.</em></h1>
          <div className="profile-gallery-meta"><span>Audio</span><span>Hand</span><span>Evidence</span></div>
        </div>
        <div className="profile-carousel" aria-label="Gallery slides">
          {CAROUSEL_IMAGES.map((_, index) => (
            <button
              key={index}
              type="button"
              className={index === currentSlide ? 'is-active' : ''}
              onClick={() => setCurrentSlide(index)}
              aria-label={`Show harp image ${index + 1}`}
              aria-current={index === currentSlide ? 'true' : undefined}
            />
          ))}
        </div>
      </section>

      <section id="profile-form" className="profile-form-side">
        <div className="profile-form-wrap">
          <p className="profile-kicker">Optional studio profile</p>
          <h2>What should the workspace call you?</h2>
          <p className="profile-intro">This display name stays in your browser. It is not an account and is not sent with an analysis.</p>

          <form onSubmit={handleProfile} className="profile-form">
            <div className="profile-name-grid">
              <label>
                <span>First name</span>
                <input value={firstName} onChange={(event) => setFirstName(event.target.value)} autoComplete="given-name" required />
              </label>
              <label>
                <span>Last name</span>
                <input value={lastName} onChange={(event) => setLastName(event.target.value)} autoComplete="family-name" required />
              </label>
            </div>
            <button type="submit" className="profile-submit">Save name and enter studio <span aria-hidden="true">↗</span></button>
          </form>

          <div className="profile-divider"><span>or</span></div>

          <button
            type="button"
            className="profile-google"
            onClick={() => loginWithGoogle()}
            disabled={isAuthenticating || !googleConfigured}
            title={googleConfigured ? 'Continue with Google' : 'Google sign-in is not configured'}
          >
            <span className="google-mark" aria-hidden="true">G</span>
            {isAuthenticating ? 'Connecting…' : googleConfigured ? 'Continue with Google' : 'Google sign-in unavailable'}
          </button>

          {authError && <p className="profile-error" role="alert">{authError}</p>}
          <Link to="/tool" className="profile-skip">Continue without a profile <span aria-hidden="true">→</span></Link>
          <p className="profile-privacy">Uploaded performances are processed for your analysis and are not used to retrain the model.</p>
        </div>
      </section>
    </main>
  )
}
