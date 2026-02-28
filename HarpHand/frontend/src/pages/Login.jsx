import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useGoogleLogin } from '@react-oauth/google';
import './Login.css';
import myanmarHarpImage from '../assets/myanmar_harp.jpg';

export default function Login() {
    const navigate = useNavigate();

    const [authError, setAuthError] = useState(null);
    const [isAuthenticating, setIsAuthenticating] = useState(false);

    const handleLogin = (e) => {
        e.preventDefault();
        // In a real app, you'd authenticate here. For now, we mock login and navigate to the tool.
        navigate('/tool');
    };

    const loginWithGoogle = useGoogleLogin({
        flow: 'auth-code',
        onSuccess: async (codeResponse) => {
            try {
                setIsAuthenticating(true);
                setAuthError(null);
                // Send the auth code to our FastAPI backend 
                // The backend will exchange it for tokens and verify
                const API = import.meta.env.VITE_API_URL || '/api';
                const res = await fetch(`${API}/auth/google`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ token: codeResponse.code }), // Send code as token
                });

                if (!res.ok) throw new Error('Failed to authenticate with backend');

                // On success, go to tool
                navigate('/tool');
            } catch (err) {
                setAuthError('Google sign-in failed. Please try again.');
                console.error(err);
            } finally {
                setIsAuthenticating(false);
            }
        },
        onError: () => {
            setAuthError('Google sign-in was canceled or failed.');
        }
    });

    return (
        <div className="login-container">
            {/* Left split: Image */}
            <div className="login-image-side">
                <img
                    src={myanmarHarpImage}
                    alt="Myanmar Harp"
                    className="login-background-image"
                />
                <div className="login-image-overlay">
                    <div className="login-logo">
                        <span className="logo-icon">Λ</span>MU
                    </div>
                    <div className="login-image-text">
                        <h2>Capturing Moments,</h2>
                        <h2>Creating Memories</h2>
                        <div className="carousel-indicators">
                            <span className="indicator active"></span>
                            <span className="indicator"></span>
                            <span className="indicator"></span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Right split: Form */}
            <div className="login-form-side">
                <div className="login-form-wrapper">
                    <h1 className="login-title">Create an account</h1>
                    <p className="login-subtitle">
                        Already have an account? <a href="#" className="login-link">Log in</a>
                    </p>

                    <form onSubmit={handleLogin} className="login-form">
                        <div className="input-group-row">
                            <input type="text" placeholder="First name" className="login-input" required />
                            <input type="text" placeholder="Last name" className="login-input" required />
                        </div>

                        <input type="email" placeholder="Email" className="login-input full-width" required />

                        <div className="password-wrapper">
                            <input type="password" placeholder="Enter your password" className="login-input full-width" required />
                            <button type="button" className="password-toggle">
                                {/* SVG Eye Icon */}
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M12 5C7.30558 5 3.32839 8.00695 1.70508 12C3.32839 15.993 7.30558 19 12 19C16.6944 19 20.6716 15.993 22.2949 12C20.6716 8.00695 16.6944 5 12 5ZM12 17C9.23858 17 7 14.7614 7 12C7 9.23858 9.23858 7 12 7C14.7614 7 17 9.23858 17 12C17 14.7614 14.7614 17 12 17ZM12 9C10.3431 9 9 10.3431 9 12C9 13.6569 10.3431 15 12 15C13.6569 15 15 13.6569 15 12C15 10.3431 13.6569 9 12 9Z" fill="currentColor" />
                                </svg>
                            </button>
                        </div>

                        <label className="terms-checkbox">
                            <input type="checkbox" required />
                            <span>I agree to the <a href="#" className="login-link">Terms & Conditions</a></span>
                        </label>

                        <button type="submit" className="login-submit-btn">
                            Create account
                        </button>
                    </form>

                    {authError && <p className="error" style={{ color: '#f87171', marginBottom: '1rem', fontSize: '0.9rem' }}>{authError}</p>}

                    <div className="login-divider">
                        <span>Or register with</span>
                    </div>

                    <div className="social-login-group">
                        <button
                            type="button"
                            className="social-btn"
                            onClick={() => loginWithGoogle()}
                            disabled={isAuthenticating}
                        >
                            <img src="https://upload.wikimedia.org/wikipedia/commons/5/53/Google_%22G%22_Logo.svg" alt="Google" className="social-icon" />
                            {isAuthenticating ? 'Signing in...' : 'Google'}
                        </button>
                        <button type="button" className="social-btn">
                            <svg className="social-icon" width="20" height="20" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg"><path d="M17.05 20.28c-.98.95-2.05.8-3.08.35-1.09-.46-2.09-.48-3.24 0-1.44.62-2.2.44-3.06-.35C2.79 15.25 3.51 7.59 9.05 7.31c1.35.07 2.29.74 3.08.8 1.18-.19 2.24-1.01 3.8-1.04 1.34.04 2.58.55 3.44 1.48-2.92 1.76-2.45 5.56.54 6.78-1.01 2.53-2.19 4.31-2.86 5.05zm-3.26-15.01c.21-1.37-.36-2.73-1.18-3.71-1.12-1.18-2.65-1.84-4.12-1.63-.23 1.42.44 2.84 1.31 3.82 1.11 1.25 2.71 1.83 4.19 1.62-.06-.03-.13-.06-.2-.1z" /></svg>
                            Apple
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
