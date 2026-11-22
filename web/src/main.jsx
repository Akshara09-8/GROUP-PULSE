import React from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, HashRouter } from 'react-router-dom'
import App from './App.jsx'
import { DEMO } from './api.js'
import { applyScheme, savedScheme } from './theme.js'
import './index.css'

// ?scheme=night (etc.) picks a colour scheme from the link; otherwise the viewer's last choice
applyScheme(new URLSearchParams(location.search).get('scheme') || savedScheme())

// Static hosts (GitHub Pages) cannot rewrite deep links, so the demo build routes on the #hash.
const Router = DEMO ? HashRouter : BrowserRouter

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <Router>
      <App />
    </Router>
  </React.StrictMode>,
)
