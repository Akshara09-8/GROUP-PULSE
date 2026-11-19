import React from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.jsx'
import { applyScheme, savedScheme } from './theme.js'
import './index.css'

// ?scheme=night (etc.) picks a colour scheme from the link; otherwise the viewer's last choice
applyScheme(new URLSearchParams(location.search).get('scheme') || savedScheme())

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
)
