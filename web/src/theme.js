// The four colour schemes from design/group-pulse-ui-schemes.html, as CSS variables.
// Each person keeps one colour on every chart in a scheme (p1..p6).

export const SCHEMES = {
  calm: {
    name: 'Clinic Calm', mood: 'Light, quiet teal',
    c: { bg: '#F3F7F6', surface: '#FFFFFF', ink: '#11302E', muted: '#557069', line: '#D6E2E0', accent: '#0F766E',
      'accent-ink': '#FFFFFF', soft: '#DDF0EC', good: '#15803D', warn: '#C2410C' },
    people: ['#0F766E', '#2563EB', '#B45309', '#9333EA', '#BE185D', '#4D7C0F'],
  },
  night: {
    name: 'Night Lab', mood: 'Dark console',
    c: { bg: '#0D1220', surface: '#151C2D', ink: '#E7EBF3', muted: '#929CB2', line: '#273149', accent: '#7EA2FF',
      'accent-ink': '#0D1220', soft: '#1E2944', good: '#34D399', warn: '#FBBF24' },
    people: ['#7EA2FF', '#34D399', '#F472B6', '#C4A1FF', '#FDBA74', '#5EEAD4'],
  },
  coral: {
    name: 'Sunrise Coral', mood: 'Warm, friendly',
    c: { bg: '#FFF7F2', surface: '#FFFFFF', ink: '#2A1C19', muted: '#76605A', line: '#F0DDD3', accent: '#B83147',
      'accent-ink': '#FFFFFF', soft: '#FBE3E2', good: '#2F7D6D', warn: '#A85B06' },
    people: ['#B83147', '#2F7D6D', '#3B6FD4', '#C98A1B', '#7C3AED', '#0E7490'],
  },
  indigo: {
    name: 'Indigo Contrast', mood: 'Crisp, accessible',
    c: { bg: '#F5F5FB', surface: '#FFFFFF', ink: '#18173A', muted: '#52517A', line: '#DCDCEE', accent: '#4338CA',
      'accent-ink': '#FFFFFF', soft: '#E6E5FA', good: '#047857', warn: '#BE123C' },
    people: ['#4338CA', '#DB2777', '#0891B2', '#CA8A04', '#059669', '#9333EA'],
  },
}

export const DEFAULT_SCHEME = 'calm'
const KEY = 'gp-scheme'

const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)).join(' ')

export function schemeVars(id) {
  const s = SCHEMES[id] || SCHEMES[DEFAULT_SCHEME]
  const vars = Object.fromEntries(Object.entries(s.c).map(([k, v]) => [`--${k}`, rgb(v)]))
  s.people.forEach((v, i) => { vars[`--p${i + 1}`] = rgb(v) })
  return vars
}

export function applyScheme(id, root = document.documentElement) {
  const scheme = SCHEMES[id] ? id : DEFAULT_SCHEME
  for (const [k, v] of Object.entries(schemeVars(scheme))) root.style.setProperty(k, v)
  root.dataset.scheme = scheme
  root.style.colorScheme = scheme === 'night' ? 'dark' : 'light'
  try { localStorage.setItem(KEY, scheme) } catch { /* private window: theme still applies */ }
  return scheme
}

export function savedScheme() {
  try { return localStorage.getItem(KEY) || DEFAULT_SCHEME } catch { return DEFAULT_SCHEME }
}

// WCAG 2.1 contrast ratio (test T4.4 checks text pairs reach 4.5:1)
function lum(hex) {
  const v = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
    .map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4))
  return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]
}
export function contrast(a, b) {
  const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p)
  return (x + 0.05) / (y + 0.05)
}

export const personColor = (label) => `rgb(var(--p${Math.min(6, parseInt(String(label).slice(1), 10) || 1)}))`
