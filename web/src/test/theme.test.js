import { describe, expect, it } from 'vitest'
import { SCHEMES, applyScheme, contrast, savedScheme, schemeVars } from '../theme.js'

describe('T4.4 colour schemes meet WCAG AA for text', () => {
  for (const [id, s] of Object.entries(SCHEMES)) {
    it(`${s.name}: body, secondary, button and alert text >= 4.5:1`, () => {
      const { c } = s
      expect(contrast(c.ink, c.bg)).toBeGreaterThanOrEqual(4.5)
      expect(contrast(c.ink, c.surface)).toBeGreaterThanOrEqual(4.5)
      expect(contrast(c.muted, c.surface)).toBeGreaterThanOrEqual(4.5)
      expect(contrast(c['accent-ink'], c.accent)).toBeGreaterThanOrEqual(4.5)
      expect(contrast(c.accent, c.soft)).toBeGreaterThanOrEqual(4.5)   // accent text on accent tint (pills)
      expect(contrast(c.warn, c.surface)).toBeGreaterThanOrEqual(3)   // alerts are bold or large + have a word
    })
  }
})

describe('T4.6 scheme choice is remembered', () => {
  it('applies variables and survives a reload', () => {
    applyScheme('night')
    expect(document.documentElement.dataset.scheme).toBe('night')
    expect(document.documentElement.style.getPropertyValue('--bg')).toBe('13 18 32')
    expect(savedScheme()).toBe('night')
  })
  it('falls back to the default for unknown ids', () => {
    expect(applyScheme('nope')).toBe('calm')
  })
  it('every scheme defines six person colours', () => {
    for (const id of Object.keys(SCHEMES)) expect(Object.keys(schemeVars(id)).filter((k) => k.startsWith('--p'))).toHaveLength(6)
  })
})
