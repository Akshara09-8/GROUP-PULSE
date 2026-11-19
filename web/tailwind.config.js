/** Colours are CSS variables set per colour scheme (src/theme.js), so one class works in every scheme. */
const v = (name) => `rgb(var(--${name}) / <alpha-value>)`

export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        bg: v('bg'), surface: v('surface'), ink: v('ink'), muted: v('muted'), line: v('line'),
        accent: v('accent'), 'accent-ink': v('accent-ink'), soft: v('soft'), good: v('good'), warn: v('warn'),
        p1: v('p1'), p2: v('p2'), p3: v('p3'), p4: v('p4'), p5: v('p5'), p6: v('p6'),
      },
      fontFamily: {
        display: ['"Space Grotesk"', '"Segoe UI"', 'system-ui', 'sans-serif'],
        sans: ['Inter', '"Segoe UI"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
}
