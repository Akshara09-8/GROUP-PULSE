import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev: the API runs on :8000 (python -m grouppulse.api); /api/* is proxied to it.
// Production: set VITE_API_URL to the deployed backend (e.g. a Hugging Face Space).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, '') } },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.js'],
    globals: true,
  },
})
