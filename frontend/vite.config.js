import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In dev, proxy API calls to the FastAPI server so there is no CORS to configure.
// In production the built `dist/` is served by FastAPI itself (same origin).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/predict': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
