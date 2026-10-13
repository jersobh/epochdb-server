import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: './',
  build: {
    outDir: '../src/static/ui',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/entity_graph': 'http://localhost:8080',
      '/remember': 'http://localhost:8080',
      '/query': 'http://localhost:8080',
      '/export': 'http://localhost:8080',
      '/import': 'http://localhost:8080',
      '/skills': 'http://localhost:8080',
      '/profiles': 'http://localhost:8080',
      '/stats': 'http://localhost:8080',
      '/stream': 'http://localhost:8080',
      '/compact': 'http://localhost:8080',
      '/delete': 'http://localhost:8080',
      '/get': 'http://localhost:8080',
      '/history': 'http://localhost:8080',
      '/cluster/metrics': 'http://localhost:8080',
      '/healthz': 'http://localhost:8080',
    },
  },
})
