import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Proxy API requests to backend in development
      '/api': 'http://localhost:8000'
    }
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets'
  }
})
