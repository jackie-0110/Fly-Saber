import { fileURLToPath, URL } from 'node:url'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The dev server proxies the backend so the browser stays same-origin (the tunnel sends
// no CORS headers). BACKEND_PROXY points it elsewhere, e.g.
// BACKEND_PROXY=https://api.jackie-courtney.com npm run dev
const backend = (process.env.BACKEND_PROXY ?? 'http://localhost:8000').replace(/\/$/, '')

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  server: {
    proxy: {
      '/api': { target: backend, changeOrigin: true },
      '/ws': { target: backend.replace(/^http/, 'ws'), ws: true, changeOrigin: true },
    },
  },
})
