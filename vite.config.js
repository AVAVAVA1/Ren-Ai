import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    vueDevTools(),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    },
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        // 立绘/背景可能数十分钟；Node 默认 socket 约 2min、代理层也可能断连，易触发后端 CancelledError
        timeout: 3_600_000,
        proxyTimeout: 3_600_000,
        configure(proxy) {
          proxy.on('proxyReq', (_proxyReq, req) => {
            req.setTimeout(0)
            req.socket?.setTimeout?.(0)
          })
        },
      },
    },
  },
})
