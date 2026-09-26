// vite.config.js —— 本地开发服务器配置。
// 浏览器请求同源 /api，Vite 再转发到 8000；正式部署时需由部署环境提供同等路由。
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
