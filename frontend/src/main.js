// main.js —— 前端入口：加载页面组件和全局样式，再挂载到 index.html 的 #app。
// KaTeX 的字体样式必须随页面一同打包，公式不能只加载渲染脚本。
import { createApp } from 'vue'
import App from './App.vue'
import 'katex/dist/katex.min.css'
import './style.css'

createApp(App).mount('#app')
