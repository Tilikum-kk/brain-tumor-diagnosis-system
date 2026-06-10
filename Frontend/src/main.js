/*
 * ============================================================================
 * 参考来源：Art Design Pro (https://github.com/Daymychen/art-design-pro)
 * 基于 MIT 开源协议，参考了其 CSS 变量体系、Element Plus 组件样式覆写、
 * 侧边栏交互动效、面包屑导航及响应式布局等设计模式。
 * ============================================================================
 */
import { createApp } from 'vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import NProgress from 'nprogress'
import 'nprogress/nprogress.css'

import App from './App.vue'
import router from './router'

// 创建 Vue 应用
const app = createApp(App)

// 注册 Element Plus
app.use(ElementPlus)

// 注册所有图标
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}

// 注册路由
app.use(router)

// 进度条
NProgress.configure({ showSpinner: false })

app.mount('#app')
