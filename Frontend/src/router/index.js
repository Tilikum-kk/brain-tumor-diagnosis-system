/**
 * 路由配置
 *
 * 参考来源：Art Design Pro (https://github.com/Daymychen/art-design-pro)
 * MIT License - 参考了其路由守卫鉴权、动态面包屑及NProgress进度条等设计模式。
 *
 * 登录后直接进入"影像分析"（核心业务页面）
 */
import { createRouter, createWebHistory } from 'vue-router'
import NProgress from 'nprogress'

const routes = [
  {
    path: '/',
    redirect: '/workspace',
  },
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/Login.vue'),
    meta: { title: '登录', public: true },
  },
  {
    path: '/workspace',
    name: 'Workspace',
    component: () => import('@/views/Workspace.vue'),
    meta: { title: '影像分析', icon: 'Picture' },
  },
  {
    path: '/result/:id',
    name: 'Result',
    component: () => import('@/views/Result.vue'),
    meta: { title: '分析结果', icon: 'Document' },
  },
  {
    path: '/patients',
    name: 'Patients',
    component: () => import('@/views/Patients.vue'),
    meta: { title: '患者管理', icon: 'User' },
  },
  {
    path: '/history',
    name: 'History',
    component: () => import('@/views/History.vue'),
    meta: { title: '历史记录', icon: 'Clock' },
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    component: () => import('@/views/NotFound.vue'),
    meta: { title: '404', public: true },
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

router.beforeEach((to, from, next) => {
  NProgress.start()
  const token = localStorage.getItem('access_token')
  if (!to.meta.public && !token) {
    next({ name: 'Login', query: { redirect: to.fullPath } })
  } else if (to.name === 'Login' && token) {
    next({ name: 'Workspace' })
  } else {
    next()
  }
})

router.afterEach((to) => {
  NProgress.done()
  document.title = `${to.meta.title} - 脑肿瘤MRI智能辅助诊断`
})

export default router
