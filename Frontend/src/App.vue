<template>
  <!--
    参考来源：Art Design Pro (https://github.com/Daymychen/art-design-pro)
    MIT License - 参考了其 CSS 变量体系、Element Plus 样式覆写、
    侧边栏交互、面包屑导航及响应式布局等设计模式。
  -->
  <div id="app-container" :class="{ 'dark-mode': isDark }">
    <!-- 公开页面（登录）不使用布局 -->
    <template v-if="$route.meta.public">
      <router-view v-slot="{ Component }">
        <transition name="fade" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </template>

    <!-- 主界面 -->
    <el-container v-else class="main-layout">
      <!-- ====== 侧边栏 ====== -->
      <el-aside :width="collapsed ? '64px' : '220px'" class="sidebar">
        <div class="sidebar-logo" @click="$router.push('/workspace')">
          <div class="logo-icon">
            <svg width="28" height="28" viewBox="0 0 28 28">
              <rect width="28" height="28" rx="6" fill="url(#logo-grad)"/>
              <text x="14" y="20" text-anchor="middle" fill="#fff" font-size="16" font-weight="700">+</text>
              <defs>
                <linearGradient id="logo-grad" x1="0" y1="0" x2="28" y2="28">
                  <stop offset="0%" stop-color="#1677ff"/>
                  <stop offset="100%" stop-color="#69b1ff"/>
                </linearGradient>
              </defs>
            </svg>
          </div>
          <transition name="fade">
            <span v-show="!collapsed" class="logo-text">脑肿瘤AI诊断</span>
          </transition>
        </div>

        <el-menu
          :default-active="route.path"
          :collapse="collapsed"
          background-color="var(--sidebar-bg)"
          text-color="var(--menu-text)"
          active-text-color="#ffffff"
          router
          class="side-menu"
        >
          <el-menu-item index="/workspace">
            <template #title>
              <el-icon><PictureFilled /></el-icon>
              <span>影像分析</span>
            </template>
          </el-menu-item>
          <el-menu-item index="/patients">
            <template #title>
              <el-icon><UserFilled /></el-icon>
              <span>患者管理</span>
            </template>
          </el-menu-item>
          <el-menu-item index="/history">
            <template #title>
              <el-icon><Clock /></el-icon>
              <span>历史记录</span>
            </template>
          </el-menu-item>
        </el-menu>

        <div class="sidebar-bottom">
          <div class="collapse-btn" @click="collapsed = !collapsed">
            <el-icon :size="16"><Fold v-if="!collapsed" /><Expand v-else /></el-icon>
          </div>
        </div>
      </el-aside>

      <!-- ====== 右侧主区域 ====== -->
      <el-container class="right-area">
        <!-- 顶栏 + 面包屑 -->
        <el-header class="topbar">
          <div class="topbar-left">
            <el-breadcrumb separator="›">
              <el-breadcrumb-item :to="{ path: '/workspace' }">
                <el-icon><HomeFilled /></el-icon> 首页
              </el-breadcrumb-item>
              <el-breadcrumb-item v-if="route.meta.title">
                {{ route.meta.title }}
              </el-breadcrumb-item>
            </el-breadcrumb>
          </div>

          <div class="topbar-right">
            <!-- 系统状态 -->
            <span class="status-indicator" :class="online ? 'online' : 'offline'">
              <span class="status-dot"></span>
              {{ online ? '运行中' : '离线' }}
            </span>

            <!-- 主题切换 -->
            <el-tooltip :content="isDark ? '切换亮色主题' : '切换暗色主题'" placement="bottom">
              <el-button text circle @click="isDark = !isDark">
                <el-icon :size="18"><Sunny v-if="isDark" /><Moon v-else /></el-icon>
              </el-button>
            </el-tooltip>

            <el-divider direction="vertical" />

            <!-- 用户菜单 -->
            <el-dropdown trigger="click">
              <span class="user-info">
                <el-avatar :size="28" :style="{ background: '#1677ff', fontSize: '13px' }">
                  {{ (userInfo?.full_name || '管')[0] }}
                </el-avatar>
                <span class="user-name">{{ userInfo?.full_name || '管理员' }}</span>
                <el-icon :size="12"><ArrowDown /></el-icon>
              </span>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item disabled>
                    <div style="line-height:1.5">
                      <div style="font-weight:600">{{ userInfo?.full_name }}</div>
                      <div style="font-size:12px;color:#999">{{ userInfo?.hospital }} · {{ userInfo?.department }}</div>
                    </div>
                  </el-dropdown-item>
                  <el-dropdown-item divided @click="handleLogout">
                    <el-icon><SwitchButton /></el-icon> 退出登录
                  </el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
          </div>
        </el-header>

        <!-- 页面内容 -->
        <el-main class="main-content">
          <router-view v-slot="{ Component }">
            <transition name="slide-fade" mode="out-in">
              <component :is="Component" />
            </transition>
          </router-view>
        </el-main>
      </el-container>
    </el-container>
  </div>
</template>

<script setup>
/**
 * 根布局组件
 *
 * 参考来源：Art Design Pro (https://github.com/Daymychen/art-design-pro)
 * MIT License
 *
 * 参考的设计模式：
 *   1. CSS 变量驱动的主题体系（variables.scss → CSS自定义属性）
 *   2. 侧边栏折叠动画与过渡效果
 *   3. 面包屑导航自动生成
 *   4. 暗色/亮色双主题切换
 *   5. 页面切换过渡动画（slide-fade）
 *   6. Element Plus 组件样式覆写
 */
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessageBox } from 'element-plus'
import {
  Fold, Expand, ArrowDown, PictureFilled, UserFilled, Clock,
  HomeFilled, Sunny, Moon, SwitchButton,
} from '@element-plus/icons-vue'
import { authApi } from '@/api'

const router = useRouter()
const route = useRoute()

// 侧边栏折叠
const collapsed = ref(false)

// 主题切换
const isDark = ref(false)

// 用户信息
const userInfo = ref(null)
onMounted(async () => {
  try { userInfo.value = await authApi.getMe() } catch { userInfo.value = null }
})

// 系统状态（仅页面加载时检查一次，不轮询）
const online = ref(true)
onMounted(async () => {
  try {
    const { systemApi } = await import('@/api')
    await systemApi.healthCheck()
    online.value = true
  } catch { online.value = false }
})

// 退出登录
const handleLogout = async () => {
  try {
    await ElMessageBox.confirm('确定要退出登录吗？', '提示', {
      type: 'warning', confirmButtonText: '确定', cancelButtonText: '取消',
    })
  } catch { return }
  localStorage.clear()
  router.push('/login')
}
</script>

<style>
/*
 * CSS 变量体系 - 参考 Art Design Pro 的设计规范
 * https://github.com/Daymychen/art-design-pro (MIT License)
 */
:root {
  /* 主色系 */
  --color-primary: #1677ff;
  --color-primary-light: #4096ff;
  --color-primary-bg: #e6f4ff;

  /* 功能色 */
  --color-success: #52c41a;
  --color-warning: #faad14;
  --color-danger: #ff4d4f;
  --color-info: #1677ff;

  /* 背景色 */
  --sidebar-bg: #001529;
  --topbar-bg: #ffffff;
  --main-bg: #f0f2f5;
  --card-bg: #ffffff;

  /* 文字色 */
  --text-primary: #1a1a1a;
  --text-secondary: #666666;
  --text-tertiary: #999999;
  --text-disabled: #cccccc;
  --menu-text: rgba(255, 255, 255, 0.65);

  /* 边框 */
  --border-color: #f0f0f0;
  --border-light: #f5f5f5;

  /* 阴影 */
  --shadow-card: 0 1px 3px rgba(0, 0, 0, 0.06);
  --shadow-dropdown: 0 6px 16px rgba(0, 0, 0, 0.08);

  /* 圆角 */
  --radius-sm: 6px;
  --radius-md: 8px;
  --radius-lg: 12px;

  /* 间距 */
  --spacing-xs: 4px;
  --spacing-sm: 8px;
  --spacing-md: 16px;
  --spacing-lg: 24px;

  /* 过渡 */
  --transition-fast: 0.15s ease;
  --transition-normal: 0.25s ease;
}

/* 暗色模式 */
.dark-mode {
  --sidebar-bg: #141414;
  --topbar-bg: #1d1d1d;
  --main-bg: #0a0a0a;
  --card-bg: #1d1d1d;
  --text-primary: #e8e8e8;
  --text-secondary: #a0a0a0;
  --text-tertiary: #707070;
  --border-color: #2a2a2a;
  --border-light: #222222;
  --shadow-card: 0 1px 3px rgba(0,0,0,0.3);
}

* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'PingFang SC', 'Microsoft YaHei', 'Helvetica Neue', sans-serif;
  color: var(--text-primary);
  background: var(--main-bg);
  -webkit-font-smoothing: antialiased;
}

/* 主布局 */
.main-layout { height: 100vh; }

/* ====== 侧边栏 ====== */
.sidebar {
  background: var(--sidebar-bg);
  display: flex;
  flex-direction: column;
  transition: width var(--transition-normal);
  overflow: hidden;
}
.sidebar-logo {
  height: 56px;
  display: flex;
  align-items: center;
  padding: 0 18px;
  cursor: pointer;
  gap: 10px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
}
.logo-text {
  color: #fff;
  font-size: 15px;
  font-weight: 600;
  white-space: nowrap;
}

.side-menu {
  flex: 1;
  border-right: none !important;
  padding: 8px;
}
.side-menu .el-menu-item {
  height: 42px;
  line-height: 42px;
  margin-bottom: 2px;
  border-radius: var(--radius-sm);
  font-size: 14px;
  transition: all var(--transition-fast);
}
.side-menu .el-menu-item.is-active {
  background: var(--color-primary) !important;
}
.side-menu .el-menu-item:not(.is-active):hover {
  background: rgba(255,255,255,0.06) !important;
}

.sidebar-bottom {
  padding: 10px;
  border-top: 1px solid rgba(255,255,255,0.08);
}
.collapse-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 36px;
  color: rgba(255,255,255,0.45);
  cursor: pointer;
  border-radius: var(--radius-sm);
  transition: all var(--transition-fast);
}
.collapse-btn:hover {
  color: #fff;
  background: rgba(255,255,255,0.08);
}

/* ====== 顶栏 ====== */
.right-area { flex-direction: column; }
.topbar {
  height: 48px;
  background: var(--topbar-bg);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.04);
  z-index: 10;
}
.topbar-left { display: flex; align-items: center; }
.topbar-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  padding: 4px 10px;
  border-radius: 12px;
  background: var(--border-light);
}
.status-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
}
.status-indicator.online .status-dot { background: var(--color-success); box-shadow: 0 0 4px var(--color-success); }
.status-indicator.offline .status-dot { background: var(--color-danger); }

.user-info {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  padding: 2px 4px;
  border-radius: var(--radius-sm);
  transition: background var(--transition-fast);
}
.user-info:hover { background: var(--border-light); }
.user-name { font-size: 13px; color: var(--text-secondary); }

/* ====== 内容区 ====== */
.main-content {
  background: var(--main-bg);
  padding: 20px;
  overflow-y: auto;
  min-height: 0;
}

/* ====== Element Plus 样式覆写 ====== */
/* 卡片 */
.el-card {
  border-radius: var(--radius-md) !important;
  border: none !important;
  box-shadow: var(--shadow-card) !important;
}

/* 表格 */
.el-table {
  --el-table-border-color: var(--border-light);
}
.el-table th.el-table__cell {
  background: #fafafa;
  font-weight: 600;
  color: var(--text-primary);
}

/* 按钮 */
.el-button--primary {
  --el-button-bg-color: var(--color-primary);
  --el-button-border-color: var(--color-primary);
}
.el-button--primary:hover {
  --el-button-bg-color: var(--color-primary-light);
  --el-button-border-color: var(--color-primary-light);
}

/* 输入框 */
.el-input .el-input__wrapper {
  border-radius: var(--radius-sm);
  box-shadow: 0 0 0 1px #e0e0e0 inset;
  transition: box-shadow var(--transition-fast);
}
.el-input .el-input__wrapper:hover {
  box-shadow: 0 0 0 1px var(--color-primary-light) inset;
}
.el-input .el-input__wrapper.is-focus {
  box-shadow: 0 0 0 1px var(--color-primary) inset, 0 0 0 2px rgba(22,119,255,0.1);
}

/* ====== 页面过渡动画 ====== */
.fade-enter-active, .fade-leave-active { transition: opacity 0.2s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }

.slide-fade-enter-active { transition: all 0.25s ease-out; }
.slide-fade-leave-active { transition: all 0.2s ease-in; }
.slide-fade-enter-from { opacity: 0; transform: translateX(16px); }
.slide-fade-leave-to { opacity: 0; transform: translateX(-16px); }

/* 响应式 */
@media (max-width: 768px) {
  .sidebar { width: 64px !important; }
  .logo-text { display: none; }
}
</style>
