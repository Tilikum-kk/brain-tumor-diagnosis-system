<template>
  <!--
    参考来源：Art Design Pro (https://github.com/Daymychen/art-design-pro)
    MIT License - 参考了其登录页左右分栏布局、品牌色渐变背景、
    卡片阴影层次及表单交互细节等设计模式。
  -->
  <div class="login-page">
    <!-- 左侧品牌区 -->
    <div class="login-brand">
      <div class="brand-content">
        <div class="brand-logo">
          <span class="logo-symbol">✚</span>
        </div>
        <h1 class="brand-name">脑肿瘤 MRI 智能辅助诊断系统</h1>
        <p class="brand-desc">基于超图神经网络的多模态医学影像 AI 分析平台</p>
        <div class="brand-features">
          <div class="feature">
            <span class="feature-dot"></span>
            <span>四种 MRI 模态融合（T1 · T1CE · T2 · FLAIR）</span>
          </div>
          <div class="feature">
            <span class="feature-dot"></span>
            <span>超图注意力机制 — 自适应模态加权</span>
          </div>
          <div class="feature">
            <span class="feature-dot"></span>
            <span>肿瘤自动分割 + WHO 分级预测</span>
          </div>
          <div class="feature">
            <span class="feature-dot"></span>
            <span>一键生成结构化 PDF 诊断报告</span>
          </div>
        </div>
        <div class="brand-footer">
          <span>重庆工商大学 · 人工智能学院</span>
          <span>HG-MFNet v1.0</span>
        </div>
      </div>
    </div>

    <!-- 右侧登录区 -->
    <div class="login-form-area">
      <div class="form-card">
        <div class="form-header">
          <h2>欢迎登录</h2>
          <p>请输入您的账号信息</p>
        </div>

        <el-form
          ref="formRef"
          :model="form"
          :rules="rules"
          class="login-form"
          @submit.prevent="handleLogin"
        >
          <el-form-item prop="username">
            <el-input
              v-model="form.username"
              placeholder="请输入用户名"
              :prefix-icon="User"
              size="large"
              class="custom-input"
            />
          </el-form-item>

          <el-form-item prop="password">
            <el-input
              v-model="form.password"
              type="password"
              placeholder="请输入密码"
              :prefix-icon="Lock"
              show-password
              size="large"
              class="custom-input"
              @keyup.enter="handleLogin"
            />
          </el-form-item>

          <el-form-item>
            <el-button
              type="primary"
              size="large"
              :loading="loading"
              class="submit-btn"
              @click="handleLogin"
            >
              {{ loading ? '正在登录…' : '登 录' }}
            </el-button>
          </el-form-item>
        </el-form>

        <div class="form-footer">
          <div class="test-account">
            <el-icon><InfoFilled /></el-icon>
            <span>测试账号：admin / 密码：admin123</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { User, Lock, InfoFilled } from '@element-plus/icons-vue'
import { authApi } from '@/api'

const router = useRouter()
const route = useRoute()
const formRef = ref(null)
const loading = ref(false)
const form = reactive({ username: 'admin', password: 'admin123' })

const rules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, message: '密码长度不能少于 6 位', trigger: 'blur' },
  ],
}

const handleLogin = async () => {
  try { await formRef.value?.validate() } catch { return }
  loading.value = true
  try {
    const res = await authApi.login(form.username, form.password)
    localStorage.setItem('access_token', res.access_token)
    localStorage.setItem('user_info', JSON.stringify(res.user))
    ElMessage.success(`欢迎回来，${res.user.full_name}`)
    router.push(route.query.redirect || '/workspace')
  } catch {
    ElMessage.error('用户名或密码错误，请重试')
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-page {
  display: flex;
  min-height: 100vh;
}

/* ===== 左侧品牌区 ===== */
.login-brand {
  flex: 1;
  background: linear-gradient(160deg, #0a1628 0%, #132744 40%, #0d2137 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
}
.login-brand::before {
  content: '';
  position: absolute;
  top: -50%;
  right: -30%;
  width: 600px;
  height: 600px;
  background: radial-gradient(circle, rgba(22,119,255,0.08) 0%, transparent 70%);
  border-radius: 50%;
}
.login-brand::after {
  content: '';
  position: absolute;
  bottom: -30%;
  left: -20%;
  width: 400px;
  height: 400px;
  background: radial-gradient(circle, rgba(82,196,26,0.06) 0%, transparent 70%);
  border-radius: 50%;
}

.brand-content {
  position: relative;
  z-index: 1;
  max-width: 440px;
  padding: 60px;
}

.brand-logo {
  width: 64px;
  height: 64px;
  background: linear-gradient(135deg, #1677ff, #4096ff);
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 24px;
  box-shadow: 0 8px 24px rgba(22,119,255,0.3);
}
.logo-symbol {
  font-size: 32px;
  color: #fff;
  font-weight: 300;
}

.brand-name {
  font-size: 26px;
  font-weight: 700;
  color: #ffffff;
  margin-bottom: 10px;
  line-height: 1.4;
}
.brand-desc {
  font-size: 14px;
  color: rgba(255,255,255,0.55);
  line-height: 1.6;
  margin-bottom: 36px;
}

.brand-features {
  display: flex;
  flex-direction: column;
  gap: 14px;
  margin-bottom: 48px;
}
.feature {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 13px;
  color: rgba(255,255,255,0.7);
}
.feature-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #1677ff;
  flex-shrink: 0;
}

.brand-footer {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
  color: rgba(255,255,255,0.3);
}

/* ===== 右侧登录区 ===== */
.login-form-area {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f5f7fa;
}

.form-card {
  width: 400px;
  background: #fff;
  border-radius: 16px;
  padding: 48px 44px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.06);
}

.form-header {
  text-align: center;
  margin-bottom: 32px;
}
.form-header h2 {
  font-size: 24px;
  font-weight: 700;
  color: #1a1a1a;
  margin-bottom: 6px;
}
.form-header p {
  font-size: 14px;
  color: #999;
}

.custom-input :deep(.el-input__wrapper) {
  border-radius: 8px;
  box-shadow: 0 0 0 1px #e0e0e0 inset;
}
.custom-input :deep(.el-input__wrapper:hover) {
  box-shadow: 0 0 0 1px #1677ff inset;
}

.submit-btn {
  width: 100%;
  height: 46px;
  font-size: 16px;
  border-radius: 8px;
  margin-top: 8px;
}

.form-footer {
  text-align: center;
  margin-top: 20px;
}
.test-account {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  background: #f0f5ff;
  border-radius: 6px;
  font-size: 12px;
  color: #1677ff;
}

@media (max-width: 768px) {
  .login-brand { display: none; }
  .form-card { width: 90%; padding: 32px 24px; }
}
</style>
