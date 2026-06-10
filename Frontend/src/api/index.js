/**
 * ==========================================================================
 * API通信模块 - Axios封装
 *
 * 封装所有后端API调用，统一处理：
 *   - 请求/响应拦截
 *   - JWT Token自动附加
 *   - 错误统一处理
 *   - Token过期自动刷新
 * ==========================================================================
 */

import axios from 'axios'
import { ElMessage, ElNotification } from 'element-plus'
import router from '@/router'

// 创建axios实例
const api = axios.create({
  baseURL: '/',           // 开发环境通过Vite代理
  timeout: 300000,        // 5分钟超时（大文件上传）
  headers: {
    'Content-Type': 'application/json',
  },
})

// ========================================================================
// 请求拦截器 - 自动附加JWT Token
// ========================================================================
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    // 文件上传时不设置Content-Type
    if (config.data instanceof FormData) {
      delete config.headers['Content-Type']
    }
    return config
  },
  (error) => Promise.reject(error)
)

// ========================================================================
// 响应拦截器 - 统一错误处理
// ========================================================================
api.interceptors.response.use(
  (response) => response.data,
  async (error) => {
    if (error.response) {
      const { status, data } = error.response

      switch (status) {
        case 401:
          // Token过期，清除并跳转登录
          localStorage.removeItem('access_token')
          localStorage.removeItem('user_info')
          ElMessage.error('登录已过期，请重新登录')
          router.push({ name: 'Login' })
          break
        case 403:
          ElMessage.error('权限不足：' + (data.detail || '无法执行此操作'))
          break
        case 404:
          ElMessage.error('请求的资源不存在')
          break
        case 413:
          ElMessage.error('上传文件过大，请压缩后重试')
          break
        case 422:
          ElMessage.error('请求参数验证失败')
          break
        case 500:
          // 服务器内部错误，轻提示（不弹大窗打扰用户）
          ElMessage.error('服务器繁忙，请稍后重试')
          break
        default:
          ElMessage.error(data.detail || `请求失败 (${status})`)
      }
    } else if (error.code === 'ECONNABORTED') {
      ElMessage.error('请求超时，请检查网络连接')
    } else {
      ElMessage.error('网络连接异常，请检查后端服务是否启动')
    }
    return Promise.reject(error)
  }
)

// ========================================================================
// 认证API - Authentication API
// ========================================================================
export const authApi = {
  /** 用户登录 */
  login(username, password) {
    const formData = new FormData()
    formData.append('username', username)
    formData.append('password', password)
    return api.post('/api/auth/login', formData, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    })
  },

  /** 用户注册 */
  register(userData) {
    return api.post('/api/auth/register', userData)
  },

  /** 获取当前用户信息 */
  getMe() {
    return api.get('/api/auth/me')
  },

  /** 刷新Token */
  refreshToken() {
    return api.post('/api/auth/refresh')
  },
}

// ========================================================================
// 患者管理API - Patient API
// ========================================================================
export const patientApi = {
  /** 创建患者 */
  create(patientData) {
    return api.post('/api/patients', patientData)
  },

  /** 患者列表 */
  list(params = {}) {
    return api.get('/api/patients', { params })
  },

  /** 患者详情 */
  getById(id) {
    return api.get(`/api/patients/${id}`)
  },
}

// ========================================================================
// 检查管理API - Examination API
// ========================================================================
export const examinationApi = {
  /** 上传MRI并分析 */
  uploadAndAnalyze(formData) {
    return api.post('/api/examinations/upload', formData, {
      timeout: 600000,  // 10分钟超时
    })
  },

  /** 检查列表 */
  list(params = {}) {
    return api.get('/api/examinations', { params })
  },

  /** 检查详情 */
  getById(id) {
    return api.get(`/api/examinations/${id}`)
  },

  /** 生成报告 */
  generateReport(id) {
    return api.post(`/api/examinations/${id}/report`)
  },
}

// ========================================================================
// 报告API - Report API
// ========================================================================
export const reportApi = {
  /** 获取报告 */
  getByExamId(examId) {
    return api.get(`/api/reports/${examId}`)
  },

  /** 下载PDF报告 */
  download(examId) {
    return api.get(`/api/reports/${examId}/download`, {
      responseType: 'blob',
    })
  },
}

// ========================================================================
// 统计分析API - Statistics API
// ========================================================================
export const statisticsApi = {
  /** 概览统计 */
  getOverview() {
    return api.get('/api/statistics/overview')
  },

  /** 模型信息 */
  getModelInfo() {
    return api.get('/api/model/info')
  },
}

// ========================================================================
// 系统API - System API
// ========================================================================
export const systemApi = {
  /** 健康检查 */
  healthCheck() {
    return axios.get('/health')
  },
}

export default api
