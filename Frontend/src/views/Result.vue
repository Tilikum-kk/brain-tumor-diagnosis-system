<template>
  <!--
    参考来源：Art Design Pro (https://github.com/Daymychen/art-design-pro)
    MIT License - 参考了其数据卡片布局、进度仪表盘样式及统计指标展示等设计模式。
  -->
  <div class="result-page">
    <!-- 返回导航 -->
    <div class="nav-bar">
      <el-button text size="default" @click="$router.push('/workspace')">
        <el-icon><ArrowLeft /></el-icon> 返回工作台
      </el-button>
      <el-divider direction="vertical" />
      <span class="nav-title">检查编号 <strong>#{{ examId }}</strong></span>
      <span v-if="examInfo" style="margin-left:12px">
        <el-tag :type="examInfo.status === 'completed' ? 'success' : 'warning'" size="small" effect="dark">
          {{ examInfo.status === 'completed' ? '分析已完成' : examInfo.status }}
        </el-tag>
      </span>
    </div>

    <!-- 加载中骨架屏 -->
    <div v-if="loading" class="loading-block" v-loading="true" element-loading-text="正在加载分析结果...">
      <div style="height:400px"></div>
    </div>

    <el-row v-else :gutter="20">
      <!-- 左侧列：影像可视化 -->
      <el-col :span="16">
        <!-- 分割覆盖图 -->
        <div class="card">
          <div class="card-title">
            <el-icon><PictureFilled /></el-icon> 肿瘤分割覆盖图
            <span class="card-sub">红色 = 增强肿瘤 · 蓝色 = 水肿 · 绿色 = 坏死核心</span>
          </div>
          <div class="img-container">
            <el-image :src="`/api/examinations/${examId}/visualization/overlay`" fit="contain" style="max-height:420px">
              <template #error>
                <div class="img-placeholder">
                  <el-icon :size="40"><PictureFilled /></el-icon>
                  <p>影像数据加载中或暂未生成</p>
                </div>
              </template>
            </el-image>
          </div>
        </div>

        <!-- 注意力热力图 -->
        <div class="card" style="margin-top:20px">
          <div class="card-title">
            <el-icon><DataAnalysis /></el-icon> 超图注意力热力图
            <span class="card-sub">颜色越亮表示该区域对诊断决策贡献越大</span>
          </div>
          <div class="img-container">
            <el-image :src="`/api/examinations/${examId}/visualization/heatmap`" fit="contain" style="max-height:340px">
              <template #error>
                <div class="img-placeholder">
                  <el-icon :size="40"><DataAnalysis /></el-icon>
                  <p>热力图数据加载中或暂未生成</p>
                </div>
              </template>
            </el-image>
          </div>
        </div>

        <!-- 原图/预测对比图 -->
        <div class="card" style="margin-top:20px">
          <div class="card-title">
            <el-icon><PictureFilled /></el-icon> 多模态 + 分割对比
            <span class="card-sub">四种MRI模态与AI分割结果全景</span>
          </div>
          <div class="img-container">
            <el-image :src="`/api/examinations/${examId}/visualization/comparison`" fit="contain" style="max-height:380px">
              <template #error>
                <div class="img-placeholder">
                  <el-icon :size="40"><PictureFilled /></el-icon>
                  <p>对比图生成中或暂未生成</p>
                </div>
              </template>
            </el-image>
          </div>
        </div>
      </el-col>

      <!-- 右侧列：数据指标 -->
      <el-col :span="8">
        <!-- 分级预测 -->
        <div class="card">
          <div class="card-title">
            <el-icon><Collection /></el-icon> WHO 分级预测
          </div>
          <div class="grade-block">
            <span class="grade-tag" :class="'grade-'+ (examInfo?.predicted_who_grade ?? 0)">
              {{ gradeText }}
            </span>
          </div>
          <div class="gauge-wrap">
            <el-progress
              type="dashboard"
              :percentage="Math.round((examInfo?.malignant_probability || 0) * 100)"
              :color="(examInfo?.malignant_probability || 0) > 0.5 ? '#ff4d4f' : '#faad14'"
              :width="160"
            >
              <template #default="{ percentage }">
                <span class="gauge-num">{{ percentage }}%</span>
                <span class="gauge-txt">恶性概率</span>
              </template>
            </el-progress>
          </div>
          <div class="risk-tag" :class="(examInfo?.malignant_probability || 0) > 0.5 ? 'high-risk' : 'low-risk'">
            {{ (examInfo?.malignant_probability || 0) > 0.5 ? '⚠ 高风险 — 建议尽快活检' : '✓ 低风险 — 建议随访观察' }}
          </div>
        </div>

        <!-- 体积测量 -->
        <div class="card" style="margin-top:20px">
          <div class="card-title">
            <el-icon><Odometer /></el-icon> 体积测量结果
          </div>
          <div class="vol-list">
            <div class="vol-row">
              <div class="vol-info">
                <span class="vol-name">增强肿瘤区</span>
                <span class="vol-desc">活跃的肿瘤组织</span>
              </div>
              <span class="vol-num danger">{{ (examInfo?.enhancing_volume_ml || 0).toFixed(2) }} ml</span>
            </div>
            <div class="vol-row">
              <div class="vol-info">
                <span class="vol-name">瘤周水肿区</span>
                <span class="vol-desc">脑组织水肿范围</span>
              </div>
              <span class="vol-num warning">{{ (examInfo?.edema_volume_ml || 0).toFixed(2) }} ml</span>
            </div>
            <div class="vol-row">
              <div class="vol-info">
                <span class="vol-name">肿瘤总负荷</span>
                <span class="vol-desc">全部病灶体积之和</span>
              </div>
              <span class="vol-num primary">{{ (examInfo?.tumor_volume_ml || 0).toFixed(2) }} ml</span>
            </div>
            <div class="vol-row" v-if="examInfo?.processing_time_seconds">
              <div class="vol-info">
                <span class="vol-name">AI 处理耗时</span>
                <span class="vol-desc">端到端推理时间</span>
              </div>
              <span class="vol-num">{{ examInfo.processing_time_seconds }} 秒</span>
            </div>
          </div>
        </div>

        <!-- 模态贡献度 -->
        <div class="card" style="margin-top:20px" v-if="examInfo?.modality_contributions">
          <div class="card-title">
            <el-icon><Connection /></el-icon> 模态贡献度分析
            <span class="card-sub">超图注意力权重</span>
          </div>
          <div class="mod-list">
            <div v-for="(val, mod) in examInfo.modality_contributions" :key="mod" class="mod-row">
              <span class="mod-label">{{ {t1:'T1 加权', t1ce:'T1 增强', t2:'T2 加权', flair:'FLAIR'}[mod] }}</span>
              <el-progress
                :percentage="Math.round(val * 100)"
                :color="{t1:'#1677ff', t1ce:'#ff4d4f', t2:'#52c41a', flair:'#faad14'}[mod]"
                :stroke-width="10"
                style="flex:1;margin:0 10px"
              />
              <span class="mod-pct">{{ (val * 100).toFixed(0) }}%</span>
            </div>
          </div>
          <el-alert
            title="说明"
            type="info"
            :closable="false"
            show-icon
            style="margin-top:12px"
          >
            <p style="font-size:12px;line-height:1.7;margin:0;color:#666">
              超图注意力机制自动学习每种 MRI 模态对肿瘤诊断的贡献权重。
              权重越高，说明该模态在此次诊断中提供的关键信息越多。
              通常 T1 增强和 FLAIR 对高级别胶质瘤贡献最大。
            </p>
          </el-alert>
        </div>

        <!-- 操作 -->
        <div class="actions" style="margin-top:20px">
          <el-button type="primary" size="large" @click="handleReport" :loading="reporting" style="width:100%">
            <el-icon><DocumentAdd /></el-icon> 生成 PDF 诊断报告
          </el-button>
          <el-button size="large" @click="$router.push('/workspace')" style="width:100%;margin-top:10px">
            <el-icon><RefreshLeft /></el-icon> 开始新的分析
          </el-button>
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import {
  ArrowLeft, PictureFilled, DataAnalysis, Collection,
  Odometer, Connection, DocumentAdd, RefreshLeft,
} from '@element-plus/icons-vue'
import { examinationApi } from '@/api'

const route = useRoute()
const router = useRouter()
const examId = computed(() => route.params.id)
const examInfo = ref(null)
const reporting = ref(false)
const loading = ref(true)

onMounted(async () => {
  try {
    examInfo.value = await examinationApi.getById(Number(examId.value))
  } catch {
    ElMessage.error('加载检查记录失败')
  } finally {
    loading.value = false
  }
})

const gradeNames = { 0: 'WHO I 级（良性）', 1: 'WHO II 级（低级别胶质瘤）', 2: 'WHO III 级（间变性胶质瘤）', 3: 'WHO IV 级（胶质母细胞瘤）' }
const gradeText = computed(() => gradeNames[examInfo.value?.predicted_who_grade] || '分级未知')

const handleReport = async () => {
  reporting.value = true
  try {
    await examinationApi.generateReport(Number(examId.value))
    ElMessage.success('PDF 诊断报告已生成')
  } catch { ElMessage.error('报告生成失败') }
  finally { reporting.value = false }
}
</script>

<style scoped>
.result-page { max-width: 1400px; margin: 0 auto; }
.loading-block { background: #fff; border-radius: 10px; padding: 40px; }

.nav-bar {
  display: flex; align-items: center; gap: 8px;
  background: #fff; padding: 12px 20px; border-radius: 8px;
  margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.nav-title { font-size: 15px; color: #333; }
.nav-title strong { color: #1677ff; }

.card {
  background: #fff; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  padding: 20px;
}
.card-title {
  display: flex; align-items: center; gap: 8px;
  font-size: 15px; font-weight: 600; color: #1a1a1a;
  margin-bottom: 16px; padding-bottom: 10px; border-bottom: 1px solid #f0f0f0;
}
.card-sub { font-size: 12px; color: #bbb; font-weight: 400; margin-left: auto; }

.img-container {
  background: #0a1628; border-radius: 8px; min-height: 280px;
  display: flex; align-items: center; justify-content: center;
  overflow: hidden;
}
.img-placeholder {
  text-align: center; color: #666; padding: 40px;
}
.img-placeholder p { margin-top: 10px; font-size: 13px; }

/* 分级 */
.grade-block { text-align: center; padding: 10px 0; }
.grade-tag {
  display: inline-block; padding: 8px 24px; border-radius: 20px;
  font-size: 17px; font-weight: 700; color: #fff;
}
.grade-0 { background: #52c41a; } .grade-1 { background: #faad14; }
.grade-2 { background: #ff7a45; } .grade-3 { background: #ff4d4f; }

.gauge-wrap { text-align: center; margin: 10px 0; }
.gauge-num { font-size: 28px; font-weight: 700; color: #333; }
.gauge-txt { font-size: 12px; color: #999; display: block; }

.risk-tag {
  text-align: center; padding: 8px; border-radius: 6px; font-size: 13px; font-weight: 500; margin-top: 8px;
}
.high-risk { background: #fff1f0; color: #cf1322; }
.low-risk { background: #f6ffed; color: #389e0d; }

/* 体积 */
.vol-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 0; border-bottom: 1px solid #f5f5f5;
}
.vol-name { font-size: 14px; color: #333; display: block; }
.vol-desc { font-size: 12px; color: #bbb; }
.vol-num { font-size: 18px; font-weight: 700; color: #333; }
.vol-num.primary { color: #1677ff; } .vol-num.danger { color: #ff4d4f; } .vol-num.warning { color: #faad14; }

/* 模态 */
.mod-row { display: flex; align-items: center; margin-bottom: 12px; }
.mod-label { width: 64px; font-size: 12px; color: #666; flex-shrink: 0; }
.mod-pct { width: 32px; text-align: right; font-size: 13px; font-weight: 600; }

.actions { display: flex; flex-direction: column; }
</style>
