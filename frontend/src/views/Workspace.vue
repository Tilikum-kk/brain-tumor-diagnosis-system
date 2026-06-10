<template>
  <!--
    参考来源：Art Design Pro (https://github.com/Daymychen/art-design-pro)
    MIT License - 参考了其步骤条进度引导、卡片式信息层级、
    CSS变量驱动的配色体系及过渡动效等设计模式。
  -->
  <div class="workspace">
    <!-- 步骤引导条 -->
    <div class="steps-bar">
      <div class="step-item" :class="{ active: currentStep >= 1, done: currentStep > 1 }">
        <div class="step-num">{{ currentStep > 1 ? '✓' : '1' }}</div>
        <div class="step-text">
          <div class="step-title">选择患者</div>
          <div class="step-desc">搜索或创建患者信息</div>
        </div>
      </div>
      <div class="step-line" :class="{ active: currentStep >= 2 }"></div>
      <div class="step-item" :class="{ active: currentStep >= 2, done: currentStep > 2 }">
        <div class="step-num">{{ currentStep > 2 ? '✓' : '2' }}</div>
        <div class="step-text">
          <div class="step-title">上传影像</div>
          <div class="step-desc">选择 MRI 模态文件</div>
        </div>
      </div>
      <div class="step-line" :class="{ active: currentStep >= 3 }"></div>
      <div class="step-item" :class="{ active: currentStep >= 3, done: currentStep > 3 }">
        <div class="step-num">3</div>
        <div class="step-text">
          <div class="step-title">AI 分析</div>
          <div class="step-desc">超图融合 + 智能诊断</div>
        </div>
      </div>
    </div>

    <!-- 主工作区 -->
    <el-row :gutter="20">
      <!-- 左侧：影像上传区 -->
      <el-col :span="16">
        <div class="card upload-card">
          <div class="card-header">
            <div class="card-title-group">
              <span class="card-icon"><el-icon><PictureFilled /></el-icon></span>
              <div>
                <h3>上传 MRI 影像</h3>
                <p>支持 NIfTI 格式文件（.nii / .nii.gz），至少上传一种模态即可分析</p>
              </div>
            </div>
            <div class="header-actions">
              <!-- 选择案例文件夹 -->
              <input
                ref="folderInput"
                type="file"
                webkitdirectory
                hidden
                @change="handleFolderPick"
              />
              <el-button type="primary" plain size="small" @click="folderInput?.click()">
                <el-icon><FolderOpened /></el-icon> 选择案例文件夹
              </el-button>
              <el-tag size="small" effect="plain">已上传 {{ uploadedCount }} / 4 种模态</el-tag>
            </div>
          </div>

          <div class="modal-grid">
            <div
              v-for="m in modalities"
              :key="m.key"
              class="modal-box"
              :class="{ filled: m.file, hovering: m.key === hoverMod }"
              @dragover.prevent="hoverMod = m.key"
              @dragleave="hoverMod = null"
              @drop.prevent="handleDrop(m.key, $event)"
              @click="triggerUpload(m.key)"
            >
              <input
                :ref="el => inputs[m.key] = el"
                type="file" :accept="'.nii,.nii.gz,.gz'" hidden
                @change="e => handlePick(m.key, e)"
              />

              <template v-if="!m.file">
                <div class="modal-icon-box" :style="{ background: m.bg }">
                  <span class="modal-abbr">{{ m.abbr }}</span>
                </div>
                <div class="modal-name">{{ m.fullName }}</div>
                <div class="modal-desc">{{ m.desc }}</div>
                <div class="modal-action">点击上传或拖拽文件</div>
              </template>

              <template v-else>
                <div class="modal-icon-box done">
                  <el-icon :size="22"><Check /></el-icon>
                </div>
                <div class="modal-name">{{ m.file.name }}</div>
                <div class="modal-desc">{{ formatSize(m.file.size) }}</div>
                <el-button text type="danger" size="small" class="modal-remove" @click.stop="m.file = null">
                  移除文件
                </el-button>
              </template>
            </div>
          </div>
        </div>
      </el-col>

      <!-- 右侧：患者信息 + 开始分析 -->
      <el-col :span="8">
        <div class="card">
          <div class="card-header">
            <div class="card-title-group">
              <span class="card-icon"><el-icon><UserFilled /></el-icon></span>
              <div>
                <h3>患者信息</h3>
                <p>选择已有患者或快速创建新患者</p>
              </div>
            </div>
          </div>

          <!-- 搜索已有患者 -->
          <div class="form-section">
            <label class="form-label">搜索已有患者</label>
            <el-select
              v-model="selectedPatientId"
              filterable
              remote
              :remote-method="searchPatients"
              :loading="searching"
              placeholder="输入患者编码或姓名搜索…"
              clearable
              class="full-width"
              popper-class="patient-dropdown"
              @change="onPatientSelected"
            >
              <el-option
                v-for="p in patientOptions"
                :key="p.id"
                :label="`${p.patient_code}`"
                :value="p.id"
              >
                <div class="patient-row">
                  <span class="p-code">{{ p.patient_code }}</span>
                  <span class="p-name">{{ p.full_name || '未填写姓名' }}</span>
                  <span v-if="p.age" class="p-meta">{{ p.age }}岁 · {{ p.gender === 'male' ? '男' : '女' }}</span>
                </div>
              </el-option>
            </el-select>

            <el-divider>
              <span style="font-size:12px;color:#bbb">或者快速创建新患者</span>
            </el-divider>

            <!-- 快速创建 -->
            <el-row :gutter="10">
              <el-col :span="12">
                <div class="form-section">
                  <label class="form-label">患者编码 <span class="required">*</span></label>
                  <el-input v-model="quick.code" placeholder="如 PT2024001" size="default" />
                </div>
              </el-col>
              <el-col :span="12">
                <div class="form-section">
                  <label class="form-label">姓名</label>
                  <el-input v-model="quick.name" placeholder="患者姓名" size="default" />
                </div>
              </el-col>
            </el-row>

            <el-row :gutter="10">
              <el-col :span="12">
                <div class="form-section">
                  <label class="form-label">年龄</label>
                  <el-input-number v-model="quick.age" :min="0" :max="120" placeholder="岁" style="width:100%" />
                </div>
              </el-col>
              <el-col :span="12">
                <div class="form-section">
                  <label class="form-label">性别</label>
                  <el-radio-group v-model="quick.gender" size="small">
                    <el-radio-button value="male">男</el-radio-button>
                    <el-radio-button value="female">女</el-radio-button>
                  </el-radio-group>
                </div>
              </el-col>
            </el-row>

            <el-row :gutter="10">
              <el-col :span="12">
                <div class="form-section">
                  <label class="form-label">已知 WHO 分级</label>
                  <el-select v-model="quick.grade" placeholder="未知" style="width:100%" size="default">
                    <el-option label="未知" :value="null" />
                    <el-option label="I 级 · 良性" :value="0" />
                    <el-option label="II 级 · 低级别" :value="1" />
                    <el-option label="III 级 · 间变性" :value="2" />
                    <el-option label="IV 级 · 胶质母细胞瘤" :value="3" />
                  </el-select>
                </div>
              </el-col>
              <el-col :span="12">
                <div class="form-section">
                  <label class="form-label">肿瘤位置</label>
                  <el-select v-model="quick.location" placeholder="请选择肿瘤所在脑区" style="width:100%" size="default">
                    <el-option label="未知" value="" />
                    <el-option label="额叶" value="frontal" />
                    <el-option label="颞叶" value="temporal" />
                    <el-option label="顶叶" value="parietal" />
                    <el-option label="枕叶" value="occipital" />
                    <el-option label="小脑" value="cerebellum" />
                    <el-option label="脑干" value="brainstem" />
                    <el-option label="丘脑" value="thalamus" />
                  </el-select>
                </div>
              </el-col>
            </el-row>

            <div class="form-section">
              <label class="form-label">
                Karnofsky 功能评分
                <el-tooltip placement="right" effect="dark">
                  <template #content>
                    <div style="max-width:300px;line-height:1.7">
                      <p><b>Karnofsky 功能状态评分 (KPS)</b> 用于评估肿瘤患者的日常生活能力：</p>
                      <p>100分 = 正常，无任何症状<br/>
                      90分 = 能正常活动，有轻微症状<br/>
                      80分 = 勉强正常活动，有一些症状<br/>
                      70分 = 生活可自理，不能正常活动<br/>
                      60分 = 生活大部分可自理，偶尔需帮助<br/>
                      50分 = 常需帮助和医疗护理<br/>
                      40分 = 生活不能自理，需特殊照顾<br/>
                      30分 = 生活严重不能自理<br/>
                      20分 = 病重，需住院积极治疗<br/>
                      10分 = 病危，濒临死亡<br/>
                      0分 = 死亡</p>
                    </div>
                  </template>
                  <el-icon :size="14" style="margin-left:4px;color:#999;cursor:help"><QuestionFilled /></el-icon>
                </el-tooltip>
              </label>
              <el-slider v-model="quick.kps" :min="0" :max="100" :step="10" show-input :marks="{0:'0',50:'50',100:'100'}" />
            </div>
          </div>

          <!-- 开始分析按钮 -->
          <el-button
            type="primary"
            size="large"
            :loading="analyzing"
            :disabled="!canAnalyze"
            class="analyze-btn"
            @click="handleAnalyze"
          >
            <el-icon v-if="!analyzing"><VideoPlay /></el-icon>
            {{ analyzing ? 'AI 正在分析中，请稍候…' : '开始 AI 辅助诊断' }}
          </el-button>

          <div v-if="!canAnalyze && !analyzing" class="hint-text">
            <el-icon><Warning /></el-icon>
            <span>请先选择或创建患者，并至少上传一种 MRI 影像</span>
          </div>
        </div>
      </el-col>
    </el-row>

    <!-- 分析进度弹窗 -->
    <el-dialog v-model="showProgress" title="AI 分析进行中" width="500px" :close-on-click-modal="false" :show-close="false">
      <div class="progress-content">
        <el-steps :active="analyzeStep" align-center finish-status="success">
          <el-step title="上传影像" description="验证文件完整性" />
          <el-step title="图像预处理" description="重采样 · 归一化 · 颅骨剥离" />
          <el-step title="超图融合推理" description="HG-MFNet · 三级融合策略" />
          <el-step title="生成诊断结果" description="分割 · 分类 · 体积测量" />
        </el-steps>
        <div class="progress-bar-wrap">
          <el-progress :percentage="analyzeStep * 25 + 25" :color="'#1677ff'" :stroke-width="8" :indeterminate="false" />
        </div>
      </div>
    </el-dialog>

    <!-- 分析结果面板 -->
    <div v-if="result" class="result-area">
      <div class="card result-card">
        <div class="result-header">
          <div class="result-title-row">
            <el-icon :size="20" color="#52c41a"><SuccessFilled /></el-icon>
            <h3>AI 分析完成</h3>
            <el-tag type="success" effect="dark" size="small">耗时 {{ result.processing_time }} 秒</el-tag>
          </div>
        </div>

        <el-row :gutter="20">
          <!-- 分割结果 -->
          <el-col :span="8">
            <div class="metric-card">
              <div class="metric-title">肿瘤分割结果</div>
              <div class="metric-list">
                <div class="metric-item">
                  <span class="metric-label">增强肿瘤区</span>
                  <span class="metric-value danger">{{ (result.volumes?.enhancing_ml || 0).toFixed(2) }} ml</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">瘤周水肿区</span>
                  <span class="metric-value warning">{{ (result.volumes?.edema_ml || 0).toFixed(2) }} ml</span>
                </div>
                <div class="metric-item">
                  <span class="metric-label">肿瘤总体积</span>
                  <span class="metric-value primary">{{ (result.volumes?.total_tumor_ml || 0).toFixed(2) }} ml</span>
                </div>
              </div>
            </div>
          </el-col>

          <!-- 分类结果 -->
          <el-col :span="8">
            <div class="metric-card">
              <div class="metric-title">WHO 分级预测</div>
              <div class="grade-show">
                <span class="grade-badge" :class="'grade-' + (result.classification?.predicted_who_grade ?? 0)">
                  {{ gradeLabel }}
                </span>
              </div>
              <div class="malignant-bar">
                <span>恶性风险概率</span>
                <el-progress
                  :percentage="Math.round((result.classification?.malignant_probability || 0) * 100)"
                  :color="(result.classification?.malignant_probability || 0) > 0.5 ? '#ff4d4f' : '#faad14'"
                  :stroke-width="12"
                />
              </div>
            </div>
          </el-col>

          <!-- 模态贡献 -->
          <el-col :span="8">
            <div class="metric-card">
              <div class="metric-title">模态贡献度（超图注意力）</div>
              <div class="contrib-list">
                <div v-for="(val, mod) in result.modality_contributions" :key="mod" class="contrib-item">
                  <span class="contrib-name">{{ modLabels[mod] }}</span>
                  <el-progress
                    :percentage="Math.round(val * 100)"
                    :color="modColors[mod]"
                    :stroke-width="10"
                    style="flex:1;margin:0 10px"
                  />
                  <span class="contrib-pct">{{ (val * 100).toFixed(0) }}%</span>
                </div>
              </div>
            </div>
          </el-col>
        </el-row>

        <!-- 操作按钮 -->
        <div class="result-actions">
          <el-button type="primary" size="large" @click="goResult">
            <el-icon><View /></el-icon> 查看详细影像结果
          </el-button>
          <el-button type="success" size="large" @click="makeReport">
            <el-icon><DocumentAdd /></el-icon> 生成 PDF 诊断报告
          </el-button>
          <el-button size="large" @click="resetAll">
            <el-icon><RefreshLeft /></el-icon> 开始新的分析
          </el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import {
  PictureFilled, UserFilled, Check, VideoPlay, Warning,
  SuccessFilled, View, DocumentAdd, RefreshLeft,
} from '@element-plus/icons-vue'
import { examinationApi, patientApi, statisticsApi } from '@/api'

const router = useRouter()

// ===== 模态配置 =====
const modLabels = { t1: 'T1 加权', t1ce: 'T1 增强', t2: 'T2 加权', flair: 'FLAIR' }
const modColors = { t1: '#1677ff', t1ce: '#ff4d4f', t2: '#52c41a', flair: '#faad14' }

const modalities = reactive([
  { key: 't1', abbr: 'T1', fullName: 'T1 加权成像', desc: '显示正常解剖结构，用于定位', bg: '#e6f4ff', color: '#1677ff', file: null },
  { key: 't1ce', abbr: 'T1C', fullName: 'T1 对比增强', desc: '造影剂增强后扫描，显示血脑屏障破坏区域', bg: '#fff1f0', color: '#ff4d4f', file: null },
  { key: 't2', abbr: 'T2', fullName: 'T2 加权成像', desc: '显示瘤周水肿和脑脊液信号', bg: '#f6ffed', color: '#52c41a', file: null },
  { key: 'flair', abbr: 'FL', fullName: 'T2 FLAIR', desc: '抑制脑脊液信号，清晰显示水肿边界', bg: '#fffbe6', color: '#faad14', file: null },
])

const uploadedCount = computed(() => modalities.filter(m => m.file).length)

// ===== 文件上传 =====
const inputs = reactive({})
const folderInput = ref(null)

// 文件夹上传：自动识别文件类型
const handleFolderPick = (e) => {
  const files = Array.from(e.target.files || [])
  if (files.length === 0) return

  // 关键字符匹配：文件名包含 t1 / t1ce / t2 / flair + .nii / .nii.gz
  const rules = [
    { key: 't1', patterns: ['_t1.nii', '_t1.'], exclude: ['t1ce'] },
    { key: 't1ce', patterns: ['_t1ce.nii', '_t1ce.'] },
    { key: 't2', patterns: ['_t2.nii', '_t2.'], exclude: ['flair'] },
    { key: 'flair', patterns: ['_flair.nii', '_flair.'] },
  ]

  let matched = 0
  for (const rule of rules) {
    const found = files.find(f => {
      const name = f.name.toLowerCase()
      const match = rule.patterns.some(p => name.includes(p))
      if (!match) return false
      if (rule.exclude && rule.exclude.some(e => name.includes(e))) return false
      return true
    })
    if (found) {
      const mod = modalities.find(m => m.key === rule.key)
      if (mod) { mod.file = found; matched++ }
    }
  }

  // 如果规则没匹配全，尝试更宽松的文件名包含匹配
  if (matched < 4) {
    for (const f of files) {
      const name = f.name.toLowerCase()
      for (const rule of rules) {
        const mod = modalities.find(m => m.key === rule.key)
        if (mod.file) continue  // 已匹配
        if (rule.patterns.some(p => name.includes(p))) {
          if (rule.exclude && rule.exclude.some(e => name.includes(e))) continue
          mod.file = f
          matched++
          break
        }
      }
    }
  }

  if (matched > 0) {
    ElMessage.success(`已自动识别文件夹中的 ${matched} 个 MRI 模态文件`)
  } else {
    ElMessage.warning('未能识别到 BraTS 格式文件，请检查文件夹内容')
  }

  // 重置 input 以允许重复选择同一文件夹
  e.target.value = ''
}
const hoverMod = ref(null)

const triggerUpload = (key) => {
  const el = inputs[key]
  if (el instanceof HTMLInputElement) el.click()
}
const handlePick = (key, e) => {
  const f = e.target.files?.[0]
  if (f) { const m = modalities.find(x => x.key === key); if (m) m.file = f }
}
const handleDrop = (key, e) => {
  hoverMod.value = null
  const f = e.dataTransfer?.files?.[0]
  if (f) { const m = modalities.find(x => x.key === key); if (m) m.file = f }
}
const formatSize = (s) => s ? (s < 1e6 ? (s/1024).toFixed(0)+' KB' : (s/1e6).toFixed(1)+' MB') : ''

// ===== 患者 =====
const selectedPatientId = ref(null)
const searching = ref(false)
const patientOptions = ref([])

const quick = reactive({ code: '', name: '', age: null, gender: 'male', grade: null, location: '', kps: 80 })

const searchPatients = async (q) => {
  if (!q) return
  searching.value = true
  try { patientOptions.value = await patientApi.list({ search: q, limit: 10 }) }
  catch { patientOptions.value = [] }
  finally { searching.value = false }
}
const onPatientSelected = () => { currentStep.value = Math.max(currentStep.value, 1) }

onMounted(async () => {
  try { patientOptions.value = await patientApi.list({ limit: 20 }) }
  catch { patientOptions.value = [] }
})

// ===== 步骤状态 =====
const currentStep = ref(1)

// ===== 分析 =====
const canAnalyze = computed(() => {
  return uploadedCount.value > 0 && (selectedPatientId.value || quick.code)
})

const analyzing = ref(false)
const showProgress = ref(false)
const analyzeStep = ref(0)
const result = ref(null)
let stepTimer = null

const gradeLabel = computed(() => {
  const g = result.value?.classification?.predicted_who_grade
  return ['WHO I 级（良性）', 'WHO II 级（低级别胶质瘤）', 'WHO III 级（间变性胶质瘤）', 'WHO IV 级（胶质母细胞瘤）'][g] || '未知'
})

const handleAnalyze = async () => {
  analyzing.value = true
  showProgress.value = true
  analyzeStep.value = 0
  result.value = null
  currentStep.value = 3

  stepTimer = setInterval(() => { if (analyzeStep.value < 3) analyzeStep.value++ }, 900)

  try {
    let pid = selectedPatientId.value
    if (!pid && quick.code) {
      try {
        const p = await patientApi.create({
          patient_code: quick.code,
          full_name: quick.name || undefined,
          age: quick.age,
          gender: quick.gender,
          who_grade: quick.grade,
          tumor_location: quick.location || undefined,
          karnofsky_score: quick.kps,
        })
        pid = p.id
        ElMessage.success('新患者已创建')
      } catch {
        // 患者可能已存在，尝试搜索获取 ID
        try {
          const found = await patientApi.list({ search: quick.code, limit: 1 })
          if (found.length > 0) {
            pid = found[0].id
            ElMessage.info('患者已存在，使用已有记录')
          }
        } catch { /* 搜索也失败了 */ }
      }
    }

    if (!pid) {
      ElMessage.error('请先选择或创建患者')
      analyzing.value = false
      clearInterval(stepTimer)
      showProgress.value = false
      return
    }

    const fd = new FormData()
    fd.append('patient_id', pid)
    for (const m of modalities) { if (m.file) fd.append(`${m.key}_file`, m.file) }
    if (quick.age) fd.append('age', quick.age)
    if (quick.gender) fd.append('gender', quick.gender)
    if (quick.grade !== null) fd.append('who_grade', quick.grade)
    if (quick.location) fd.append('tumor_location', quick.location)
    if (quick.kps) fd.append('karnofsky_score', quick.kps)

    const res = await examinationApi.uploadAndAnalyze(fd)
    result.value = res
    analyzeStep.value = 4
    currentStep.value = 3
    showProgress.value = false
    ElMessage.success('AI 分析完成！请在下方查看诊断结果')
  } catch (e) {
    showProgress.value = false
    currentStep.value = 2
    ElMessage.error('分析失败：' + (e?.response?.data?.detail || e?.message || '服务器错误'))
  } finally {
    analyzing.value = false
    clearInterval(stepTimer)
  }
}

const goResult = () => { if (result.value) router.push(`/result/${result.value.examination_id}`) }
const makeReport = async () => {
  if (!result.value) return
  try { await examinationApi.generateReport(result.value.examination_id); ElMessage.success('报告已生成'); goResult() }
  catch { ElMessage.error('报告生成失败') }
}
const resetAll = () => {
  result.value = null; currentStep.value = 1
  modalities.forEach(m => m.file = null)
  selectedPatientId.value = null
  Object.assign(quick, { code: '', name: '', age: null, gender: 'male', grade: null, location: '', kps: 80 })
}
</script>

<style scoped>
.workspace { max-width: 1400px; margin: 0 auto; }

/* ===== 步骤引导条 ===== */
.steps-bar {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0;
  background: #fff;
  border-radius: 10px;
  padding: 20px 40px;
  margin-bottom: 20px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.step-item {
  display: flex;
  align-items: center;
  gap: 10px;
  opacity: 0.4;
  transition: all 0.3s;
}
.step-item.active { opacity: 1; }
.step-item.done { opacity: 1; }
.step-num {
  width: 32px; height: 32px;
  border-radius: 50%;
  background: #e8e8e8;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-weight: 700;
  color: #999;
  flex-shrink: 0;
  transition: all 0.3s;
}
.step-item.active .step-num { background: #1677ff; color: #fff; }
.step-item.done .step-num { background: #52c41a; color: #fff; }
.step-title { font-size: 14px; font-weight: 600; color: #333; }
.step-desc { font-size: 12px; color: #999; }
.step-line {
  width: 60px; height: 2px; background: #e8e8e8; margin: 0 16px; transition: all 0.3s;
}
.step-line.active { background: #1677ff; }

/* ===== 卡片 ===== */
.card {
  background: #fff;
  border-radius: 10px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  margin-bottom: 20px;
}
.card-header {
  padding: 18px 20px;
  border-bottom: 1px solid #f0f0f0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 10px;
}
.card-title-group { display: flex; align-items: center; gap: 12px; }
.card-title-group h3 { font-size: 16px; font-weight: 600; color: #1a1a1a; margin: 0; }
.card-title-group p { font-size: 12px; color: #999; margin: 2px 0 0; }
.card-icon {
  width: 40px; height: 40px; border-radius: 8px; background: #f0f5ff;
  display: flex; align-items: center; justify-content: center; color: #1677ff; font-size: 18px;
}
.header-actions { display: flex; align-items: center; gap: 8px; }

/* ===== 上传区 ===== */
.upload-card { padding-bottom: 20px; }
.modal-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 20px;
}
.modal-box {
  border: 2px dashed #d9d9d9;
  border-radius: 12px;
  padding: 24px 16px;
  text-align: center;
  cursor: pointer;
  transition: all 0.25s;
  background: #fafafa;
  position: relative;
}
.modal-box:hover { border-color: #1677ff; background: #f5f8ff; box-shadow: 0 2px 8px rgba(22,119,255,0.08); }
.modal-box.hovering { border-color: #1677ff; background: #e6f4ff; box-shadow: 0 0 0 3px rgba(22,119,255,0.12); transform: scale(1.01); }
.modal-box.filled { border-color: #52c41a; border-style: solid; background: #f6ffed; }
.modal-icon-box {
  width: 48px; height: 48px; border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  margin: 0 auto 12px; transition: all 0.25s;
}
.modal-icon-box.done { background: #52c41a !important; color: #fff; }
.modal-abbr { font-size: 16px; font-weight: 800; color: #fff; }
.modal-name { font-size: 14px; font-weight: 600; color: #333; margin-bottom: 4px; }
.modal-desc { font-size: 12px; color: #999; margin-bottom: 6px; }
.modal-action { font-size: 12px; color: #bbb; margin-top: 4px; }
.modal-remove { margin-top: 6px; }

/* ===== 患者表单 ===== */
.form-section { margin-bottom: 12px; }
.form-label { font-size: 13px; color: #666; margin-bottom: 4px; display: block; }
.form-label .required { color: #ff4d4f; }
.full-width { width: 100%; }

.hint-text {
  display: flex; align-items: center; gap: 6px;
  margin-top: 12px; padding: 8px 12px; background: #fffbe6; border-radius: 6px;
  font-size: 12px; color: #ad8b00;
}

/* ===== 分析按钮 ===== */
.analyze-btn {
  width: 100%; height: 50px; font-size: 17px; border-radius: 10px; margin-top: 16px;
}

/* ===== 进度弹窗 ===== */
.progress-content { padding: 20px 0; }
.progress-bar-wrap { margin-top: 24px; }

/* ===== 结果区 ===== */
.result-area { animation: fadeIn 0.4s ease; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }

.result-header { padding: 16px 20px; border-bottom: 1px solid #f0f0f0; }
.result-title-row { display: flex; align-items: center; gap: 10px; }
.result-title-row h3 { font-size: 17px; font-weight: 600; margin: 0; }
.result-card { padding-bottom: 20px; }

.metric-card {
  background: #fafafa; border-radius: 10px; padding: 16px; height: 100%;
}
.metric-title { font-size: 14px; font-weight: 600; color: #333; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 2px solid #1677ff; }
.metric-list { display: flex; flex-direction: column; gap: 0; }
.metric-item {
  display: flex; justify-content: space-between; align-items: center;
  padding: 10px 0; border-bottom: 1px solid #f0f0f0; font-size: 13px;
}
.metric-label { color: #666; }
.metric-value { font-weight: 700; font-size: 15px; }
.metric-value.primary { color: #1677ff; }
.metric-value.danger { color: #ff4d4f; }
.metric-value.warning { color: #faad14; }

.grade-show { text-align: center; margin: 12px 0; }
.grade-badge {
  display: inline-block; padding: 8px 22px; border-radius: 20px;
  font-size: 16px; font-weight: 700; color: #fff;
}
.grade-0 { background: #52c41a; } .grade-1 { background: #faad14; }
.grade-2 { background: #ff7a45; } .grade-3 { background: #ff4d4f; }

.malignant-bar { margin-top: 16px; }
.malignant-bar > span { font-size: 13px; color: #666; display: block; margin-bottom: 6px; }

.contrib-item { display: flex; align-items: center; margin-bottom: 10px; }
.contrib-name { width: 64px; font-size: 12px; color: #666; flex-shrink: 0; }
.contrib-pct { width: 32px; text-align: right; font-size: 13px; font-weight: 600; color: #333; }

.result-actions {
  display: flex; gap: 12px; justify-content: center;
  margin-top: 20px; padding-top: 20px; border-top: 1px solid #f0f0f0;
}

/* ===== 患者下拉 ===== */
.patient-row { display: flex; align-items: center; gap: 10px; }
.p-code { font-weight: 600; color: #333; }
.p-name { color: #666; }
.p-meta { color: #999; font-size: 12px; margin-left: auto; }
</style>
