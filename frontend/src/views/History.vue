<template>
  <!--
    参考来源：Art Design Pro (https://github.com/Daymychen/art-design-pro)
    MIT License - 参考了其数据表格、筛选工具栏、分页布局及状态标签等设计模式。
  -->
  <div class="history-page">
    <div class="top-bar">
      <h2>历史检查记录</h2>
      <div class="top-actions">
        <el-select v-model="filterStatus" clearable placeholder="全部状态" @change="loadList" style="width:130px" size="default">
          <el-option label="已完成" value="completed" />
          <el-option label="处理中" value="processing" />
          <el-option label="失败" value="failed" />
        </el-select>
        <el-button @click="loadList">刷新列表</el-button>
        <el-button type="primary" @click="$router.push('/workspace')">
          <el-icon><Plus /></el-icon> 新建检查
        </el-button>
      </div>
    </div>

    <div class="card">
      <el-table :data="list" v-loading="loading" stripe @row-click="goDetail" style="cursor:pointer" :empty-text="'暂无检查记录，请先在工作台上传影像进行分析'">
        <el-table-column prop="id" label="编号" width="70" align="center" />
        <el-table-column label="患者" width="90" align="center">
          <template #default="{row}">
            <span style="color:#1677ff;font-weight:500">#{{ row.patient_id }}</span>
          </template>
        </el-table-column>
        <el-table-column label="分析状态" width="100" align="center">
          <template #default="{row}">
            <el-tag
              :type="{pending:'info',processing:'warning',completed:'success',failed:'danger'}[row.status]"
              size="small" effect="dark"
            >
              {{ {pending:'待处理',processing:'分析中',completed:'已完成',failed:'失败'}[row.status] || row.status }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="预测分级" width="130" align="center">
          <template #default="{row}">
            <span v-if="row.predicted_who_grade != null" class="grade-dot" :class="'g'+row.predicted_who_grade">
              {{ ['I 级 · 良性','II 级 · 低级别','III 级 · 间变性','IV 级 · 胶质母细胞瘤'][row.predicted_who_grade] }}
            </span>
            <span v-else style="color:#ccc">—</span>
          </template>
        </el-table-column>
        <el-table-column label="恶性概率" width="140">
          <template #default="{row}">
            <div v-if="row.malignant_probability != null" style="display:flex;align-items:center;gap:8px">
              <el-progress
                :percentage="Math.round(row.malignant_probability * 100)"
                :color="row.malignant_probability > 0.5 ? '#ff4d4f' : '#faad14'"
                :stroke-width="6" :show-text="false" style="flex:1"
              />
              <span :style="{color: row.malignant_probability > 0.5 ? '#ff4d4f' : '#faad14', fontWeight: 600, fontSize: '13px'}">
                {{ Math.round(row.malignant_probability * 100) }}%
              </span>
            </div>
            <span v-else style="color:#ccc">—</span>
          </template>
        </el-table-column>
        <el-table-column label="肿瘤体积" width="120" align="center">
          <template #default="{row}">
            <span v-if="row.tumor_volume_ml" style="font-weight:500">{{ row.tumor_volume_ml.toFixed(2) }} ml</span>
            <span v-else style="color:#ccc">—</span>
          </template>
        </el-table-column>
        <el-table-column label="处理耗时" width="90" align="center">
          <template #default="{row}">
            <span v-if="row.processing_time_seconds">{{ row.processing_time_seconds }}秒</span>
            <span v-else style="color:#ccc">—</span>
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="检查时间" width="170" align="center" />
        <el-table-column label="操作" fixed="right" width="160" align="center">
          <template #default="{row}">
            <el-button link type="primary" size="small" @click.stop="goDetail(row)" :disabled="row.status !== 'completed'">
              查看详情
            </el-button>
            <el-button link type="success" size="small" @click.stop="downloadReport(row)" :disabled="row.status !== 'completed'">
              下载报告
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <div style="display:flex;justify-content:center;margin-top:16px">
        <el-pagination
          v-model:current-page="page" :total="total" layout="total, prev, pager, next"
          @current-change="loadList" background
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Plus } from '@element-plus/icons-vue'
import { examinationApi, reportApi } from '@/api'
import { saveAs } from 'file-saver'

const router = useRouter()
const loading = ref(false)
const list = ref([])
const filterStatus = ref('')
const page = ref(1)
const total = ref(0)

const loadList = async () => {
  loading.value = true
  try {
    const res = await examinationApi.list({ skip: (page.value-1)*20, limit: 20, status: filterStatus.value || undefined })
    list.value = res
    total.value = res.length
  } catch { list.value = [] }
  finally { loading.value = false }
}
onMounted(loadList)

const goDetail = (row) => {
  if (row.status === 'completed') router.push(`/result/${row.id}`)
  else ElMessage.warning('该检查尚未完成分析，无法查看详情')
}
const downloadReport = async (row) => {
  try {
    const blob = await reportApi.download(row.id)
    saveAs(blob, `脑肿瘤MRI诊断报告_检查${row.id}.pdf`)
    ElMessage.success('PDF 报告下载成功')
  } catch { ElMessage.warning('请先生成诊断报告后再下载') }
}
</script>

<style scoped>
.history-page { max-width: 1400px; margin: 0 auto; }

.top-bar {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 16px;
}
.top-bar h2 { font-size: 18px; font-weight: 600; color: #1a1a1a; margin: 0; }
.top-actions { display: flex; gap: 8px; }

.card {
  background: #fff; border-radius: 10px; padding: 20px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.grade-dot {
  display: inline-block; padding: 2px 10px; border-radius: 12px;
  font-size: 12px; color: #fff; font-weight: 600;
}
.g0 { background:#52c41a; } .g1 { background:#faad14; }
.g2 { background:#ff7a45; } .g3 { background:#ff4d4f; }
</style>
