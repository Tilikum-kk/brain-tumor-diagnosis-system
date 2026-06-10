<template>
  <!--
    参考来源：Art Design Pro (https://github.com/Daymychen/art-design-pro)
    MIT License - 参考了其表格列表布局、搜索栏交互及弹窗表单等设计模式。
  -->
  <div class="patients-page">
    <div class="top-bar">
      <h2>患者管理</h2>
      <el-button type="primary" @click="showDialog = true">
        <el-icon><Plus /></el-icon> 添加新患者
      </el-button>
    </div>

    <div class="card">
      <!-- 搜索 -->
      <div class="search-row">
        <el-input v-model="keyword" placeholder="搜索患者编码或姓名" clearable @clear="loadList" @keyup.enter="loadList" style="width:340px" size="default">
          <template #prefix><el-icon><Search /></el-icon></template>
        </el-input>
        <el-button type="primary" @click="loadList">搜索</el-button>
      </div>

      <!-- 表格 -->
      <el-table :data="list" v-loading="loading" stripe :empty-text="'暂无患者数据，请点击右上角添加新患者'">
        <el-table-column prop="id" label="编号" width="70" align="center" />
        <el-table-column prop="patient_code" label="患者编码">
          <template #default="{row}">
            <span style="font-weight:600;color:#1677ff">{{ row.patient_code }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="full_name" label="姓名">
          <template #default="{row}">
            {{ row.full_name || '未填写' }}
          </template>
        </el-table-column>
        <el-table-column label="性别" width="70" align="center">
          <template #default="{row}">
            <el-tag :type="row.gender === 'male' ? 'primary' : 'danger'" size="small" effect="plain">
              {{ row.gender === 'male' ? '男' : row.gender === 'female' ? '女' : '-' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="age" label="年龄" width="70" align="center">
          <template #default="{row}">{{ row.age || '-' }}</template>
        </el-table-column>
        <el-table-column label="WHO 分级" width="110" align="center">
          <template #default="{row}">
            <span v-if="row.who_grade != null" class="grade-dot" :class="'g'+row.who_grade">
              {{ ['I','II','III','IV'][row.who_grade] }} 级
            </span>
            <span v-else style="color:#ccc">未知</span>
          </template>
        </el-table-column>
        <el-table-column label="肿瘤位置" width="100" align="center">
          <template #default="{row}">
            {{ {frontal:'额叶',temporal:'颞叶',parietal:'顶叶',occipital:'枕叶',cerebellum:'小脑',brainstem:'脑干',thalamus:'丘脑'}[row.tumor_location] || row.tumor_location || '-' }}
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="创建时间" width="170" align="center" />
        <el-table-column label="操作" width="120" fixed="right" align="center">
          <template #default="{row}">
            <el-button link type="primary" size="small" @click="goExam(row)">
              发起检查
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

    <!-- 新建患者弹窗 -->
    <el-dialog v-model="showDialog" title="添加新患者" width="480px" :close-on-click-modal="false">
      <el-form :model="form" :rules="rules" ref="formRef" label-width="80px">
        <el-form-item label="患者编码" prop="patient_code">
          <el-input v-model="form.patient_code" placeholder="唯一标识编码，如 PT2024001" />
        </el-form-item>
        <el-form-item label="姓名">
          <el-input v-model="form.full_name" placeholder="患者姓名（选填）" />
        </el-form-item>
        <el-row :gutter="12">
          <el-col :span="12">
            <el-form-item label="年龄">
              <el-input-number v-model="form.age" :min="0" :max="120" style="width:100%" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="性别">
              <el-radio-group v-model="form.gender" size="small">
                <el-radio-button value="male">男</el-radio-button>
                <el-radio-button value="female">女</el-radio-button>
              </el-radio-group>
            </el-form-item>
          </el-col>
        </el-row>
        <el-form-item label="WHO 分级">
          <el-select v-model="form.who_grade" placeholder="未知" style="width:100%">
            <el-option label="未知" :value="null" />
            <el-option label="I 级 · 良性" :value="0" />
            <el-option label="II 级 · 低级别" :value="1" />
            <el-option label="III 级 · 间变性" :value="2" />
            <el-option label="IV 级 · 胶质母细胞瘤" :value="3" />
          </el-select>
        </el-form-item>
        <el-form-item label="肿瘤位置">
          <el-select v-model="form.tumor_location" placeholder="请选择" style="width:100%">
            <el-option label="额叶" value="frontal" />
            <el-option label="颞叶" value="temporal" />
            <el-option label="顶叶" value="parietal" />
            <el-option label="枕叶" value="occipital" />
            <el-option label="小脑" value="cerebellum" />
            <el-option label="脑干" value="brainstem" />
            <el-option label="丘脑" value="thalamus" />
          </el-select>
        </el-form-item>
        <el-form-item label="病史备注">
          <el-input v-model="form.medical_history" type="textarea" :rows="2" placeholder="简要病史，选填" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showDialog = false">取消</el-button>
        <el-button type="primary" :loading="saving" @click="handleCreate">确认创建</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Plus, Search } from '@element-plus/icons-vue'
import { patientApi } from '@/api'

const router = useRouter()
const loading = ref(false)
const list = ref([])
const page = ref(1)
const total = ref(0)
const keyword = ref('')

const loadList = async () => {
  loading.value = true
  try {
    const res = await patientApi.list({ skip: (page.value-1)*20, limit: 20, search: keyword.value || undefined })
    list.value = res
    total.value = res.length
  } catch { list.value = [] }
  finally { loading.value = false }
}
onMounted(loadList)

// 新建患者
const showDialog = ref(false)
const saving = ref(false)
const formRef = ref(null)
const form = reactive({
  patient_code: '', full_name: '', age: null, gender: 'male',
  who_grade: null, tumor_location: '', medical_history: '',
})
const rules = { patient_code: [{ required: true, message: '患者编码不能为空', trigger: 'blur' }] }

const handleCreate = async () => {
  try { await formRef.value?.validate() } catch { return }
  saving.value = true
  try {
    await patientApi.create(form)
    ElMessage.success('患者创建成功')
    showDialog.value = false
    Object.assign(form, { patient_code: '', full_name: '', age: null, gender: 'male', who_grade: null, tumor_location: '', medical_history: '' })
    loadList()
  } catch { /* handled */ }
  finally { saving.value = false }
}

const goExam = () => router.push('/workspace')
</script>

<style scoped>
.patients-page { max-width: 1400px; margin: 0 auto; }

.top-bar {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 16px;
}
.top-bar h2 { font-size: 18px; font-weight: 600; color: #1a1a1a; margin: 0; }

.card {
  background: #fff; border-radius: 10px; padding: 20px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.search-row { display: flex; gap: 8px; margin-bottom: 16px; }

.grade-dot {
  display: inline-block; padding: 2px 10px; border-radius: 12px;
  font-size: 12px; color: #fff; font-weight: 600;
}
.g0 { background:#52c41a; } .g1 { background:#faad14; }
.g2 { background:#ff7a45; } .g3 { background:#ff4d4f; }
</style>
