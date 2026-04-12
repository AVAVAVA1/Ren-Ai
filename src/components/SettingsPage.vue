<script setup>
import { ref, watch, onMounted } from 'vue'

const RUNNINGHUB_WORKFLOW_STORAGE_KEY = 'renai_runninghub_workflow_id'
const IMAGE_BACKEND_STORAGE_KEY = 'renai_image_backend'
const COMFYUI_CKPT_STORAGE_KEY = 'renai_comfyui_checkpoint'
const COMFYUI_WORKFLOW_STORAGE_KEY = 'renai_comfyui_workflow'
const COMFYUI_SIZE_RATIO_STORAGE_KEY = 'renai_comfyui_size_ratio'

const settings = ref({
  theme: 'dark',
  language: 'zh-CN',
  autoSave: true,
  snapToGrid: true,
  gridSize: 20
})

/** RunningHub 工作流 ID，人物卡「生成立绘」时使用 */
const runninghubWorkflowId = ref('')
/** runninghub | comfyui：人物立绘与流程背景共用 */
const imageBackend = ref('runninghub')
/** ComfyUI Checkpoint 文件名，空则使用后端默认（工作流 JSON / .env） */
const comfyuiCheckpoint = ref('')
/** public/comfyui 下工作流 JSON 文件名 */
const comfyUiWorkflowFile = ref('workflow1.json')
const comfyWorkflowOptions = ref([])
const comfyWorkflowsLoadError = ref('')
const sizePresetList = ref([{ ratio: '1.0', width: 1024, height: 1024, label: '1.0 — 1024×1024' }])
const comfySizeRatio = ref('1.0')

async function loadSizePresets() {
  let saved = ''
  try {
    saved = localStorage.getItem(COMFYUI_SIZE_RATIO_STORAGE_KEY) || ''
  } catch {
    /* ignore */
  }
  try {
    const r = await fetch('/api/image/comfyui-size-presets')
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    const j = await r.json()
    const list = Array.isArray(j.presets) ? j.presets : []
    if (list.length) sizePresetList.value = list
    const def = (j.default_ratio || '1.0').trim() || '1.0'
    const ratios = new Set(sizePresetList.value.map((p) => p.ratio))
    if (saved && ratios.has(saved)) comfySizeRatio.value = saved
    else if (ratios.has(def)) comfySizeRatio.value = def
    else comfySizeRatio.value = sizePresetList.value[0]?.ratio || '1.0'
  } catch {
    comfySizeRatio.value = saved || '1.0'
  }
}

async function loadComfyWorkflows() {
  comfyWorkflowsLoadError.value = ''
  let saved = ''
  try {
    saved = localStorage.getItem(COMFYUI_WORKFLOW_STORAGE_KEY) || ''
  } catch {
    /* ignore */
  }
  try {
    const r = await fetch('/api/image/comfyui-workflows')
    if (!r.ok) throw new Error(`HTTP ${r.status}`)
    const j = await r.json()
    const list = Array.isArray(j.workflows) ? j.workflows : []
    comfyWorkflowOptions.value = list
    const def = (j.default_file || 'workflow1.json').trim() || 'workflow1.json'
    const names = new Set(list.map((w) => w.file))
    if (saved && names.has(saved)) comfyUiWorkflowFile.value = saved
    else if (names.has(def)) comfyUiWorkflowFile.value = def
    else if (list[0]?.file) comfyUiWorkflowFile.value = list[0].file
    else comfyUiWorkflowFile.value = def
  } catch (e) {
    comfyWorkflowsLoadError.value = e?.message || String(e)
    comfyWorkflowOptions.value = [
      {
        file: 'workflow1.json',
        label: 'workflow1.json（离线默认）',
        mapping_ok: true,
        mapping_source: 'fallback'
      }
    ]
    comfyUiWorkflowFile.value = saved || 'workflow1.json'
  }
}

onMounted(async () => {
  try {
    const v = localStorage.getItem(RUNNINGHUB_WORKFLOW_STORAGE_KEY)
    if (v) runninghubWorkflowId.value = v
    const b = localStorage.getItem(IMAGE_BACKEND_STORAGE_KEY)
    if (b === 'comfyui' || b === 'runninghub') imageBackend.value = b
    const ck = localStorage.getItem(COMFYUI_CKPT_STORAGE_KEY)
    if (ck) comfyuiCheckpoint.value = ck
  } catch {
    /* ignore */
  }
  await Promise.all([loadComfyWorkflows(), loadSizePresets()])
})

watch(runninghubWorkflowId, (v) => {
  try {
    localStorage.setItem(RUNNINGHUB_WORKFLOW_STORAGE_KEY, v || '')
  } catch {
    /* ignore */
  }
})

watch(imageBackend, (v) => {
  try {
    localStorage.setItem(IMAGE_BACKEND_STORAGE_KEY, v || 'runninghub')
  } catch {
    /* ignore */
  }
})

watch(comfyuiCheckpoint, (v) => {
  try {
    localStorage.setItem(COMFYUI_CKPT_STORAGE_KEY, v || '')
  } catch {
    /* ignore */
  }
})

watch(comfyUiWorkflowFile, (v) => {
  try {
    localStorage.setItem(COMFYUI_WORKFLOW_STORAGE_KEY, v || '')
  } catch {
    /* ignore */
  }
})

watch(comfySizeRatio, (v) => {
  try {
    localStorage.setItem(COMFYUI_SIZE_RATIO_STORAGE_KEY, v || '')
  } catch {
    /* ignore */
  }
})

function handleSettingChange(key, value) {
  settings.value[key] = value
}
</script>

<template>
  <div class="settings-page">
    <div class="page-header">
      <h2>设置</h2>
      <p class="subtitle">自定义应用配置</p>
    </div>
    
    <div class="settings-list">
      <div class="setting-item">
        <div class="setting-info">
          <h4>主题</h4>
          <p>选择界面主题风格</p>
        </div>
        <select
          v-model="settings.theme"
          class="setting-select"
          @change="handleSettingChange('theme', settings.theme)"
        >
          <option value="dark">深色模式</option>
          <option value="light">浅色模式</option>
        </select>
      </div>
      
      <div class="setting-item">
        <div class="setting-info">
          <h4>语言</h4>
          <p>选择界面语言</p>
        </div>
        <select
          v-model="settings.language"
          class="setting-select"
          @change="handleSettingChange('language', settings.language)"
        >
          <option value="zh-CN">简体中文</option>
          <option value="en-US">English</option>
        </select>
      </div>
      
      <div class="setting-item">
        <div class="setting-info">
          <h4>自动保存</h4>
          <p>编辑时自动保存更改</p>
        </div>
        <label class="toggle-switch">
          <input
            type="checkbox"
            v-model="settings.autoSave"
            @change="handleSettingChange('autoSave', settings.autoSave)"
          />
          <span class="toggle-slider"></span>
        </label>
      </div>
      
      <div class="setting-item">
        <div class="setting-info">
          <h4>网格对齐</h4>
          <p>节点自动对齐到网格</p>
        </div>
        <label class="toggle-switch">
          <input
            type="checkbox"
            v-model="settings.snapToGrid"
            @change="handleSettingChange('snapToGrid', settings.snapToGrid)"
          />
          <span class="toggle-slider"></span>
        </label>
      </div>
      
      <div class="setting-item">
        <div class="setting-info">
          <h4>网格大小</h4>
          <p>设置网格单元大小（像素）</p>
        </div>
        <input
          type="number"
          v-model.number="settings.gridSize"
          class="setting-input"
          min="10"
          max="50"
          @change="handleSettingChange('gridSize', settings.gridSize)"
        />
      </div>

      <div class="setting-item setting-item-wide">
        <div class="setting-info setting-info-full">
          <h4>生图方式</h4>
          <p>
            人物立绘与流程图「一键生成背景」共用。选「本地 ComfyUI」时需本机已启动 ComfyUI；工作流来自
            <code>public/comfyui/*.json</code>，正/负向与 Checkpoint 注入位置由同名的
            <code>.mapping.json</code> 或 <code>.txt</code> 描述（见仓库内示例）。
          </p>
        </div>
        <select v-model="imageBackend" class="setting-select setting-input-full">
          <option value="runninghub">RunningHub 云端</option>
          <option value="comfyui">本地 ComfyUI</option>
        </select>
      </div>

      <div class="setting-item setting-item-wide">
        <div class="setting-info setting-info-full">
          <h4>ComfyUI 工作流 JSON</h4>
          <p>
            下拉列表由后端扫描 <code>public/comfyui</code> 生成。新增工作流时放入
            <code>foo.json</code>，并添加 <code>foo.mapping.json</code>（推荐）或
            <code>foo.txt</code> 声明 positive / negative / checkpoint 的节点与 input 键。默认文件可由环境变量
            <code>COMFYUI_WORKFLOW_JSON</code> 指定。
          </p>
          <p v-if="comfyWorkflowsLoadError" class="setting-fetch-warning">
            无法拉取工作流列表：{{ comfyWorkflowsLoadError }}（已使用本地默认选项）
          </p>
        </div>
        <select v-model="comfyUiWorkflowFile" class="setting-select setting-input-full">
          <option v-for="w in comfyWorkflowOptions" :key="w.file" :value="w.file">
            {{ w.label }} — {{ w.file }}
            {{ w.mapping_ok ? '' : '（映射需修复）' }}
          </option>
        </select>
      </div>

      <div class="setting-item setting-item-wide">
        <div class="setting-info setting-info-full">
          <h4>ComfyUI 生图尺寸（ratio）</h4>
          <p>
            仅当所选工作流的 <code>.mapping.json</code> 中包含 <code>size</code> 时生效：
            <code>empty_latent</code> 写入 width/height；
            <code>mx_slider2d</code> 写入 Xi/Xf/Yi/Yf（与 workflow1 节点 19 一致）。默认 ratio 可由
            <code>COMFYUI_SIZE_RATIO</code> 配置。
          </p>
        </div>
        <select v-model="comfySizeRatio" class="setting-select setting-input-full">
          <option v-for="p in sizePresetList" :key="p.ratio" :value="p.ratio">
            {{ p.label }}
          </option>
        </select>
      </div>

      <div class="setting-item setting-item-wide">
        <div class="setting-info setting-info-full">
          <h4>ComfyUI Checkpoint（可选）</h4>
          <p>
            填写 ComfyUI 模型列表中的完整文件名（含 .safetensors）。留空则使用后端默认：优先
            <code>COMFYUI_DEFAULT_CHECKPOINT</code>，否则使用工作流 JSON 中的模型名。
          </p>
        </div>
        <input
          v-model.trim="comfyuiCheckpoint"
          type="text"
          class="setting-input setting-input-full"
          placeholder="例如：waiIllustriousSDXL_v160 (1).safetensors"
          autocomplete="off"
        />
      </div>

      <div class="setting-item setting-item-wide">
        <div class="setting-info setting-info-full">
          <h4>RunningHub 工作流 ID</h4>
          <p>选择「RunningHub 云端」生图时必填；与 RunningHub 控制台中的 workflowId 一致。</p>
        </div>
        <input
          v-model.trim="runninghubWorkflowId"
          type="text"
          class="setting-input setting-input-full"
          placeholder="例如：2037082428853981185"
          autocomplete="off"
        />
      </div>
    </div>
  </div>
</template>

<style scoped>
.settings-page {
  padding: 24px;
  height: calc(100vh - 60px);
  overflow-y: auto;
  background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
}

.page-header {
  margin-bottom: 32px;
}

.page-header h2 {
  margin: 0 0 8px 0;
  font-size: 1.75rem;
  color: #fff;
}

.subtitle {
  margin: 0;
  color: rgba(255, 255, 255, 0.5);
}

.settings-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.setting-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px 24px;
  background: rgba(26, 26, 46, 0.8);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  transition: all 0.3s ease;
}

.setting-item-wide {
  flex-direction: column;
  align-items: stretch;
  gap: 12px;
}

.setting-info-full {
  width: 100%;
}

.setting-input-full {
  width: 100%;
  min-width: 0;
  box-sizing: border-box;
}

.setting-item:hover {
  border-color: rgba(0, 212, 255, 0.3);
}

.setting-info h4 {
  margin: 0 0 4px 0;
  font-size: 1rem;
  color: #fff;
}

.setting-info p {
  margin: 0;
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.5);
}

.setting-select,
.setting-input {
  padding: 10px 16px;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 8px;
  color: #fff;
  font-size: 0.95rem;
  min-width: 150px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.setting-select:hover,
.setting-input:hover {
  border-color: rgba(0, 212, 255, 0.5);
}

.setting-select:focus,
.setting-input:focus {
  outline: none;
  border-color: #00d4ff;
}

.setting-select option {
  background: #1a1a2e;
  color: #fff;
}

.setting-info code {
  font-size: 0.8em;
  padding: 2px 6px;
  border-radius: 4px;
  background: rgba(0, 0, 0, 0.35);
  color: #a8e6ff;
}

.setting-fetch-warning {
  margin-top: 8px;
  font-size: 0.8rem;
  color: #ffb86c;
}

.toggle-switch {
  position: relative;
  display: inline-block;
  width: 52px;
  height: 28px;
}

.toggle-switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.toggle-slider {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 28px;
  transition: all 0.3s ease;
}

.toggle-slider::before {
  position: absolute;
  content: '';
  height: 22px;
  width: 22px;
  left: 3px;
  bottom: 3px;
  background: #fff;
  border-radius: 50%;
  transition: all 0.3s ease;
}

.toggle-switch input:checked + .toggle-slider {
  background: linear-gradient(135deg, #00d4ff 0%, #7b2cbf 100%);
}

.toggle-switch input:checked + .toggle-slider::before {
  transform: translateX(24px);
}
</style>
