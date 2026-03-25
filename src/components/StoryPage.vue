<script setup>
import { ref, onMounted } from 'vue'

const emit = defineEmits(['open-flow'])

const storyContent = ref('')
const isGenerating = ref(false)
const currentStage = ref('')
const progress = ref(0)
const strictModel = ref(false)

const editorPlaceholder = `在这里输入故事大纲...

可以包含：
• 故事背景设定
• 主要角色介绍（可点击"选择角色"导入角色卡信息）
• 情节发展脉络
• 关键转折点
• 结局走向

点击"生成故事"按钮，系统将自动：
1. 生成完整大纲
2. 生成详细剧本
3. 生成对话剧本

最终显示对话剧本结果。`

const showCharacterSelector = ref(false)
const showCharacterJson = ref(false)
const selectedCharacterJson = ref(null)
const savedCharacters = ref([])

const API_BASE_URL = 'http://localhost:8000'

/** 一键生成完成后，后端 SSE 下发的原始对话列表，用于导出流程图 JSON */
const lastDialogueResults = ref(null)
const isExportingFlow = ref(false)

onMounted(() => {
  loadSavedCharacters()
})

function loadSavedCharacters() {
  try {
    const saved = localStorage.getItem('characters_data')
    if (saved) {
      savedCharacters.value = JSON.parse(saved)
    }
  } catch (error) {
    console.error('加载角色卡失败:', error)
  }
}

function toggleStrictModel() {
  strictModel.value = !strictModel.value
}

function openCharacterSelector() {
  showCharacterSelector.value = true
}

function closeCharacterSelector() {
  showCharacterSelector.value = false
}

function showCharacterJsonData(character) {
  selectedCharacterJson.value = JSON.stringify(character.data, null, 2)
  showCharacterJson.value = true
}

function closeCharacterJson() {
  showCharacterJson.value = false
  selectedCharacterJson.value = null
}

function insertCharacterInfo(character) {
  const charInfo = formatCharacterInfo(character)
  const textarea = document.querySelector('.story-editor')
  if (textarea) {
    const start = textarea.selectionStart
    const end = textarea.selectionEnd
    const text = storyContent.value
    storyContent.value = text.substring(0, start) + charInfo + text.substring(end)
    textarea.focus()
  }
  closeCharacterSelector()
}

function formatCharacterInfo(character) {
  const data = character.data
  let info = `\n【角色：${data.name || '未命名'}】\n`
  
  if (data.gender) info += `性别：${data.gender}\n`
  if (data.age) info += `年龄：${data.age}\n`
  if (data.appearance) info += `外貌：${data.appearance}\n`
  if (data.personality) info += `性格：${data.personality}\n`
  if (data.background) info += `背景：${data.background}\n`
  
  info += `\n`
  return info
}

async function generateStory() {
  if (!storyContent.value.trim()) {
    alert('请先输入故事大纲')
    return
  }
  
  isGenerating.value = true
  progress.value = 0
  currentStage.value = '准备开始...'
  lastDialogueResults.value = null

  try {
    const response = await fetch(`${API_BASE_URL}/api/story/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        user_input: storyContent.value,
        strict_model: strictModel.value
      })
    })
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let sseBuffer = ''

    while (true) {
      const { done, value } = await reader.read()
      sseBuffer += decoder.decode(value ?? new Uint8Array(), { stream: !done })

      const lines = sseBuffer.split('\n')
      sseBuffer = done ? '' : (lines.pop() ?? '')

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const payload = line.slice(6).trim()
        if (!payload) continue

        let data
        try {
          data = JSON.parse(payload)
        } catch {
          console.warn('SSE 行 JSON 解析失败（可能分块截断，已缓冲后续拼接）:', payload)
          continue
        }

        if (data.stage === 'error') {
          throw new Error(data.message || '生成失败')
        }
        if (data.stage) {
          currentStage.value = data.message
          progress.value = data.progress
        }
        if (data.dialogue_results) {
          lastDialogueResults.value = data.dialogue_results
        }
        if (data.stage === 'complete' && data.final_result != null) {
          storyContent.value = data.final_result
        }
      }

      if (done) break
    }
    
  } catch (error) {
    console.error('生成故事失败:', error)
    alert('生成故事失败: ' + error.message)
  } finally {
    isGenerating.value = false
    progress.value = 0
    currentStage.value = ''
  }
}

async function exportStructuredAndOpenFlow() {
  const payload = lastDialogueResults.value
  if (!payload || !Array.isArray(payload) || payload.length === 0) {
    alert('请先生成故事，待对话阶段完成后再导出流程图。')
    return
  }
  isExportingFlow.value = true
  try {
    const response = await fetch(`${API_BASE_URL}/api/story/export-structured`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dialogue_results: payload })
    })
    if (!response.ok) {
      const errText = await response.text()
      throw new Error(errText || `HTTP ${response.status}`)
    }
    const result = await response.json()
    const url = result.public_url
    if (!url) {
      throw new Error('后端未返回 public_url')
    }
    emit('open-flow', url)
  } catch (error) {
    console.error('导出流程图 JSON 失败:', error)
    alert('导出失败: ' + (error.message || String(error)))
  } finally {
    isExportingFlow.value = false
  }
}

function getCharacterInitial(name) {
  return name ? name.charAt(0).toUpperCase() : '?'
}
</script>

<template>
  <div class="story-page">
    <div class="story-header">
      <h2>故事生成</h2>
      <p class="subtitle">输入大纲，一键生成完整对话剧本</p>
    </div>
    
    <div class="main-content">
      <div class="toolbar">
        <button 
          :class="['action-btn', 'strict-btn', { active: strictModel }]"
          @click="toggleStrictModel"
          :title="strictModel ? '严格模式已开启' : '严格模式已关闭'"
        >
          <span>🎯</span> 严格模式
        </button>
        
        <button class="action-btn" @click="openCharacterSelector">
          <span>👤</span> 选择角色
        </button>
        
        <button 
          class="action-btn generate-btn" 
          @click="generateStory"
          :disabled="isGenerating"
        >
          <span>{{ isGenerating ? '⏳' : '✨' }}</span> 
          {{ isGenerating ? '生成中...' : '生成故事' }}
        </button>

        <button
          type="button"
          class="action-btn flow-export-btn"
          :disabled="isGenerating || isExportingFlow || !lastDialogueResults"
          :title="!lastDialogueResults ? '请先生成故事' : '保存为流程图 JSON 并打开流程页'"
          @click="exportStructuredAndOpenFlow"
        >
          <span>{{ isExportingFlow ? '⏳' : '📊' }}</span>
          {{ isExportingFlow ? '导出中...' : '导出并打开流程图' }}
        </button>
      </div>
      
      <div v-if="isGenerating" class="progress-bar">
        <div class="progress-fill" :style="{ width: progress + '%' }"></div>
        <span class="progress-text">{{ currentStage }} ({{ progress }}%)</span>
      </div>
      
      <div class="editor-container">
        <textarea
          v-model="storyContent"
          class="story-editor"
          :placeholder="editorPlaceholder"
          :disabled="isGenerating"
        ></textarea>
      </div>
    </div>
    
    <div v-if="showCharacterSelector" class="character-selector-overlay" @click="closeCharacterSelector">
      <div class="character-selector" @click.stop>
        <div class="selector-header">
          <h3>选择角色</h3>
          <button class="close-btn" @click="closeCharacterSelector">×</button>
        </div>
        <div class="selector-grid">
          <div
            v-for="char in savedCharacters"
            :key="char.id"
            class="selector-item"
          >
            <div class="selector-avatar">
              <img v-if="char.images && char.images[0]" :src="char.images[0].url" :alt="char.data.name" />
              <span v-else class="avatar-placeholder">{{ getCharacterInitial(char.data.name) }}</span>
            </div>
            <div class="selector-info">
              <span class="selector-name">{{ char.data.name || '未命名' }}</span>
              <span class="selector-meta">{{ char.data.gender || '' }} · {{ char.data.age || '' }}</span>
              <div class="selector-details" v-if="char.data.appearance">
                <span class="detail-label">外貌：</span>{{ char.data.appearance }}
              </div>
              <div class="selector-details" v-if="char.data.personality">
                <span class="detail-label">性格：</span>{{ char.data.personality }}
              </div>
              <div class="selector-details" v-if="char.data.background">
                <span class="detail-label">背景：</span>{{ char.data.background }}
              </div>
            </div>
            <div class="selector-actions">
              <button class="selector-action-btn insert-btn" @click.stop="insertCharacterInfo(char)" title="插入角色信息">
                <span>📝</span>
              </button>
              <button class="selector-action-btn" @click.stop="showCharacterJsonData(char)" title="查看JSON数据">
                <span>📋</span>
              </button>
            </div>
          </div>
          
          <div v-if="savedCharacters.length === 0" class="empty-selector">
            <p>暂无可用角色</p>
            <p class="hint">请先在角色管理页面创建或导入角色卡</p>
          </div>
        </div>
      </div>
    </div>
    
    <div v-if="showCharacterJson" class="character-json-overlay" @click="closeCharacterJson">
      <div class="character-json-modal" @click.stop>
        <div class="json-header">
          <h3>角色JSON数据</h3>
          <button class="close-btn" @click="closeCharacterJson">×</button>
        </div>
        <div class="json-content">
          <pre>{{ selectedCharacterJson }}</pre>
        </div>
        <div class="json-footer">
          <button class="action-btn" @click="closeCharacterJson">关闭</button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.story-page {
  height: calc(100vh - 60px);
  display: flex;
  flex-direction: column;
  background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
  overflow: hidden;
}

.story-header {
  padding: 20px 24px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  flex-shrink: 0;
}

.story-header h2 {
  margin: 0 0 4px 0;
  font-size: 1.5rem;
  color: #fff;
}

.subtitle {
  margin: 0;
  color: rgba(255, 255, 255, 0.5);
  font-size: 0.9rem;
}

.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  padding: 16px 24px;
}

.toolbar {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
  flex-shrink: 0;
}

.action-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 20px;
  background: rgba(0, 212, 255, 0.15);
  border: 1px solid rgba(0, 212, 255, 0.3);
  border-radius: 8px;
  color: #00d4ff;
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.3s ease;
}

.action-btn:hover {
  background: rgba(0, 212, 255, 0.25);
  border-color: #00d4ff;
  transform: translateY(-2px);
}

.action-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.strict-btn {
  background: rgba(123, 44, 191, 0.15);
  border-color: rgba(123, 44, 191, 0.3);
  color: #7b2cbf;
}

.strict-btn:hover {
  background: rgba(123, 44, 191, 0.25);
  border-color: #7b2cbf;
}

.strict-btn.active {
  background: rgba(123, 44, 191, 0.35);
  border-color: #7b2cbf;
  color: #fff;
  box-shadow: 0 0 15px rgba(123, 44, 191, 0.4);
}

.generate-btn {
  background: linear-gradient(135deg, rgba(46, 204, 113, 0.2) 0%, rgba(0, 212, 255, 0.2) 100%);
  border-color: rgba(46, 204, 113, 0.4);
  color: #2ecc71;
  font-weight: 600;
}

.generate-btn:hover:not(:disabled) {
  background: linear-gradient(135deg, rgba(46, 204, 113, 0.3) 0%, rgba(0, 212, 255, 0.3) 100%);
  border-color: #2ecc71;
  box-shadow: 0 4px 15px rgba(46, 204, 113, 0.3);
}

.flow-export-btn {
  background: rgba(241, 196, 15, 0.12);
  border-color: rgba(241, 196, 15, 0.35);
  color: #f1c40f;
}

.flow-export-btn:hover:not(:disabled) {
  background: rgba(241, 196, 15, 0.22);
  border-color: #f1c40f;
  box-shadow: 0 4px 12px rgba(241, 196, 15, 0.25);
}

.progress-bar {
  height: 40px;
  background: rgba(26, 26, 46, 0.6);
  border-radius: 8px;
  margin-bottom: 16px;
  position: relative;
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, 0.1);
  flex-shrink: 0;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #00d4ff 0%, #7b2cbf 100%);
  transition: width 0.3s ease;
  position: absolute;
  top: 0;
  left: 0;
}

.progress-text {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: #fff;
  font-size: 0.85rem;
  font-weight: 500;
  white-space: nowrap;
  z-index: 1;
}

.editor-container {
  flex: 1;
  overflow: hidden;
}

.story-editor {
  width: 100%;
  height: 100%;
  padding: 24px;
  background: rgba(26, 26, 46, 0.6);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  color: rgba(255, 255, 255, 0.9);
  font-size: 0.95rem;
  line-height: 1.8;
  resize: none;
  font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
  transition: all 0.3s ease;
}

.story-editor:focus {
  outline: none;
  border-color: rgba(0, 212, 255, 0.3);
  box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1);
}

.story-editor::placeholder {
  color: rgba(255, 255, 255, 0.3);
}

.story-editor:disabled {
  opacity: 0.7;
  cursor: not-allowed;
}

.character-selector-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  backdrop-filter: blur(4px);
}

.character-selector {
  background: rgba(26, 26, 46, 0.95);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  padding: 24px;
  max-width: 700px;
  width: 90%;
  max-height: 80vh;
  overflow-y: auto;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
}

.selector-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 16px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.selector-header h3 {
  margin: 0;
  font-size: 1.2rem;
  color: rgba(255, 255, 255, 0.9);
}

.close-btn {
  background: transparent;
  border: none;
  color: rgba(255, 255, 255, 0.6);
  font-size: 1.5rem;
  cursor: pointer;
  padding: 4px 8px;
  border-radius: 4px;
  transition: all 0.2s;
}

.close-btn:hover {
  background: rgba(255, 59, 48, 0.2);
  color: #ff3b30;
}

.selector-grid {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.selector-item {
  display: flex;
  align-items: flex-start;
  gap: 16px;
  padding: 20px;
  background: rgba(15, 15, 26, 0.8);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.selector-item:hover {
  background: rgba(0, 212, 255, 0.1);
  border-color: rgba(0, 212, 255, 0.3);
  transform: translateY(-2px);
}

.selector-avatar {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  overflow: hidden;
  background: linear-gradient(135deg, #00d4ff 0%, #7b2cbf 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  font-size: 2rem;
  color: #fff;
}

.selector-avatar img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.selector-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 0;
}

.selector-name {
  font-size: 1.1rem;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.9);
}

.selector-meta {
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.5);
}

.selector-details {
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.7);
  line-height: 1.5;
}

.detail-label {
  color: #00d4ff;
  font-weight: 500;
}

.selector-actions {
  display: flex;
  flex-direction: column;
  gap: 8px;
  flex-shrink: 0;
}

.selector-action-btn {
  width: 40px;
  height: 40px;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(26, 26, 46, 0.6);
  color: rgba(255, 255, 255, 0.7);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s ease;
  font-size: 1.2rem;
}

.selector-action-btn:hover {
  background: rgba(0, 212, 255, 0.2);
  border-color: rgba(0, 212, 255, 0.3);
  color: #00d4ff;
  transform: scale(1.05);
}

.insert-btn {
  background: rgba(46, 204, 113, 0.2);
  border-color: rgba(46, 204, 113, 0.3);
}

.insert-btn:hover {
  background: rgba(46, 204, 113, 0.3);
  border-color: #2ecc71;
  color: #2ecc71;
}

.character-json-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1100;
  backdrop-filter: blur(4px);
}

.character-json-modal {
  background: rgba(26, 26, 46, 0.98);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  padding: 24px;
  max-width: 800px;
  width: 90%;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
}

.json-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 16px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.json-header h3 {
  margin: 0;
  font-size: 1.2rem;
  color: rgba(255, 255, 255, 0.9);
}

.json-content {
  flex: 1;
  overflow: auto;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 8px;
  padding: 16px;
}

.json-content pre {
  margin: 0;
  font-family: 'Courier New', monospace;
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.9);
  white-space: pre-wrap;
  word-wrap: break-word;
  line-height: 1.6;
}

.json-footer {
  margin-top: 20px;
  padding-top: 16px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  justify-content: flex-end;
}

.empty-selector {
  text-align: center;
  padding: 60px 20px;
  color: rgba(255, 255, 255, 0.5);
}

.empty-selector p {
  margin: 0 0 8px 0;
  font-size: 1rem;
}

.empty-selector .hint {
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.3);
}
</style>
