<script setup>
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'

const emit = defineEmits(['open-flow'])

/** 故事页草稿版本：v1=内联分段；v2=已选角色条 + 独立大纲文本框 */
const STORY_DRAFT_VERSION = 2

/** 已选角色（与人物卡同步）；提交时按顺序展开为完整角色信息 */
const storyCast = ref([])
/** 故事大纲与自由说明（与角色条分离） */
const outlineDraftText = ref('')

const isGenerating = ref(false)
const currentStage = ref('')
const progress = ref(0)
const strictModel = ref(false)

const editorPlaceholder = `在下方输入故事大纲（上方「已选角色」中的角色会附带完整人物卡信息一并提交）。

可以包含：
• 故事背景设定
• 情节发展脉络
• 关键转折点
• 结局走向

点击「选择角色」将角色加入上方条；姓名、年龄、性别等以人物卡为准，生成时传给大模型。

点击「生成故事」将依次：生成大纲 → 剧本 → 对话剧本。`

const showCharacterSelector = ref(false)
const showCharacterJson = ref(false)
const selectedCharacterJson = ref(null)
const savedCharacters = ref([])

const API_BASE_URL = 'http://localhost:8000'

const STORY_DRAFT_STORAGE_KEY = 'renai_story_draft_v1'

/** 一键生成完成后，后端 SSE 下发的原始对话列表，用于导出流程图 JSON */
const lastDialogueResults = ref(null)
const isExportingFlow = ref(false)

const outlineTextareaRef = ref(null)

const hasComposerInput = computed(
  () => storyCast.value.length > 0 || (outlineDraftText.value && outlineDraftText.value.trim().length > 0)
)

function autoGrowOutlineTextarea() {
  nextTick(() => {
    const el = outlineTextareaRef.value
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.max(120, el.scrollHeight)}px`
  })
}

function buildUserInputForApi() {
  const blocks = storyCast.value.map((c) => formatCharacterInfo(c))
  const body = (outlineDraftText.value || '').trim()
  if (body) blocks.push(body)
  return blocks.join('\n\n')
}

function setStoryFromApiPlainText(text) {
  outlineDraftText.value = text || ''
  nextTick(() => autoGrowOutlineTextarea())
}

/** 从 v1 内联分段草稿迁移到 v2（已选角色 + 单一大纲框） */
function migrateDraftV1SegmentsToV2(storySegments) {
  if (!Array.isArray(storySegments) || storySegments.length === 0) return false
  const seen = new Set()
  const cast = []
  const textParts = []
  for (const seg of storySegments) {
    if (seg.type === 'character' && seg.character && seg.character.id != null) {
      const id = seg.character.id
      if (!seen.has(id)) {
        seen.add(id)
        cast.push({
          id,
          data: { ...(seg.character.data || {}) },
          images: Array.isArray(seg.character.images) ? [...seg.character.images] : []
        })
      }
    } else if (seg.type === 'text') {
      textParts.push(seg.content || '')
    }
  }
  storyCast.value = cast
  outlineDraftText.value = textParts.join('\n\n')
  return true
}

function loadStoryDraft() {
  try {
    const raw = localStorage.getItem(STORY_DRAFT_STORAGE_KEY)
    if (!raw) return
    const s = JSON.parse(raw)
    if (s?.v === STORY_DRAFT_VERSION) {
      storyCast.value = Array.isArray(s.storyCast) ? s.storyCast : []
      outlineDraftText.value = typeof s.outlineText === 'string' ? s.outlineText : ''
    } else if (s?.v === 1) {
      if (Array.isArray(s.storySegments) && s.storySegments.length > 0) {
        migrateDraftV1SegmentsToV2(s.storySegments)
      } else {
        storyCast.value = []
        outlineDraftText.value = ''
      }
    } else {
      return
    }
    if (typeof s.strictModel === 'boolean') strictModel.value = s.strictModel
    if (Object.prototype.hasOwnProperty.call(s, 'lastDialogueResults')) {
      lastDialogueResults.value = s.lastDialogueResults
    }
    nextTick(() => autoGrowOutlineTextarea())
  } catch (e) {
    console.warn('恢复故事草稿失败:', e)
  }
}

let storyDraftPersistTimer = null
function persistStoryDraft() {
  if (storyDraftPersistTimer) clearTimeout(storyDraftPersistTimer)
  storyDraftPersistTimer = setTimeout(() => {
    storyDraftPersistTimer = null
    try {
      localStorage.setItem(
        STORY_DRAFT_STORAGE_KEY,
        JSON.stringify({
          v: STORY_DRAFT_VERSION,
          storyCast: JSON.parse(JSON.stringify(storyCast.value)),
          outlineText: outlineDraftText.value,
          strictModel: strictModel.value,
          lastDialogueResults: lastDialogueResults.value
        })
      )
    } catch (e) {
      console.warn('无法保存故事草稿（可能超出浏览器存储配额）', e)
    }
  }, 400)
}

watch([storyCast, outlineDraftText, strictModel, lastDialogueResults], () => persistStoryDraft(), { deep: true })

const CHARACTERS_STORAGE_KEY = 'characters_data'

function loadSavedCharacters() {
  try {
    const saved = localStorage.getItem(CHARACTERS_STORAGE_KEY)
    if (saved) {
      const parsed = JSON.parse(saved)
      savedCharacters.value = Array.isArray(parsed) ? parsed : []
    } else {
      savedCharacters.value = []
    }
  } catch (error) {
    console.error('加载角色卡失败:', error)
    savedCharacters.value = []
  }
}

/** 将已选角色与当前人物卡列表对齐（编辑/删除人物卡后同步） */
function syncStoryCharacterRefsFromCards() {
  const list = savedCharacters.value
  const byId = new Map(list.map((c) => [c.id, c]))
  storyCast.value = storyCast.value.map((slot) => {
    const live = byId.get(slot.id)
    if (!live) {
      return {
        ...slot,
        _missingCard: true
      }
    }
    return {
      id: live.id,
      data: { ...(live.data || {}) },
      images: live.images ? [...live.images] : []
    }
  })
}

function refreshCharactersFromStorage() {
  loadSavedCharacters()
  syncStoryCharacterRefsFromCards()
}

function onRenaiCharactersStorage(e) {
  const d = e?.detail?.characters
  if (Array.isArray(d)) {
    savedCharacters.value = d
    syncStoryCharacterRefsFromCards()
    return
  }
  refreshCharactersFromStorage()
}

function onWindowStorage(e) {
  if (e.key === CHARACTERS_STORAGE_KEY) {
    refreshCharactersFromStorage()
  }
}

onMounted(() => {
  refreshCharactersFromStorage()
  loadStoryDraft()
  syncStoryCharacterRefsFromCards()
  window.addEventListener('renai-characters-storage', onRenaiCharactersStorage)
  window.addEventListener('storage', onWindowStorage)
  nextTick(() => autoGrowOutlineTextarea())
})

onUnmounted(() => {
  window.removeEventListener('renai-characters-storage', onRenaiCharactersStorage)
  window.removeEventListener('storage', onWindowStorage)
})

function toggleStrictModel() {
  strictModel.value = !strictModel.value
}

function openCharacterSelector() {
  refreshCharactersFromStorage()
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

function addCharacterToStoryCast(character) {
  if (storyCast.value.some((c) => c.id === character.id)) {
    closeCharacterSelector()
    return
  }
  storyCast.value.push({
    id: character.id,
    data: { ...(character.data || {}) },
    images: character.images ? [...character.images] : []
  })
  closeCharacterSelector()
}

function removeCharacterFromCast(index) {
  if (index < 0 || index >= storyCast.value.length) return
  storyCast.value.splice(index, 1)
}

function chipDisplayName(character) {
  const name = character?.data?.name?.trim() || '未命名'
  if (character?._missingCard) return `${name}（人物卡已删除）`
  return name
}

function formatCharacterInfo(character) {
  const data = character.data || {}
  const name = (data.name || '未命名').trim() || '未命名'
  let info = `【角色】\n`
  info += `角色姓名（全文须与此逐字一致，禁止翻译、改写或替换称呼）：${name}\n`
  if (character._missingCard) {
    info += '（注意：该人物卡已从本地删除，以下为加入故事时的快照）\n'
  }
  if (data.age !== undefined && data.age !== null && String(data.age).trim()) {
    info += `年龄：${String(data.age).trim()}\n`
  }
  if (data.gender !== undefined && data.gender !== null && String(data.gender).trim()) {
    info += `性别：${String(data.gender).trim()}\n`
  }
  if (data.appearance) info += `外貌描述：${data.appearance}\n`
  if (data.personality) info += `性格设定：${data.personality}\n`
  if (data.background) info += `背景故事：${data.background}\n`
  if (data.dialogue_examples) info += `对话示例：${data.dialogue_examples}\n`
  if (data.other_settings) info += `其他设定：${data.other_settings}\n`
  return info.trimEnd()
}

async function generateStory() {
  const userInput = buildUserInputForApi()
  if (!userInput.trim()) {
    alert('请先选择至少一名角色，或输入故事大纲（也可两者都填）')
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
        user_input: userInput,
        strict_model: strictModel.value,
        story_cast: storyCast.value
          .map((c) => ({
            character_name: String(c.data?.name || '').trim()
          }))
          .filter((x) => x.character_name)
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
          setStoryFromApiPlainText(data.final_result)
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
      
      <div class="editor-container editor-composite-wrap">
        <div
          class="story-editor story-composite"
          :class="{ 'is-disabled': isGenerating }"
        >
          <p
            v-if="!hasComposerInput && !isGenerating"
            class="story-empty-hint"
          >
            {{ editorPlaceholder }}
          </p>
          <div class="story-cast-panel">
            <div class="story-cast-head">
              <span class="story-cast-title">已选角色</span>
              <span class="story-cast-sub">
                点击下方「选择角色」加入；生成时会按顺序附带姓名、年龄、性别、外貌、性格、背景、对话示例、其他设定
              </span>
            </div>
            <div class="story-cast-chips">
              <template v-if="storyCast.length">
                <span
                  v-for="(c, idx) in storyCast"
                  :key="c.id"
                  class="char-chip-inline"
                  :class="{ 'char-chip-inline--missing': c._missingCard }"
                  :title="'提交时附带完整人物卡：' + chipDisplayName(c)"
                >
                  <span class="char-chip-name">{{ chipDisplayName(c) }}</span>
                  <button
                    type="button"
                    class="char-chip-remove"
                    :disabled="isGenerating"
                    title="从本故事移除该角色"
                    @click.stop="removeCharacterFromCast(idx)"
                  >
                    ×
                  </button>
                </span>
              </template>
              <span v-else class="story-cast-empty">暂无已选角色</span>
            </div>
          </div>
          <textarea
            ref="outlineTextareaRef"
            v-model="outlineDraftText"
            class="story-outline-main"
            :disabled="isGenerating"
            rows="6"
            placeholder="在此编写故事大纲与情节说明（与上方角色条分开；角色名以人物卡为准，模型须逐字使用）"
            @input="autoGrowOutlineTextarea"
          />
        </div>
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
              <button
                class="selector-action-btn insert-btn"
                @click.stop="addCharacterToStoryCast(char)"
                title="加入上方「已选角色」；生成时按顺序提交完整人物卡"
              >
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
  position: relative;
  min-height: 0;
}

.editor-composite-wrap:focus-within .story-composite {
  border-color: rgba(0, 212, 255, 0.3);
  box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1);
}

/** 空状态说明：放在文档流内，避免与半透明编辑区叠字 */
.story-empty-hint {
  margin: 0 0 14px 0;
  padding: 0 2px 12px 2px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  white-space: pre-wrap;
  font-size: 0.88rem;
  line-height: 1.75;
  color: rgba(255, 255, 255, 0.38);
  font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
}

.story-editor.story-composite {
  width: 100%;
  height: 100%;
  min-height: 200px;
  max-height: 100%;
  overflow-y: auto;
  padding: 20px 22px;
  box-sizing: border-box;
  background: rgba(26, 26, 46, 0.6);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  color: rgba(255, 255, 255, 0.9);
  font-size: 0.95rem;
  line-height: 1.8;
  font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
  transition: border-color 0.3s ease, box-shadow 0.3s ease;
  display: block;
  white-space: normal;
  position: relative;
  /* 避免底层与浮层叠字：空状态提示在内部文档流，面板略加不透明度 */
  background: rgba(22, 22, 40, 0.92);
}

.story-editor.story-composite.is-disabled {
  opacity: 0.7;
  pointer-events: none;
}

.story-cast-panel {
  margin-bottom: 16px;
  padding-bottom: 14px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.story-cast-head {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 10px;
}

.story-cast-title {
  font-size: 0.95rem;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.88);
}

.story-cast-sub {
  font-size: 0.78rem;
  line-height: 1.45;
  color: rgba(255, 255, 255, 0.42);
}

.story-cast-chips {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  min-height: 36px;
}

.story-cast-empty {
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.35);
}

.story-outline-main {
  box-sizing: border-box;
  display: block;
  width: 100%;
  min-height: 140px;
  margin: 0;
  padding: 12px 14px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 10px;
  background: rgba(10, 10, 22, 0.45);
  color: rgba(255, 255, 255, 0.92);
  font-size: 0.95rem;
  line-height: 1.75;
  font-family: inherit;
  resize: vertical;
}

.story-outline-main::placeholder {
  color: rgba(255, 255, 255, 0.28);
}

.story-outline-main:focus {
  outline: none;
  border-color: rgba(0, 212, 255, 0.35);
  box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.08);
}

.story-outline-main:disabled {
  cursor: not-allowed;
  opacity: 0.75;
}

.char-chip-inline {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  max-width: 100%;
  padding: 2px 4px 2px 10px;
  border-radius: 8px;
  background: linear-gradient(135deg, rgba(123, 44, 191, 0.35) 0%, rgba(0, 212, 255, 0.2) 100%);
  border: 1px solid rgba(0, 212, 255, 0.45);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
  font-size: 0.82rem;
  line-height: 1.5;
  vertical-align: top;
  margin: 2px 6px 2px 0;
}

.char-chip-inline--missing {
  border-color: rgba(255, 140, 90, 0.55);
  background: linear-gradient(135deg, rgba(180, 70, 50, 0.28) 0%, rgba(90, 40, 40, 0.22) 100%);
}

.char-chip-name {
  color: #e0f7ff;
  font-weight: 600;
  max-width: 160px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.char-chip-remove {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  padding: 0;
  border: none;
  border-radius: 6px;
  background: rgba(0, 0, 0, 0.25);
  color: rgba(255, 255, 255, 0.65);
  font-size: 1rem;
  line-height: 1;
  cursor: pointer;
  flex-shrink: 0;
}

.char-chip-remove:hover:not(:disabled) {
  background: rgba(255, 59, 48, 0.35);
  color: #fff;
}

.char-chip-remove:disabled {
  opacity: 0.5;
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
