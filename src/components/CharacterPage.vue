<script setup>
import { ref, computed, watch, onMounted, nextTick } from 'vue'

const defaultCharacterData = {
  name: '',
  age: '',
  gender: '',
  appearance: '',
  personality: '',
  /** 立绘生图时附加在每条正向 prompt 末尾（如画师串、风格词），不经 LLM */
  image_prompt_extra: '',
  background: '',
  dialogue_examples: '',
  other_settings: '',
  metadata: {}
}

const characters = ref([])
const selectedCharacterId = ref(null)
const selectedImageIndex = ref(0)
const viewMode = ref('chat')

onMounted(() => {
  _suppressCharactersWatch = true
  loadCharactersFromStorage()
  nextTick(() => {
    _suppressCharactersWatch = false
  })
})

const CHARACTERS_STORAGE_KEY = 'characters_data'

function loadCharactersFromStorage() {
  try {
    const saved = localStorage.getItem(CHARACTERS_STORAGE_KEY)
    if (saved) {
      const parsed = JSON.parse(saved)
      if (Array.isArray(parsed)) {
        characters.value = parsed
        if (characters.value.length > 0) {
          selectedCharacterId.value = characters.value[0].id
        }
      }
    }
  } catch (error) {
    console.error('加载角色数据失败:', error)
  }
}

/** 用于写入 localStorage：去掉超长 data URL，避免配额爆掉导致刷新后角色卡消失 */
function buildCharactersPayloadForDisk(list, stripDataUrls) {
  return list.map((c) => ({
    id: c.id,
    data: { ...(c.data || {}) },
    images: (c.images || []).map((img) => ({
      id: img.id,
      name: img.name || '',
      url:
        stripDataUrls &&
        typeof img.url === 'string' &&
        img.url.startsWith('data:') &&
        img.url.length > 8000
          ? ''
          : img.url || ''
    }))
  }))
}

/** 曾触发过配额限制后，跳过大图 JSON 直写，避免每次保存都失败一遍 */
let _useLiteLocalStoragePersist = false
let _quotaHintShown = false

function persistCharactersToLocalStorage() {
  const tryWrite = (list) => {
    const str = JSON.stringify(list)
    localStorage.setItem(CHARACTERS_STORAGE_KEY, str)
    return true
  }

  if (_useLiteLocalStoragePersist) {
    try {
      const lite = buildCharactersPayloadForDisk(characters.value, true)
      tryWrite(lite)
      return { ok: true, stripped: true }
    } catch (e2) {
      console.error('精简图片后仍无法保存:', e2)
    }
    try {
      const minimal = characters.value.map((c) => ({
        id: c.id,
        data: { ...(c.data || {}) },
        images: []
      }))
      tryWrite(minimal)
      return { ok: true, stripped: true, imagesDropped: true }
    } catch (e3) {
      console.error('保存角色数据失败（已尝试去掉全部图片）:', e3)
      return { ok: false, stripped: true, error: e3 }
    }
  }

  try {
    tryWrite(characters.value)
    return { ok: true, stripped: false }
  } catch (e) {
    const isQuota =
      e?.name === 'QuotaExceededError' ||
      e?.code === 22 ||
      (typeof e?.message === 'string' && e.message.toLowerCase().includes('quota'))
    if (!isQuota) {
      console.error('保存角色数据失败:', e)
      return { ok: false, stripped: false, error: e }
    }
    _useLiteLocalStoragePersist = true
    try {
      const lite = buildCharactersPayloadForDisk(characters.value, true)
      tryWrite(lite)
      return { ok: true, stripped: true }
    } catch (e2) {
      console.error('精简图片后仍无法保存:', e2)
    }
    try {
      const minimal = characters.value.map((c) => ({
        id: c.id,
        data: { ...(c.data || {}) },
        images: []
      }))
      tryWrite(minimal)
      return { ok: true, stripped: true, imagesDropped: true }
    } catch (e3) {
      console.error('保存角色数据失败（已尝试去掉全部图片）:', e3)
      return { ok: false, stripped: true, error: e3 }
    }
  }
}

function safeSnapshotForBroadcast() {
  try {
    return JSON.parse(JSON.stringify(characters.value))
  } catch (e) {
    console.warn('角色卡深拷贝失败，使用精简字段同步故事页', e)
    return characters.value.map((c) => ({
      id: c.id,
      data: { ...defaultCharacterData, ...(c.data || {}) },
      images: Array.isArray(c.images)
        ? c.images.map((img) => ({
            id: img.id,
            name: img.name || '',
            url: typeof img.url === 'string' ? img.url : ''
          }))
        : []
    }))
  }
}

function saveCharactersToStorage() {
  const persist = persistCharactersToLocalStorage()

  if (persist.ok && persist.stripped && !_quotaHintShown) {
    _quotaHintShown = true
    const msg =
      persist.imagesDropped === true
        ? '浏览器存储空间不足：已保存角色文字信息，但本地未保存立绘图片；请用「备份全部」导出 JSON 以防丢失。'
        : '浏览器存储空间不足：已保存角色卡，但过大的图片未写入本地；刷新后可能无头像，请使用「备份全部」保存完整数据。'
    if (typeof window !== 'undefined') {
      window.setTimeout(() => alert(msg), 0)
    }
  } else if (!persist.ok && !_quotaHintShown) {
    _quotaHintShown = true
    if (typeof window !== 'undefined') {
      window.setTimeout(
        () =>
          alert(
            '角色卡无法写入本地存储，刷新后可能丢失本次导入；请立即使用「备份全部」导出 JSON，或清理站点数据后重试。'
          ),
        0
      )
    }
  }

  const broadcast = safeSnapshotForBroadcast()
  window.dispatchEvent(
    new CustomEvent('renai-characters-storage', {
      detail: {
        characters: broadcast,
        persistOk: persist.ok
      }
    })
  )
}

/** 初次从磁盘载入时不要触发 watch 写回，避免与 Story 等组件挂载顺序竞争 */
let _suppressCharactersWatch = false

watch(characters, () => {
  if (_suppressCharactersWatch) return
  saveCharactersToStorage()
}, { deep: true })

const selectedCharacter = computed(() => {
  return characters.value.find(c => c.id === selectedCharacterId.value) || null
})

const currentImages = computed(() => {
  return selectedCharacter.value?.images || []
})

const currentMainImage = computed(() => {
  return currentImages.value[selectedImageIndex.value] || null
})

const firstImage = computed(() => {
  return currentImages.value[0] || null
})

const characterForm = ref({ ...defaultCharacterData })

const chatMessages = ref([])
const chatInput = ref('')
const chatMessagesRef = ref(null)
const chatSending = ref(false)

const RUNNINGHUB_WORKFLOW_STORAGE_KEY = 'renai_runninghub_workflow_id'
const IMAGE_BACKEND_STORAGE_KEY = 'renai_image_backend'
const COMFYUI_CKPT_STORAGE_KEY = 'renai_comfyui_checkpoint'
const COMFYUI_WORKFLOW_STORAGE_KEY = 'renai_comfyui_workflow'
const COMFYUI_SIZE_RATIO_STORAGE_KEY = 'renai_comfyui_size_ratio'
const API_BASE_URL = 'http://localhost:8000'
const runninghubPicLoading = ref(false)
const removeStandBgLoading = ref(false)

/** 立绘：default=内置表情全集；custom=metadata.stand_custom_items */
const standExpressionMode = ref('default')
const standCustomRows = ref([{ id: '', description: '' }])

function syncStandUiFromCharacter(char) {
  const c = char ?? selectedCharacter.value
  if (!c?.data?.metadata) {
    standExpressionMode.value = 'default'
    standCustomRows.value = [{ id: '', description: '' }]
    return
  }
  const m = c.data.metadata
  standExpressionMode.value = m.stand_expression_mode === 'custom' ? 'custom' : 'default'
  const items = m.stand_custom_items
  if (Array.isArray(items) && items.length) {
    standCustomRows.value = items.map((it) => ({
      id: String(it.id ?? ''),
      description: String(it.description ?? '')
    }))
  } else {
    standCustomRows.value = [{ id: '', description: '' }]
  }
}

function addStandCustomRow() {
  standCustomRows.value.push({ id: '', description: '' })
}

function removeStandCustomRow(index) {
  if (standCustomRows.value.length <= 1) return
  standCustomRows.value.splice(index, 1)
}

watch(selectedCharacter, async (newChar) => {
  if (newChar) {
    characterForm.value = {
      name: newChar.data.name || '',
      age: newChar.data.age || '',
      gender: newChar.data.gender || '',
      appearance: newChar.data.appearance || '',
      personality: newChar.data.personality || '',
      image_prompt_extra: newChar.data.image_prompt_extra || '',
      background: newChar.data.background || '',
      dialogue_examples: newChar.data.dialogue_examples || '',
      other_settings: newChar.data.other_settings || '',
      metadata: newChar.data.metadata || {}
    }
    selectedImageIndex.value = 0
    syncStandUiFromCharacter(newChar)
    await loadCharacterChatHistory()
  } else {
    characterForm.value = { ...defaultCharacterData }
    chatMessages.value = []
    standExpressionMode.value = 'default'
    standCustomRows.value = [{ id: '', description: '' }]
  }
})

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).substr(2)
}

function createNewCharacter() {
  const newChar = {
    id: generateId(),
    data: { ...defaultCharacterData },
    images: []
  }
  characters.value.push(newChar)
  selectedCharacterId.value = newChar.id
  viewMode.value = 'edit'
  saveCharactersToStorage()
}

function selectCharacter(id) {
  selectedCharacterId.value = id
  viewMode.value = 'chat'
}

function deleteCharacter(id) {
  const index = characters.value.findIndex(c => c.id === id)
  if (index !== -1) {
    characters.value.splice(index, 1)
    if (selectedCharacterId.value === id) {
      selectedCharacterId.value = characters.value.length > 0 ? characters.value[0].id : null
    }
    saveCharactersToStorage()
  }
}

function openEditMode() {
  viewMode.value = 'edit'
}

function closeEditMode() {
  viewMode.value = 'chat'
}

function updateCharacterData() {
  if (!selectedCharacter.value) return
  selectedCharacter.value.data = { ...characterForm.value }
}

async function generateRunninghubCharacterPics() {
  const backend = (localStorage.getItem(IMAGE_BACKEND_STORAGE_KEY) || 'runninghub').trim()
  const wf = localStorage.getItem(RUNNINGHUB_WORKFLOW_STORAGE_KEY)?.trim()
  if (backend === 'runninghub' && !wf) {
    alert('请先在「设置」页面填写 RunningHub 工作流 ID')
    return
  }
  const appearance = (characterForm.value.appearance || '').trim()
  const personality = (characterForm.value.personality || '').trim()
  if (!appearance && !personality) {
    alert('请先填写「外貌描述」或「性格设定」')
    return
  }
  let stand_custom_items = []
  if (standExpressionMode.value === 'custom') {
    stand_custom_items = standCustomRows.value
      .map((r) => ({
        id: (r.id || '').trim(),
        description: (r.description || '').trim()
      }))
      .filter((r) => r.id && r.description)
    if (!stand_custom_items.length) {
      alert('自定义模式请至少填写一组「id」与「description」')
      return
    }
  }
  if (!characterForm.value.metadata || typeof characterForm.value.metadata !== 'object') {
    characterForm.value.metadata = {}
  }
  characterForm.value.metadata.stand_expression_mode = standExpressionMode.value
  characterForm.value.metadata.stand_custom_items =
    standExpressionMode.value === 'custom' ? stand_custom_items : []
  updateCharacterData()
  runninghubPicLoading.value = true
  try {
    const res = await fetch(`${API_BASE_URL}/api/image/generate-character-pics`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        workflow_id: wf || '',
        character_name: (characterForm.value.name || '').trim(),
        appearance,
        personality,
        image_backend: backend,
        comfyui_checkpoint: (localStorage.getItem(COMFYUI_CKPT_STORAGE_KEY) || '').trim(),
        comfyui_workflow: (localStorage.getItem(COMFYUI_WORKFLOW_STORAGE_KEY) || '').trim(),
        comfyui_size_ratio: (localStorage.getItem(COMFYUI_SIZE_RATIO_STORAGE_KEY) || '').trim(),
        stand_expression_mode: standExpressionMode.value,
        stand_custom_items,
        image_prompt_extra: (characterForm.value.image_prompt_extra || '').trim()
      })
    })
    const raw = await res.text()
    let data
    try {
      data = JSON.parse(raw)
    } catch {
      data = { detail: raw }
    }
    if (!res.ok) {
      const msg =
        typeof data.detail === 'string'
          ? data.detail
          : Array.isArray(data.detail)
            ? data.detail.map((d) => d.msg || d).join('; ')
            : raw
      throw new Error(msg || `HTTP ${res.status}`)
    }
    alert(
      `${data.message || '已完成'}\n静态路径：${data.public_base || '/sources/pic/'}（文件名如 happy_1.png；耗时较长属正常；可打开该目录将图片再「添加」到本角色）`
    )
  } catch (e) {
    console.error(e)
    alert('生成失败：' + (e.message || String(e)))
  } finally {
    runninghubPicLoading.value = false
  }
}

async function applyRemoveStandPicBackgrounds() {
  if (!selectedCharacter.value) return
  updateCharacterData()
  const name = (characterForm.value.name || '').trim()
  const mode = characterForm.value.metadata?.stand_expression_mode
  const items = characterForm.value.metadata?.stand_custom_items
  let stand_expression_ids
  if (mode === 'custom' && Array.isArray(items) && items.length) {
    stand_expression_ids = items.map((it) => String(it.id || '').trim()).filter(Boolean)
  }
  removeStandBgLoading.value = true
  try {
    const res = await fetch(`${API_BASE_URL}/api/image/remove-stand-pic-backgrounds`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        character_name: name,
        ...(stand_expression_ids?.length ? { stand_expression_ids } : {})
      })
    })
    const raw = await res.text()
    let data
    try {
      data = JSON.parse(raw)
    } catch {
      data = { detail: raw }
    }
    if (!res.ok) {
      const msg =
        typeof data.detail === 'string'
          ? data.detail
          : Array.isArray(data.detail)
            ? data.detail.map((d) => d.msg || d).join('; ')
            : raw
      throw new Error(msg || `HTTP ${res.status}`)
    }
    const v = Date.now()
    selectedCharacter.value.images = (data.images || []).map((im) => ({
      id: generateId(),
      url: `${im.url}?v=${v}`,
      name: im.name
    }))
    selectedImageIndex.value = 0
    saveCharactersToStorage()
    alert(data.message || '已用去背景立绘替换当前角色的图片列表')
  } catch (e) {
    console.error(e)
    alert('去背景失败：' + (e.message || String(e)))
  } finally {
    removeStandBgLoading.value = false
  }
}

function selectImage(index) {
  selectedImageIndex.value = index
}

function handleImageUpload(event) {
  const files = event.target.files
  if (!files || !selectedCharacter.value) return
  
  Array.from(files).forEach(file => {
    const reader = new FileReader()
    reader.onload = (e) => {
      selectedCharacter.value.images.push({
        id: generateId(),
        url: e.target.result,
        name: file.name
      })
    }
    reader.readAsDataURL(file)
  })
  event.target.value = ''
}

function removeImage(index) {
  if (!selectedCharacter.value) return
  selectedCharacter.value.images.splice(index, 1)
  if (selectedImageIndex.value >= currentImages.value.length) {
    selectedImageIndex.value = Math.max(0, currentImages.value.length - 1)
  }
}

const fileInputRef = ref(null)
const importInputRef = ref(null)

function triggerImport() {
  importInputRef.value?.click()
}

function triggerExport() {
  if (!selectedCharacter.value) {
    alert('请先选择一个角色')
    return
  }
  exportCharacterCard(selectedCharacter.value)
}

async function handleImportCharacterCard(event) {
  const file = event.target.files?.[0]
  if (!file) return

  const nameLower = (file.name || '').toLowerCase()

  try {
    if (nameLower.endsWith('.json')) {
      const text = await file.text()
      const parsed = JSON.parse(text)
      const list = Array.isArray(parsed) ? parsed : parsed && typeof parsed === 'object' ? [parsed] : []
      if (list.length === 0) {
        throw new Error('JSON 中未找到角色数据')
      }
      for (const item of list) {
        if (!item || typeof item !== 'object' || !item.data) continue
        characters.value.push({
          id: item.id || generateId(),
          data: { ...defaultCharacterData, ...item.data },
          images: Array.isArray(item.images) ? item.images : []
        })
      }
      const last = characters.value[characters.value.length - 1]
      if (last) selectedCharacterId.value = last.id
      saveCharactersToStorage()
      event.target.value = ''
      return
    }

    const characterData = await parseCharacterCard(file)
    if (characterData) {
      const newChar = {
        id: generateId(),
        data: characterData.data,
        images: characterData.images || []
      }
      characters.value.push(newChar)
      selectedCharacterId.value = newChar.id
      saveCharactersToStorage()
    }
  } catch (error) {
    console.error('导入角色卡失败:', error)
    alert('导入角色卡失败: ' + error.message)
  }
  event.target.value = ''
}

function parseCharacterCard(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    
    reader.onload = (e) => {
      try {
        const arrayBuffer = e.target.result
        const uint8Array = new Uint8Array(arrayBuffer)
        
        const pngSignature = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
        const isPng = pngSignature.every((byte, index) => uint8Array[index] === byte)
        
        if (!isPng) {
          reject(new Error('不是有效的PNG文件'))
          return
        }
        
        let textChunks = []
        let offset = 8
        
        while (offset < uint8Array.length) {
          const length = (uint8Array[offset] << 24) | (uint8Array[offset + 1] << 16) | 
                         (uint8Array[offset + 2] << 8) | uint8Array[offset + 3]
          const type = String.fromCharCode(...uint8Array.slice(offset + 4, offset + 8))
          
          if (type === 'tEXt' || type === 'iTXt') {
            const chunkData = uint8Array.slice(offset + 8, offset + 8 + length)
            const nullIndex = chunkData.findIndex(byte => byte === 0)
            const keyword = String.fromCharCode(...chunkData.slice(0, nullIndex))
            const text = new TextDecoder('utf-8').decode(chunkData.slice(nullIndex + 1))
            textChunks.push({ keyword, text })
          }
          
          offset += 12 + length
        }
        
        const charaChunk = textChunks.find(c => c.keyword === 'chara') || 
                           textChunks.find(c => c.keyword === 'ccv3') ||
                           textChunks.find(c => c.keyword === 'ccv2')
        
        if (!charaChunk) {
          reject(new Error('未找到角色数据'))
          return
        }
        
        let jsonStr
        try {
          jsonStr = atob(charaChunk.text)
        } catch {
          jsonStr = charaChunk.text
        }
        
        const charData = JSON.parse(jsonStr)
        
        const images = []
        
        let binary = ''
        for (let i = 0; i < uint8Array.length; i++) {
          binary += String.fromCharCode(uint8Array[i])
        }
        const base64Png = btoa(binary)
        images.push({
          id: generateId(),
          url: 'data:image/png;base64,' + base64Png,
          name: file.name || 'character_card'
        })
        
        const mappedData = {
          name: charData.name || charData.data?.name || charData.char_name || '',
          age: charData.age || charData.data?.extensions?.age || '',
          gender: charData.gender || charData.data?.extensions?.gender || '',
          appearance: charData.appearance || charData.data?.description || charData.description || '',
          personality: charData.personality || charData.data?.personality || charData.personality_prompt || '',
          image_prompt_extra:
            charData.data?.extensions?.image_prompt_extra ||
            charData.image_prompt_extra ||
            '',
          background: charData.background || charData.data?.scenario || charData.backstory || '',
          dialogue_examples: charData.mes_example || charData.data?.mes_example || charData.dialogue_examples || '',
          other_settings: charData.other_settings || '',
          metadata: {
            creator: charData.creator || charData.data?.creator || '',
            tags: charData.tags || charData.data?.tags || [],
            system_prompt: charData.system_prompt || charData.data?.system_prompt || '',
            post_history_instructions:
              charData.post_history_instructions || charData.data?.post_history_instructions || '',
            character_version: charData.character_version || charData.data?.character_version || '',
            first_mes: charData.first_mes || charData.data?.first_mes || ''
          }
        }
        
        resolve({ data: mappedData, images })
      } catch (error) {
        reject(error)
      }
    }
    
    reader.onerror = () => reject(new Error('文件读取失败'))
    reader.readAsArrayBuffer(file)
  })
}

function exportCharacterCard(character) {
  const exportData = {
    spec: 'chara_card_v2',
    spec_version: '2.0',
    data: {
      name: character.data.name,
      description: character.data.appearance,
      personality: character.data.personality,
      scenario: character.data.background,
      first_mes: '',
      mes_example: character.data.dialogue_examples,
      creator_notes: '',
      system_prompt: character.data.metadata?.system_prompt || '',
      post_history_instructions: character.data.metadata?.post_history_instructions || '',
      alternate_greetings: [],
      tags: character.data.metadata?.tags || [],
      creator: character.data.metadata?.creator || '',
      character_version: character.data.metadata?.character_version || '1.0',
      extensions: {
        age: character.data.age,
        gender: character.data.gender,
        other_settings: character.data.other_settings,
        image_prompt_extra: character.data.image_prompt_extra || ''
      }
    }
  }
  
  const jsonStr = JSON.stringify(exportData, null, 2)
  const base64Data = btoa(unescape(encodeURIComponent(jsonStr)))
  
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  
  let imageToUse = character.images[0]?.url
  if (imageToUse) {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      canvas.width = img.width
      canvas.height = img.height
      ctx.drawImage(img, 0, 0)
      
      const pngDataUrl = canvas.toDataURL('image/png')
      const base64Png = pngDataUrl.split(',')[1]
      
      const pngWithMetadata = injectTextChunk(base64Png, 'chara', base64Data)
      
      downloadFile(pngWithMetadata, `${character.data.name || 'character'}_card.png`)
    }
    img.onerror = () => {
      createDefaultCharacterCard(exportData, character.data.name)
    }
    img.src = imageToUse
  } else {
    createDefaultCharacterCard(exportData, character.data.name)
  }
}

function createDefaultCharacterCard(exportData, name) {
  const canvas = document.createElement('canvas')
  canvas.width = 400
  canvas.height = 400
  const ctx = canvas.getContext('2d')
  
  const gradient = ctx.createLinearGradient(0, 0, 400, 400)
  gradient.addColorStop(0, '#1a1a2e')
  gradient.addColorStop(1, '#16213e')
  ctx.fillStyle = gradient
  ctx.fillRect(0, 0, 400, 400)
  
  ctx.fillStyle = '#00d4ff'
  ctx.font = 'bold 48px Arial'
  ctx.textAlign = 'center'
  ctx.fillText(name?.charAt(0) || '?', 200, 220)
  
  const pngDataUrl = canvas.toDataURL('image/png')
  const base64Png = pngDataUrl.split(',')[1]
  const jsonStr = JSON.stringify(exportData, null, 2)
  const base64Data = btoa(unescape(encodeURIComponent(jsonStr)))
  
  const pngWithMetadata = injectTextChunk(base64Png, 'chara', base64Data)
  downloadFile(pngWithMetadata, `${name || 'character'}_card.png`)
}

function injectTextChunk(base64Png, keyword, text) {
  const binaryString = atob(base64Png)
  const bytes = new Uint8Array(binaryString.length)
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i)
  }
  
  const keywordBytes = new TextEncoder().encode(keyword)
  const textBytes = new TextEncoder().encode(text)
  const chunkData = new Uint8Array(keywordBytes.length + 1 + textBytes.length)
  chunkData.set(keywordBytes, 0)
  chunkData[keywordBytes.length] = 0
  chunkData.set(textBytes, keywordBytes.length + 1)
  
  const chunkType = new TextEncoder().encode('tEXt')
  const crcData = new Uint8Array(chunkType.length + chunkData.length)
  crcData.set(chunkType, 0)
  crcData.set(chunkData, chunkType.length)
  const crc = calculateCRC32(crcData)
  
  const lengthBytes = new Uint8Array(4)
  lengthBytes[0] = (chunkData.length >> 24) & 0xFF
  lengthBytes[1] = (chunkData.length >> 16) & 0xFF
  lengthBytes[2] = (chunkData.length >> 8) & 0xFF
  lengthBytes[3] = chunkData.length & 0xFF
  
  const crcBytes = new Uint8Array(4)
  crcBytes[0] = (crc >> 24) & 0xFF
  crcBytes[1] = (crc >> 16) & 0xFF
  crcBytes[2] = (crc >> 8) & 0xFF
  crcBytes[3] = crc & 0xFF
  
  let ihdrEnd = 8
  while (ihdrEnd < bytes.length) {
    const chunkLen = (bytes[ihdrEnd] << 24) | (bytes[ihdrEnd + 1] << 16) | 
                     (bytes[ihdrEnd + 2] << 8) | bytes[ihdrEnd + 3]
    const chunkTypeStr = String.fromCharCode(...bytes.slice(ihdrEnd + 4, ihdrEnd + 8))
    if (chunkTypeStr === 'IHDR') {
      ihdrEnd += 12 + chunkLen
      break
    }
    ihdrEnd += 12 + chunkLen
  }
  
  const newPng = new Uint8Array(bytes.length + 12 + chunkData.length)
  newPng.set(bytes.slice(0, ihdrEnd), 0)
  newPng.set(lengthBytes, ihdrEnd)
  newPng.set(chunkType, ihdrEnd + 4)
  newPng.set(chunkData, ihdrEnd + 8)
  newPng.set(crcBytes, ihdrEnd + 8 + chunkData.length)
  newPng.set(bytes.slice(ihdrEnd), ihdrEnd + 12 + chunkData.length)
  
  let binary = ''
  for (let i = 0; i < newPng.length; i++) {
    binary += String.fromCharCode(newPng[i])
  }
  return btoa(binary)
}

function calculateCRC32(data) {
  let crc = 0xFFFFFFFF
  const table = []
  
  for (let i = 0; i < 256; i++) {
    let c = i
    for (let j = 0; j < 8; j++) {
      c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1)
    }
    table[i] = c
  }
  
  for (let i = 0; i < data.length; i++) {
    crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >>> 8)
  }
  
  return (crc ^ 0xFFFFFFFF) >>> 0
}

function downloadFile(base64Data, filename) {
  const link = document.createElement('a')
  link.href = 'data:image/png;base64,' + base64Data
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

function exportAllCharacters() {
  const data = characters.value.map(c => ({
    id: c.id,
    data: c.data,
    images: c.images
  }))
  const jsonStr = JSON.stringify(data, null, 2)
  const blob = new Blob([jsonStr], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = 'characters_backup.json'
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

function scrollChatToBottom() {
  nextTick(() => {
    const el = chatMessagesRef.value
    if (el) {
      el.scrollTop = el.scrollHeight
    }
  })
}

async function loadCharacterChatHistory() {
  const char = selectedCharacter.value
  if (!char) {
    chatMessages.value = []
    return
  }
  const sid = char.id
  const nameKey = (char.data?.name || '').trim() || '未命名'
  try {
    const res = await fetch(
      `${API_BASE_URL}/api/character-chat/history?character_name=${encodeURIComponent(nameKey)}`
    )
    const rawText = await res.text()
    let data
    try {
      data = JSON.parse(rawText)
    } catch {
      data = { detail: rawText }
    }
    if (!res.ok) {
      const msg =
        typeof data.detail === 'string'
          ? data.detail
          : Array.isArray(data.detail)
            ? data.detail.map((d) => d.msg || d).join('; ')
            : rawText
      throw new Error(msg || `HTTP ${res.status}`)
    }
    if (selectedCharacterId.value !== sid) return
    const arr = Array.isArray(data.messages) ? data.messages : []
    chatMessages.value = arr.map((m) => ({
      id: generateId(),
      role: m.role === 'assistant' ? 'assistant' : 'user',
      content: String(m.content || '')
    }))
    scrollChatToBottom()
  } catch (e) {
    console.warn('加载人物对话历史失败:', e)
    if (selectedCharacterId.value === sid) {
      chatMessages.value = []
    }
  }
}

async function sendMessage() {
  const text = chatInput.value.trim()
  if (!text || !selectedCharacter.value || chatSending.value) return

  const sid = selectedCharacter.value.id
  const userMsg = {
    id: generateId(),
    role: 'user',
    content: text
  }
  chatMessages.value.push(userMsg)
  chatInput.value = ''
  scrollChatToBottom()

  chatSending.value = true
  try {
    const card = { ...(selectedCharacter.value.data || {}) }
    const nameKey = (card.name || '').trim() || '未命名'
    const payloadMsgs = chatMessages.value.map((m) => ({
      role: m.role,
      content: m.content
    }))
    const res = await fetch(`${API_BASE_URL}/api/character-chat/send`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        character_name: nameKey,
        card,
        messages: payloadMsgs
      })
    })
    const rawText = await res.text()
    let data
    try {
      data = JSON.parse(rawText)
    } catch {
      data = { detail: rawText }
    }
    if (!res.ok) {
      const msg =
        typeof data.detail === 'string'
          ? data.detail
          : Array.isArray(data.detail)
            ? data.detail.map((d) => d.msg || d).join('; ')
            : rawText
      throw new Error(msg || `HTTP ${res.status}`)
    }
    if (selectedCharacterId.value !== sid) return
    const list = Array.isArray(data.messages) ? data.messages : []
    chatMessages.value = list.map((m) => ({
      id: generateId(),
      role: m.role === 'assistant' ? 'assistant' : 'user',
      content: String(m.content || '')
    }))
    scrollChatToBottom()
  } catch (e) {
    console.error('人物对话发送失败:', e)
    if (selectedCharacterId.value === sid) {
      chatMessages.value = chatMessages.value.filter((m) => m.id !== userMsg.id)
    }
    alert(e?.message || String(e))
  } finally {
    if (selectedCharacterId.value === sid) {
      chatSending.value = false
    }
    scrollChatToBottom()
  }
}

function handleKeyDown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    if (chatSending.value) return
    event.preventDefault()
    sendMessage()
  }
}
</script>

<template>
  <div class="character-page">
    <input
      ref="fileInputRef"
      type="file"
      accept="image/*"
      multiple
      style="display: none"
      @change="handleImageUpload"
    />
    <input
      ref="importInputRef"
      type="file"
      accept=".png,.json"
      style="display: none"
      @change="handleImportCharacterCard"
    />
    
    <div class="page-header">
      <h2>人物卡管理</h2>
      <div class="header-actions">
        <button class="action-btn import-btn" @click="triggerImport">
          <span class="btn-icon">📥</span>
          导入角色卡
        </button>
        <button class="action-btn export-btn" @click="triggerExport">
          <span class="btn-icon">📤</span>
          导出角色卡
        </button>
        <button class="action-btn backup-btn" @click="exportAllCharacters">
          <span class="btn-icon">💾</span>
          备份全部
        </button>
      </div>
    </div>
    
    <div class="main-container">
      <template v-if="viewMode === 'chat'">
        <div class="chat-panel">
          <div v-if="selectedCharacter" class="chat-content chat-layout-split">
            <aside class="chat-left-rail" aria-label="角色立绘与简介">
              <div class="rail-portrait">
                <img
                  v-if="firstImage"
                  class="rail-portrait-img"
                  :src="firstImage.url"
                  :alt="selectedCharacter.data.name || '角色'"
                />
                <div v-else class="no-preview-image rail-placeholder">
                  <span class="no-image-icon">👤</span>
                  <span>{{ selectedCharacter.data.name?.charAt(0) || '?' }}</span>
                </div>
              </div>
              <div class="rail-meta">
                <h3 class="rail-title">{{ selectedCharacter.data.name || '未命名角色' }}</h3>
                <p v-if="selectedCharacter.data.personality" class="rail-personality">
                  {{ selectedCharacter.data.personality.slice(0, 220)
                  }}{{ selectedCharacter.data.personality.length > 220 ? '…' : '' }}
                </p>
                <p class="rail-hint">
                  对话按角色名保存至 <code>public/character_chat</code>；模型侧会注入人物卡为性格设定，长对话自动压缩较早上下文。
                </p>
              </div>
            </aside>

            <div class="chat-main-column">
              <div ref="chatMessagesRef" class="chat-messages">
                <div v-if="chatMessages.length === 0" class="empty-chat">
                  <span class="empty-icon">💬</span>
                  <p>开始与 {{ selectedCharacter.data.name || '角色' }} 对话</p>
                </div>
                <div
                  v-for="msg in chatMessages"
                  :key="msg.id"
                  :class="['message', msg.role]"
                >
                  <div class="message-avatar">
                    <img v-if="msg.role === 'assistant' && firstImage" :src="firstImage.url" alt="" />
                    <span v-else>我</span>
                  </div>
                  <div class="message-content">
                    <div class="message-name">{{ msg.role === 'user' ? '我' : selectedCharacter.data.name }}</div>
                    <div class="message-text">{{ msg.content }}</div>
                  </div>
                </div>
              </div>

              <div v-if="chatSending" class="chat-typing" role="status">正在回复…</div>

              <div class="chat-input-area">
                <textarea
                  v-model="chatInput"
                  placeholder="输入消息…（Enter 发送，Shift+Enter 换行）"
                  :disabled="chatSending"
                  @keydown="handleKeyDown"
                  rows="1"
                ></textarea>
                <button class="send-btn" type="button" :disabled="chatSending" @click="sendMessage">
                  <span>{{ chatSending ? '等待中' : '发送' }}</span>
                </button>
              </div>
            </div>
          </div>
          
          <div v-else class="no-character-selected">
            <div class="empty-state">
              <span class="empty-icon">👤</span>
              <p>请选择或创建一个角色</p>
              <button class="create-btn" @click="createNewCharacter">
                <span>✨</span> 创建新角色
              </button>
            </div>
          </div>
        </div>
      </template>
      
      <template v-else>
        <div class="edit-panel">
          <div class="left-panel">
            <div class="image-preview-section">
              <div class="main-image-container">
                <div v-if="currentMainImage" class="main-image">
                  <img :src="currentMainImage.url" :alt="selectedCharacter?.data?.name" />
                </div>
                <div v-else class="no-image">
                  <span class="no-image-icon">🖼️</span>
                  <span class="no-image-text">暂无图片</span>
                </div>
              </div>
              
              <div class="image-actions" v-if="selectedCharacter">
                <button class="upload-btn" @click="fileInputRef?.click()">
                  <span>📷</span> 添加图片
                </button>
              </div>
              
              <div class="thumbnail-list" v-if="currentImages.length > 0">
                <div
                  v-for="(img, index) in currentImages"
                  :key="img.id"
                  :class="['thumbnail-item', { active: selectedImageIndex === index }]"
                  @click="selectImage(index)"
                >
                  <img :src="img.url" :alt="`图片 ${index + 1}`" />
                  <button class="remove-img-btn" @click.stop="removeImage(index)">×</button>
                </div>
              </div>
            </div>
          </div>
          
          <div class="center-panel">
            <div v-if="selectedCharacter" class="character-editor">
              <div class="editor-header">
                <button class="back-btn" @click="closeEditMode">
                  <span>←</span> 返回对话
                </button>
                <h3>{{ characterForm.name || '未命名角色' }}</h3>
              </div>
              
              <div class="editor-content">
                <div class="form-group">
                  <label>角色姓名</label>
                  <input
                    type="text"
                    v-model="characterForm.name"
                    @input="updateCharacterData"
                    placeholder="输入角色姓名"
                  />
                </div>
                
                <div class="form-row">
                  <div class="form-group">
                    <label>年龄</label>
                    <input
                      type="text"
                      v-model="characterForm.age"
                      @input="updateCharacterData"
                      placeholder="输入年龄"
                    />
                  </div>
                  <div class="form-group">
                    <label>性别</label>
                    <select v-model="characterForm.gender" @change="updateCharacterData">
                      <option value="">请选择</option>
                      <option value="男">男</option>
                      <option value="女">女</option>
                      <option value="其他">其他</option>
                    </select>
                  </div>
                </div>
                
                <div class="form-group">
                  <label>外貌描述</label>
                  <textarea
                    v-model="characterForm.appearance"
                    @input="updateCharacterData"
                    placeholder="描述角色的外貌特征..."
                    rows="4"
                  ></textarea>
                </div>
                
                <div class="form-group">
                  <label>性格设定</label>
                  <textarea
                    v-model="characterForm.personality"
                    @input="updateCharacterData"
                    placeholder="描述角色的性格特点..."
                    rows="4"
                  ></textarea>
                </div>

                <div class="form-group">
                  <label>自定义生图参数（可选）</label>
                  <textarea
                    v-model="characterForm.image_prompt_extra"
                    @input="updateCharacterData"
                    placeholder="英文逗号分隔，拼到每条立绘正向 prompt 末尾（如画师、画风、Lora 触发词）；不经大模型改写"
                    rows="2"
                  ></textarea>
                </div>

                <div class="form-group runninghub-pic-block">
                  <label>立绘生成</label>
                  <div class="stand-mode-options">
                    <span class="stand-mode-label">表情方案</span>
                    <label class="stand-mode-radio">
                      <input v-model="standExpressionMode" type="radio" value="default" />
                      默认（内置全套表情）
                    </label>
                    <label class="stand-mode-radio">
                      <input v-model="standExpressionMode" type="radio" value="custom" />
                      自定义（自行填写 id + description）
                    </label>
                  </div>
                  <div v-if="standExpressionMode === 'custom'" class="stand-custom-editor">
                    <p class="stand-custom-hint">
                      id：保存为 <code>{id}_1.png</code>；description：与原先内置表情相同，拼在
                      <code>…, cowboy_shot, &lt;description&gt;</code> 末尾。
                    </p>
                    <div
                      v-for="(row, idx) in standCustomRows"
                      :key="idx"
                      class="stand-custom-row"
                    >
                      <input
                        v-model.trim="row.id"
                        type="text"
                        class="stand-custom-input"
                        placeholder="id，如 happy、pose_01"
                        autocomplete="off"
                      />
                      <input
                        v-model.trim="row.description"
                        type="text"
                        class="stand-custom-input"
                        placeholder="description，如 happy 或英文分词"
                        autocomplete="off"
                      />
                      <button
                        type="button"
                        class="stand-custom-remove"
                        :disabled="standCustomRows.length <= 1"
                        @click="removeStandCustomRow(idx)"
                      >
                        删
                      </button>
                    </div>
                    <button type="button" class="stand-custom-add" @click="addStandCustomRow">
                      + 添加一组
                    </button>
                  </div>
                  <p class="runninghub-pic-hint">
                    在「设置」中选择 RunningHub 或本地 ComfyUI，并按要求填写工作流 ID / Checkpoint。将结合上方外貌与性格生成多套表情图到
                    public/sources/pic（与原先 RunningHub 路径一致）。云端排队较久请勿关页；本地 ComfyUI
                    需已启动且工作流与仓库
                    public/comfyui/workflow1.json 一致。「去背景」走本地 ComfyUI（默认
                    public/comfyui/sp_costum_workflow/remove_bg.json，需 RMBG 节点）。
                  </p>
                  <div class="runninghub-pic-actions">
                    <button
                      type="button"
                      class="runninghub-pic-btn"
                      :disabled="runninghubPicLoading || removeStandBgLoading"
                      @click="generateRunninghubCharacterPics"
                    >
                      {{
                        runninghubPicLoading
                          ? '生成中（请勿关闭页面）…'
                          : '🖼 生成立绘'
                      }}
                    </button>
                    <button
                      type="button"
                      class="runninghub-pic-btn runninghub-pic-btn-secondary"
                      :disabled="runninghubPicLoading || removeStandBgLoading"
                      @click="applyRemoveStandPicBackgrounds"
                    >
                      {{
                        removeStandBgLoading
                          ? '去背景处理中…'
                          : '✂️ 立绘去背景并替换'
                      }}
                    </button>
                  </div>
                </div>
                
                <div class="form-group">
                  <label>背景故事</label>
                  <textarea
                    v-model="characterForm.background"
                    @input="updateCharacterData"
                    placeholder="描述角色的背景故事..."
                    rows="5"
                  ></textarea>
                </div>
                
                <div class="form-group">
                  <label>对话示例</label>
                  <textarea
                    v-model="characterForm.dialogue_examples"
                    @input="updateCharacterData"
                    placeholder="输入角色的对话示例..."
                    rows="5"
                  ></textarea>
                </div>
                
                <div class="form-group">
                  <label>其他设定</label>
                  <textarea
                    v-model="characterForm.other_settings"
                    @input="updateCharacterData"
                    placeholder="其他补充设定..."
                    rows="3"
                  ></textarea>
                </div>
                
                <div class="form-group metadata-group">
                  <label>元数据</label>
                  <div class="metadata-info" v-if="characterForm.metadata">
                    <div class="metadata-item" v-if="characterForm.metadata.creator">
                      <span class="meta-label">创建者:</span>
                      <span class="meta-value">{{ characterForm.metadata.creator }}</span>
                    </div>
                    <div class="metadata-item" v-if="characterForm.metadata.character_version">
                      <span class="meta-label">版本:</span>
                      <span class="meta-value">{{ characterForm.metadata.character_version }}</span>
                    </div>
                    <div class="metadata-item" v-if="characterForm.metadata.tags?.length">
                      <span class="meta-label">标签:</span>
                      <div class="tags-list">
                        <span v-for="tag in characterForm.metadata.tags" :key="tag" class="tag">{{ tag }}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </template>
      
      <div class="right-panel">
        <div class="character-list-header">
          <h3>角色列表</h3>
          <button class="add-character-btn" @click="createNewCharacter">
            <span>+</span> 新建
          </button>
        </div>
        
        <div class="character-list">
          <div
            v-for="char in characters"
            :key="char.id"
            :class="['character-item', { active: selectedCharacterId === char.id }]"
          >
            <div class="char-main" @click="selectCharacter(char.id)">
              <div class="char-avatar">
                <img v-if="char.images[0]?.url" :src="char.images[0].url" :alt="char.data.name" />
                <span v-else class="avatar-placeholder">{{ char.data.name?.charAt(0) || '?' }}</span>
              </div>
              <div class="char-info">
                <span class="char-name">{{ char.data.name || '未命名' }}</span>
                <span class="char-meta">{{ char.images.length }} 张图片</span>
              </div>
            </div>
            <div class="char-actions">
              <button class="edit-btn" @click="selectedCharacterId = char.id; openEditMode()" title="编辑">
                ✏️
              </button>
              <button class="delete-btn" @click.stop="deleteCharacter(char.id)" title="删除">
                🗑️
              </button>
            </div>
          </div>
          
          <div v-if="characters.length === 0" class="empty-list">
            <p>暂无角色</p>
            <button class="create-btn-small" @click="createNewCharacter">创建第一个角色</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.character-page {
  height: calc(100vh - 60px);
  display: flex;
  flex-direction: column;
  background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
  overflow: hidden;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 24px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  flex-shrink: 0;
}

.page-header h2 {
  margin: 0;
  font-size: 1.5rem;
  color: #fff;
}

.header-actions {
  display: flex;
  gap: 10px;
}

.action-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  border: none;
  border-radius: 8px;
  font-size: 0.85rem;
  cursor: pointer;
  transition: all 0.3s ease;
}

.btn-icon {
  font-size: 1rem;
}

.import-btn {
  background: rgba(0, 212, 255, 0.2);
  color: #00d4ff;
  border: 1px solid rgba(0, 212, 255, 0.3);
}

.import-btn:hover {
  background: rgba(0, 212, 255, 0.3);
  border-color: #00d4ff;
}

.export-btn {
  background: rgba(123, 44, 191, 0.2);
  color: #b366e9;
  border: 1px solid rgba(123, 44, 191, 0.3);
}

.export-btn:hover {
  background: rgba(123, 44, 191, 0.3);
  border-color: #b366e9;
}

.backup-btn {
  background: rgba(46, 204, 113, 0.2);
  color: #2ecc71;
  border: 1px solid rgba(46, 204, 113, 0.3);
}

.backup-btn:hover {
  background: rgba(46, 204, 113, 0.3);
  border-color: #2ecc71;
}

.main-container {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.chat-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.chat-content {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
}

.chat-layout-split {
  flex-direction: row;
  align-items: stretch;
}

.chat-left-rail {
  width: 268px;
  min-width: 220px;
  max-width: min(32vw, 300px);
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  border-right: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(15, 15, 26, 0.88);
}

.rail-portrait {
  flex: 1;
  min-height: 180px;
  max-height: min(52vh, 520px);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 14px 12px;
  background: rgba(26, 26, 46, 0.55);
}

.rail-portrait-img {
  max-width: 100%;
  max-height: 100%;
  width: auto;
  height: auto;
  object-fit: contain;
  border-radius: 14px;
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35);
}

.rail-placeholder {
  min-height: 160px;
}

.no-preview-image {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: rgba(255, 255, 255, 0.5);
}

.no-preview-image .no-image-icon {
  font-size: 2.5rem;
}

.no-preview-image span:last-child {
  font-size: 1.75rem;
  font-weight: 600;
}

.rail-meta {
  flex-shrink: 0;
  padding: 12px 14px 16px;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
}

.rail-title {
  margin: 0 0 8px;
  font-size: 1.1rem;
  color: #fff;
}

.rail-personality {
  margin: 0 0 10px;
  font-size: 0.78rem;
  line-height: 1.45;
  color: rgba(255, 255, 255, 0.72);
  max-height: 7.5em;
  overflow-y: auto;
}

.rail-hint {
  margin: 0;
  font-size: 0.68rem;
  line-height: 1.4;
  color: rgba(255, 255, 255, 0.42);
}

.rail-hint code {
  font-size: 0.65em;
  color: rgba(0, 212, 255, 0.85);
}

.chat-main-column {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.chat-typing {
  padding: 4px 16px 0;
  font-size: 0.78rem;
  color: rgba(0, 212, 255, 0.75);
  flex-shrink: 0;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.empty-chat {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: rgba(255, 255, 255, 0.4);
}

.empty-chat .empty-icon {
  font-size: 3rem;
  margin-bottom: 12px;
}

.empty-chat p {
  margin: 0;
}

.message {
  display: flex;
  gap: 12px;
  max-width: 80%;
}

.message.user {
  align-self: flex-end;
  flex-direction: row-reverse;
}

.message.assistant {
  align-self: flex-start;
}

.message-avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  overflow: hidden;
  background: linear-gradient(135deg, #00d4ff 0%, #7b2cbf 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.message-avatar img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.message-avatar span {
  font-size: 0.85rem;
  font-weight: 600;
  color: #fff;
}

.message-content {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.message-name {
  font-size: 0.8rem;
  color: rgba(255, 255, 255, 0.5);
}

.message.user .message-name {
  text-align: right;
}

.message-text {
  padding: 12px 16px;
  border-radius: 16px;
  font-size: 0.95rem;
  line-height: 1.5;
  word-break: break-word;
}

.message.user .message-text {
  background: linear-gradient(135deg, #00d4ff 0%, #7b2cbf 100%);
  color: #fff;
  border-bottom-right-radius: 4px;
}

.message.assistant .message-text {
  background: rgba(26, 26, 46, 0.8);
  color: rgba(255, 255, 255, 0.9);
  border-bottom-left-radius: 4px;
}

.chat-input-area {
  display: flex;
  gap: 12px;
  padding: 16px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(15, 15, 26, 0.8);
  flex-shrink: 0;
}

.chat-input-area textarea {
  flex: 1;
  padding: 12px 16px;
  background: rgba(26, 26, 46, 0.8);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 12px;
  color: #fff;
  font-size: 0.95rem;
  resize: none;
  font-family: inherit;
  min-height: 44px;
  max-height: 120px;
}

.chat-input-area textarea:focus {
  outline: none;
  border-color: #00d4ff;
}

.chat-input-area textarea::placeholder {
  color: rgba(255, 255, 255, 0.3);
}

.send-btn {
  padding: 12px 24px;
  background: linear-gradient(135deg, #00d4ff 0%, #7b2cbf 100%);
  border: none;
  border-radius: 12px;
  color: #fff;
  font-size: 0.95rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
}

.send-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
}

.send-btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
  transform: none;
}

.edit-panel {
  flex: 1;
  display: flex;
  overflow: hidden;
}

.left-panel {
  width: 320px;
  min-width: 280px;
  border-right: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  flex-direction: column;
  background: rgba(15, 15, 26, 0.5);
}

.image-preview-section {
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 16px;
}

.main-image-container {
  flex: 1;
  min-height: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(26, 26, 46, 0.6);
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  overflow: hidden;
  margin-bottom: 12px;
}

.main-image {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.main-image img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.no-image {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  color: rgba(255, 255, 255, 0.3);
}

.no-image-icon {
  font-size: 3rem;
}

.no-image-text {
  font-size: 0.9rem;
}

.image-actions {
  display: flex;
  justify-content: center;
  margin-bottom: 12px;
}

.upload-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 20px;
  background: linear-gradient(135deg, #00d4ff 0%, #7b2cbf 100%);
  border: none;
  border-radius: 8px;
  color: #fff;
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.3s ease;
}

.upload-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
}

.thumbnail-list {
  display: flex;
  gap: 8px;
  overflow-x: auto;
  padding: 8px 0;
  flex-shrink: 0;
}

.thumbnail-item {
  position: relative;
  width: 60px;
  height: 60px;
  flex-shrink: 0;
  border-radius: 8px;
  overflow: hidden;
  cursor: pointer;
  border: 2px solid transparent;
  transition: all 0.3s ease;
}

.thumbnail-item.active {
  border-color: #00d4ff;
}

.thumbnail-item:hover {
  transform: scale(1.05);
}

.thumbnail-item img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.remove-img-btn {
  position: absolute;
  top: 2px;
  right: 2px;
  width: 18px;
  height: 18px;
  background: rgba(255, 59, 48, 0.9);
  border: none;
  border-radius: 50%;
  color: #fff;
  font-size: 12px;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.thumbnail-item:hover .remove-img-btn {
  opacity: 1;
}

.center-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  min-width: 0;
}

.character-editor {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

.editor-header {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px 24px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  flex-shrink: 0;
}

.back-btn {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 8px 12px;
  background: rgba(255, 255, 255, 0.1);
  border: none;
  border-radius: 6px;
  color: rgba(255, 255, 255, 0.8);
  font-size: 0.85rem;
  cursor: pointer;
  transition: all 0.3s ease;
}

.back-btn:hover {
  background: rgba(255, 255, 255, 0.15);
  color: #fff;
}

.editor-header h3 {
  margin: 0;
  font-size: 1.25rem;
  color: #00d4ff;
}

.editor-content {
  flex: 1;
  overflow-y: auto;
  padding: 20px 24px;
}

.form-group {
  margin-bottom: 20px;
}

.form-group label {
  display: block;
  margin-bottom: 8px;
  font-size: 0.9rem;
  font-weight: 500;
  color: rgba(255, 255, 255, 0.8);
}

.form-group input,
.form-group textarea,
.form-group select {
  width: 100%;
  padding: 12px 16px;
  background: rgba(26, 26, 46, 0.8);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 8px;
  color: #fff;
  font-size: 0.95rem;
  transition: all 0.3s ease;
  font-family: inherit;
}

.form-group textarea {
  resize: vertical;
  min-height: 80px;
}

.form-group input:focus,
.form-group textarea:focus,
.form-group select:focus {
  outline: none;
  border-color: #00d4ff;
  box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1);
}

.form-group input::placeholder,
.form-group textarea::placeholder {
  color: rgba(255, 255, 255, 0.3);
}

.form-group select {
  cursor: pointer;
}

.form-group select option {
  background: #1a1a2e;
  color: #fff;
}

.runninghub-pic-block {
  padding: 16px;
  border-radius: 12px;
  border: 1px solid rgba(0, 212, 255, 0.22);
  background: rgba(0, 212, 255, 0.06);
}

.stand-mode-options {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px 16px;
  margin-bottom: 12px;
  font-size: 0.88rem;
  color: rgba(255, 255, 255, 0.85);
}

.stand-mode-label {
  font-weight: 600;
  color: rgba(255, 255, 255, 0.9);
}

.stand-mode-radio {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
}

.stand-mode-radio input {
  width: auto;
  margin: 0;
}

.stand-custom-editor {
  margin-bottom: 14px;
  padding: 12px;
  border-radius: 8px;
  background: rgba(0, 0, 0, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.stand-custom-hint {
  margin: 0 0 10px 0;
  font-size: 0.75rem;
  line-height: 1.5;
  color: rgba(255, 255, 255, 0.45);
}

.stand-custom-hint code {
  font-size: 0.72rem;
  padding: 1px 5px;
  border-radius: 4px;
  background: rgba(0, 0, 0, 0.35);
  color: #a8e6ff;
}

.stand-custom-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 8px;
  align-items: center;
}

.stand-custom-input {
  flex: 1;
  min-width: 120px;
  padding: 8px 10px;
  background: rgba(26, 26, 46, 0.9);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 6px;
  color: #fff;
  font-size: 0.88rem;
}

.stand-custom-remove {
  padding: 6px 10px;
  border-radius: 6px;
  border: 1px solid rgba(255, 120, 120, 0.35);
  background: rgba(80, 20, 20, 0.35);
  color: #ffb4b4;
  font-size: 0.8rem;
  cursor: pointer;
}

.stand-custom-remove:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

.stand-custom-add {
  margin-top: 4px;
  padding: 6px 12px;
  border-radius: 6px;
  border: 1px solid rgba(0, 212, 255, 0.35);
  background: rgba(0, 40, 60, 0.4);
  color: #7ee8ff;
  font-size: 0.82rem;
  cursor: pointer;
}

.runninghub-pic-hint {
  margin: 0 0 12px 0;
  font-size: 0.78rem;
  line-height: 1.55;
  color: rgba(255, 255, 255, 0.48);
}

.runninghub-pic-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
}

.runninghub-pic-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 18px;
  border-radius: 8px;
  border: 1px solid rgba(123, 44, 191, 0.45);
  background: linear-gradient(135deg, rgba(123, 44, 191, 0.28) 0%, rgba(0, 212, 255, 0.14) 100%);
  color: #e8e8ff;
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.runninghub-pic-btn:hover:not(:disabled) {
  border-color: rgba(0, 212, 255, 0.55);
  box-shadow: 0 4px 14px rgba(0, 212, 255, 0.12);
}

.runninghub-pic-btn:disabled {
  opacity: 0.65;
  cursor: not-allowed;
}

.runninghub-pic-btn-secondary {
  border-color: rgba(0, 212, 255, 0.35);
  background: linear-gradient(135deg, rgba(0, 212, 255, 0.12) 0%, rgba(123, 44, 191, 0.1) 100%);
}

.form-row {
  display: flex;
  gap: 16px;
}

.form-row .form-group {
  flex: 1;
}

.metadata-group {
  background: rgba(26, 26, 46, 0.4);
  padding: 16px;
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.05);
}

.metadata-info {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.metadata-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.meta-label {
  color: rgba(255, 255, 255, 0.5);
  font-size: 0.85rem;
}

.meta-value {
  color: rgba(255, 255, 255, 0.9);
  font-size: 0.9rem;
}

.tags-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.tag {
  padding: 4px 10px;
  background: rgba(0, 212, 255, 0.15);
  border-radius: 12px;
  font-size: 0.8rem;
  color: #00d4ff;
}

.no-character-selected {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
}

.empty-state {
  text-align: center;
  color: rgba(255, 255, 255, 0.5);
}

.empty-icon {
  font-size: 4rem;
  display: block;
  margin-bottom: 16px;
}

.empty-state p {
  margin: 0 0 20px 0;
  font-size: 1.1rem;
}

.create-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 12px 24px;
  background: linear-gradient(135deg, #00d4ff 0%, #7b2cbf 100%);
  border: none;
  border-radius: 10px;
  color: #fff;
  font-size: 1rem;
  cursor: pointer;
  transition: all 0.3s ease;
}

.create-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(0, 212, 255, 0.3);
}

.right-panel {
  width: 280px;
  min-width: 240px;
  border-left: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  flex-direction: column;
  background: rgba(15, 15, 26, 0.5);
}

.character-list-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  flex-shrink: 0;
}

.character-list-header h3 {
  margin: 0;
  font-size: 1rem;
  color: rgba(255, 255, 255, 0.9);
}

.add-character-btn {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 6px 12px;
  background: rgba(0, 212, 255, 0.2);
  border: 1px solid rgba(0, 212, 255, 0.3);
  border-radius: 6px;
  color: #00d4ff;
  font-size: 0.85rem;
  cursor: pointer;
  transition: all 0.3s ease;
}

.add-character-btn:hover {
  background: rgba(0, 212, 255, 0.3);
}

.character-list {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
}

.character-item {
  display: flex;
  flex-direction: column;
  background: rgba(26, 26, 46, 0.6);
  border-radius: 10px;
  margin-bottom: 8px;
  border: 1px solid transparent;
  transition: all 0.3s ease;
  overflow: hidden;
}

.character-item:hover {
  background: rgba(26, 26, 46, 0.9);
}

.character-item.active {
  background: rgba(0, 212, 255, 0.15);
  border-color: rgba(0, 212, 255, 0.3);
}

.char-main {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px;
  cursor: pointer;
}

.char-avatar {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  overflow: hidden;
  background: linear-gradient(135deg, #00d4ff 0%, #7b2cbf 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.char-avatar img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.avatar-placeholder {
  font-size: 1.25rem;
  font-weight: 600;
  color: #fff;
}

.char-info {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.char-name {
  font-size: 0.95rem;
  font-weight: 500;
  color: #fff;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.char-meta {
  font-size: 0.8rem;
  color: rgba(255, 255, 255, 0.5);
}

.char-actions {
  display: flex;
  gap: 4px;
  padding: 8px 12px;
  border-top: 1px solid rgba(255, 255, 255, 0.05);
  justify-content: flex-end;
}

.edit-btn,
.delete-btn {
  padding: 6px 10px;
  background: transparent;
  border: none;
  cursor: pointer;
  opacity: 0.6;
  transition: all 0.2s;
  font-size: 0.9rem;
  border-radius: 4px;
}

.edit-btn:hover {
  opacity: 1;
  background: rgba(0, 212, 255, 0.2);
}

.delete-btn:hover {
  opacity: 1;
  background: rgba(255, 59, 48, 0.2);
}

.empty-list {
  text-align: center;
  padding: 40px 20px;
  color: rgba(255, 255, 255, 0.4);
}

.empty-list p {
  margin: 0 0 16px 0;
}

.create-btn-small {
  padding: 8px 16px;
  background: rgba(0, 212, 255, 0.2);
  border: 1px solid rgba(0, 212, 255, 0.3);
  border-radius: 6px;
  color: #00d4ff;
  font-size: 0.85rem;
  cursor: pointer;
  transition: all 0.3s ease;
}

.create-btn-small:hover {
  background: rgba(0, 212, 255, 0.3);
}
</style>
