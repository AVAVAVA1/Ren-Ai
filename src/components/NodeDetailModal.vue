<script setup>
import { ref, computed, watch } from 'vue'

const props = defineProps({
  node: {
    type: Object,
    required: true
  }
})

const emit = defineEmits(['close', 'update'])

const editedId = ref('')
const editedName = ref('')
const editedContent = ref('')
const editedBackground = ref('')
const editedCharacter = ref('')
const editedMusic = ref('')
const editedSound = ref('')
const editedTransition = ref('')
const editedMenu = ref([])
const editedSetOrChangeFlag = ref('')
const editedCheckFlag = ref({})

watch(() => props.node, (newNode) => {
  if (newNode) {
    editedId.value = newNode.id || ''
    editedName.value = newNode.data?.name || ''
    editedContent.value = newNode.data?.content || ''
    editedBackground.value = newNode.data?.background || ''
    editedCharacter.value = newNode.data?.character || ''
    editedMusic.value = newNode.data?.music || ''
    editedSound.value = newNode.data?.sound || ''
    editedTransition.value = newNode.data?.transition || ''
    editedMenu.value = Array.isArray(newNode.data?.menu) 
      ? newNode.data.menu.map(item => {
          if (typeof item === 'object' && item !== null) {
            return { content: item.content || '', flag: item.flag || '' }
          }
          return { content: item || '', flag: '' }
        }) 
      : []
    editedSetOrChangeFlag.value = newNode.data?.setOrChangeFlag || ''
    
    const checkFlagData = newNode.data?.checkFlag || {}
    if (typeof checkFlagData === 'object' && !Array.isArray(checkFlagData)) {
      editedCheckFlag.value = { ...checkFlagData }
    } else if (typeof checkFlagData === 'string' && checkFlagData) {
      editedCheckFlag.value = {}
    } else {
      editedCheckFlag.value = {}
    }
  }
}, { immediate: true })

const branchNum = computed(() => props.node?.data?.branch_num || 0)
const parentId = computed(() => props.node?.data?.parent_id || '无')
const children = computed(() => props.node?.data?.children || [])

const canEditCheckFlag = computed(() => branchNum.value > 1)

function addMenuItem() {
  editedMenu.value.push({ content: '', flag: '' })
}

function removeMenuItem(index) {
  editedMenu.value.splice(index, 1)
}

function handleClose() {
  emit('close')
}

function handleOverlayClick(event) {
  if (event.target === event.currentTarget) {
    handleClose()
  }
}

function handleSave() {
  let checkFlagValue = {}
  if (canEditCheckFlag.value) {
    children.value.forEach(childId => {
      if (editedCheckFlag.value[childId] !== undefined) {
        checkFlagValue[childId] = editedCheckFlag.value[childId]
      } else {
        checkFlagValue[childId] = ''
      }
    })
  }
  
  const menuItems = editedMenu.value
    .filter(item => item.content !== '')
    .map(item => ({ content: item.content, flag: item.flag || '' }))
  
  emit('update', {
    oldId: props.node.id,
    newId: editedId.value,
    name: editedName.value,
    content: editedContent.value,
    background: editedBackground.value,
    character: editedCharacter.value,
    music: editedMusic.value,
    sound: editedSound.value,
    transition: editedTransition.value,
    menu: menuItems,
    setOrChangeFlag: editedSetOrChangeFlag.value,
    checkFlag: checkFlagValue
  })
  handleClose()
}
</script>

<template>
  <div class="modal-overlay" @click="handleOverlayClick">
    <div class="modal-container">
      <div class="modal-header">
        <h3 class="modal-title">节点详情</h3>
        <button class="close-btn" @click="handleClose">×</button>
      </div>
      
      <div class="modal-body">
        <div class="info-row">
          <label class="info-label">节点ID:</label>
          <input 
            type="text" 
            class="edit-input" 
            v-model="editedId"
            placeholder="输入节点ID"
          />
        </div>
        
        <div class="info-row">
          <label class="info-label">角色名称:</label>
          <input 
            type="text" 
            class="edit-input highlight" 
            v-model="editedName"
            placeholder="输入角色名称"
          />
        </div>
        
        <div class="info-row">
          <label class="info-label">对话内容:</label>
          <textarea 
            class="edit-textarea" 
            v-model="editedContent"
            placeholder="输入对话内容"
            rows="3"
          ></textarea>
        </div>
        
        <div class="info-row">
          <label class="info-label">背景信息:</label>
          <input 
            type="text" 
            class="edit-input" 
            v-model="editedBackground"
            placeholder="输入背景信息"
          />
        </div>
        
        <div class="info-row">
          <label class="info-label">人物信息:</label>
          <input 
            type="text" 
            class="edit-input" 
            v-model="editedCharacter"
            placeholder="输入人物信息"
          />
        </div>
        
        <div class="info-row">
          <label class="info-label">音乐信息:</label>
          <input 
            type="text" 
            class="edit-input" 
            v-model="editedMusic"
            placeholder="输入音乐信息"
          />
        </div>
        
        <div class="info-row">
          <label class="info-label">音效信息:</label>
          <input 
            type="text" 
            class="edit-input" 
            v-model="editedSound"
            placeholder="输入音效信息"
          />
        </div>
        
        <div class="info-row">
          <label class="info-label">转场方式:</label>
          <input 
            type="text" 
            class="edit-input" 
            v-model="editedTransition"
            placeholder="输入转场方式"
          />
        </div>
        
        <div class="info-row menu-row">
          <div class="menu-header">
            <label class="info-label">Menu:</label>
            <button class="add-menu-btn" @click="addMenuItem" type="button">+ 添加</button>
          </div>
          <div class="menu-list" v-if="editedMenu.length > 0">
            <div 
              v-for="(item, index) in editedMenu" 
              :key="index" 
              class="menu-item-row"
            >
              <div class="menu-item-fields">
                <input 
                  type="text" 
                  class="edit-input menu-content-input" 
                  v-model="item.content"
                  :placeholder="`菜单项 ${index + 1}`"
                />
                <input 
                  type="text" 
                  class="edit-input menu-flag-input" 
                  v-model="item.flag"
                  placeholder="flag"
                />
              </div>
              <button class="remove-menu-btn" @click="removeMenuItem(index)" type="button">×</button>
            </div>
          </div>
          <div class="menu-empty" v-else>
            <span class="empty-hint">暂无菜单项，点击"添加"创建</span>
          </div>
        </div>
        
        <div class="info-row">
          <label class="info-label">SetOrChangeFlag:</label>
          <input 
            type="text" 
            class="edit-input" 
            v-model="editedSetOrChangeFlag"
            placeholder="输入SetOrChangeFlag"
          />
        </div>
        
        <div class="info-row checkflag-row" :class="{ 'readonly': !canEditCheckFlag }">
          <div class="checkflag-header">
            <label class="info-label">
              CheckFlag:
              <span v-if="!canEditCheckFlag" class="readonly-hint">(分支数>1时可编辑)</span>
            </label>
          </div>
          <div class="checkflag-list" v-if="canEditCheckFlag && children.length > 0">
            <div 
              v-for="childId in children" 
              :key="childId" 
              class="checkflag-item-row"
            >
              <span class="checkflag-label">{{ childId }}</span>
              <input 
                type="text" 
                class="edit-input checkflag-input" 
                v-model="editedCheckFlag[childId]"
                :placeholder="`flag for ${childId}`"
              />
            </div>
          </div>
          <div class="checkflag-empty" v-else>
            <span class="empty-hint">{{ canEditCheckFlag ? '无分支节点' : '分支数≤1时不可编辑' }}</span>
          </div>
        </div>
        
        <div class="info-row readonly">
          <span class="info-label">分支数量:</span>
          <span class="info-value readonly-value">{{ branchNum }}</span>
          <span class="readonly-hint">(自动计算)</span>
        </div>
        
        <div class="info-row readonly">
          <span class="info-label">父节点ID:</span>
          <span class="info-value readonly-value">{{ parentId }}</span>
          <span class="readonly-hint">(自动维护)</span>
        </div>
        
        <div class="info-row readonly" v-if="children.length > 0">
          <span class="info-label">子节点ID:</span>
          <div class="children-list">
            <span
              v-for="childId in children"
              :key="childId"
              class="child-tag"
            >
              {{ childId }}
            </span>
          </div>
        </div>
      </div>
      
      <div class="modal-footer">
        <button class="btn btn-secondary" @click="handleClose">取消</button>
        <button class="btn btn-primary" @click="handleSave">保存</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(4px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10000;
}

.modal-container {
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  width: 90%;
  max-width: 500px;
  max-height: 85vh;
  overflow: hidden;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px 24px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.modal-title {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 600;
  color: #fff;
}

.close-btn {
  width: 32px;
  height: 32px;
  border: none;
  background: rgba(255, 255, 255, 0.1);
  color: rgba(255, 255, 255, 0.7);
  font-size: 1.5rem;
  line-height: 1;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.close-btn:hover {
  background: rgba(255, 71, 87, 0.2);
  color: #ff4757;
}

.modal-body {
  padding: 24px;
  overflow-y: auto;
  max-height: calc(85vh - 140px);
}

.info-row {
  margin-bottom: 14px;
}

.info-row:last-child {
  margin-bottom: 0;
}

.info-row.readonly {
  background: rgba(0, 0, 0, 0.2);
  padding: 10px 12px;
  border-radius: 8px;
  margin-left: -12px;
  margin-right: -12px;
  padding-left: 12px;
  padding-right: 12px;
}

.info-label {
  display: block;
  font-size: 0.8rem;
  color: rgba(255, 255, 255, 0.5);
  margin-bottom: 4px;
}

.edit-input {
  width: 100%;
  padding: 8px 12px;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 8px;
  font-size: 0.9rem;
  color: rgba(255, 255, 255, 0.9);
  transition: all 0.2s ease;
  box-sizing: border-box;
}

.edit-input:focus {
  outline: none;
  border-color: #00d4ff;
  box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.2);
}

.edit-input.highlight {
  color: #00d4ff;
  font-weight: 600;
}

.edit-textarea {
  width: 100%;
  padding: 10px 12px;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 8px;
  font-size: 0.9rem;
  color: rgba(255, 255, 255, 0.85);
  line-height: 1.5;
  resize: vertical;
  min-height: 80px;
  transition: all 0.2s ease;
  box-sizing: border-box;
  font-family: inherit;
}

.edit-textarea:focus {
  outline: none;
  border-color: #00d4ff;
  box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.2);
}

.menu-row {
  background: rgba(0, 0, 0, 0.2);
  padding: 12px;
  border-radius: 8px;
  margin-left: -12px;
  margin-right: -12px;
  padding-left: 12px;
  padding-right: 12px;
}

.menu-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}

.menu-header .info-label {
  margin-bottom: 0;
}

.add-menu-btn {
  padding: 4px 12px;
  background: rgba(0, 212, 255, 0.2);
  border: 1px solid rgba(0, 212, 255, 0.4);
  border-radius: 6px;
  color: #00d4ff;
  font-size: 0.75rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.add-menu-btn:hover {
  background: rgba(0, 212, 255, 0.3);
  border-color: rgba(0, 212, 255, 0.6);
}

.menu-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.menu-item-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.menu-item-fields {
  flex: 1;
  display: flex;
  gap: 8px;
}

.menu-content-input {
  flex: 2;
}

.menu-flag-input {
  flex: 1;
  background: rgba(123, 44, 191, 0.15);
  border-color: rgba(123, 44, 191, 0.3);
}

.menu-flag-input:focus {
  border-color: #b366e9;
  box-shadow: 0 0 0 3px rgba(123, 44, 191, 0.2);
}

.menu-flag-input::placeholder {
  color: rgba(179, 102, 233, 0.5);
}

.remove-menu-btn {
  width: 28px;
  height: 28px;
  border: none;
  background: rgba(255, 71, 87, 0.2);
  color: #ff4757;
  font-size: 1rem;
  line-height: 1;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
  flex-shrink: 0;
}

.remove-menu-btn:hover {
  background: rgba(255, 71, 87, 0.4);
}

.menu-empty {
  padding: 12px;
  text-align: center;
}

.empty-hint {
  font-size: 0.8rem;
  color: rgba(255, 255, 255, 0.3);
}

.checkflag-row {
  background: rgba(0, 0, 0, 0.2);
  padding: 12px;
  border-radius: 8px;
  margin-left: -12px;
  margin-right: -12px;
  padding-left: 12px;
  padding-right: 12px;
}

.checkflag-header {
  margin-bottom: 8px;
}

.checkflag-header .info-label {
  margin-bottom: 0;
}

.checkflag-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.checkflag-item-row {
  display: flex;
  align-items: center;
  gap: 10px;
}

.checkflag-label {
  min-width: 80px;
  padding: 4px 10px;
  background: rgba(50, 205, 50, 0.15);
  border: 1px solid rgba(50, 205, 50, 0.4);
  border-radius: 6px;
  font-size: 0.8rem;
  color: #32cd32;
  text-align: center;
  flex-shrink: 0;
}

.checkflag-input {
  flex: 1;
  background: rgba(255, 165, 0, 0.1);
  border-color: rgba(255, 165, 0, 0.3);
}

.checkflag-input:focus {
  border-color: #ffa500;
  box-shadow: 0 0 0 3px rgba(255, 165, 0, 0.2);
}

.checkflag-input::placeholder {
  color: rgba(255, 165, 0, 0.4);
}

.checkflag-empty {
  padding: 12px;
  text-align: center;
}

.empty-value {
  color: rgba(255, 255, 255, 0.3);
}

.info-value {
  font-size: 0.9rem;
  color: rgba(255, 255, 255, 0.9);
}

.readonly-value {
  color: rgba(255, 255, 255, 0.6);
}

.readonly-hint {
  font-size: 0.7rem;
  color: rgba(255, 255, 255, 0.3);
  margin-left: 8px;
}

.children-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.child-tag {
  display: inline-block;
  padding: 3px 10px;
  background: rgba(123, 44, 191, 0.2);
  border: 1px solid rgba(123, 44, 191, 0.4);
  border-radius: 20px;
  font-size: 0.8rem;
  color: #b366e9;
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  padding: 16px 24px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.btn {
  padding: 10px 24px;
  border: none;
  border-radius: 8px;
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.btn-secondary {
  background: rgba(255, 255, 255, 0.1);
  color: rgba(255, 255, 255, 0.7);
}

.btn-secondary:hover {
  background: rgba(255, 255, 255, 0.15);
  color: #fff;
}

.btn-primary {
  background: linear-gradient(135deg, #00d4ff 0%, #7b2cbf 100%);
  color: #fff;
}

.btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4);
}
</style>
