<script setup>
import { ref, computed } from 'vue'
import { Handle, Position } from '@vue-flow/core'

const props = defineProps({
  id: String,
  data: Object,
  selected: Boolean
})

const emit = defineEmits(['show-menu', 'delete', 'copy'])

const showMenu = ref(false)
const menuPosition = ref({ x: 0, y: 0 })

const nodeStyle = computed(() => {
  const colors = {
    '旁白': { bg: 'rgba(100, 149, 237, 0.15)', border: 'rgba(100, 149, 237, 0.6)', text: '#6495ed' },
    '艾略特·雷恩': { bg: 'rgba(50, 205, 50, 0.15)', border: 'rgba(50, 205, 50, 0.6)', text: '#32cd32' },
    '玛拉': { bg: 'rgba(255, 105, 180, 0.15)', border: 'rgba(255, 105, 180, 0.6)', text: '#ff69b4' },
    '新角色': { bg: 'rgba(255, 165, 0, 0.15)', border: 'rgba(255, 165, 0, 0.6)', text: '#ffa500' }
  }
  const name = props.data?.name || ''
  return colors[name] || { bg: 'rgba(148, 163, 184, 0.15)', border: 'rgba(148, 163, 184, 0.6)', text: '#94a3b8' }
})

function handleContextMenu(event) {
  event.preventDefault()
  showMenu.value = true
  menuPosition.value = { x: event.clientX, y: event.clientY }
}

function handleLeftClick(event) {
  if (event.target.closest('.vue-flow__handle')) return
  event.stopPropagation()
  showMenu.value = true
  const rect = event.target.closest('.dialogue-node').getBoundingClientRect()
  menuPosition.value = { x: rect.left + rect.width / 2, y: rect.top + rect.height }
}

function closeMenu() {
  showMenu.value = false
}

function handleViewDetails() {
  emit('show-menu', { type: 'view', nodeId: props.id, data: props.data })
  closeMenu()
}

function handleDelete() {
  emit('delete', props.id)
  closeMenu()
}

function handleCopy() {
  emit('copy', { nodeId: props.id, data: props.data })
  closeMenu()
}

function handleClickOutside(event) {
  if (!event.target.closest('.node-menu') && !event.target.closest('.dialogue-node')) {
    showMenu.value = false
  }
}
</script>

<template>
  <div
    :class="['dialogue-node', { selected }]"
    :style="{
      background: nodeStyle.bg,
      borderColor: nodeStyle.border
    }"
    @contextmenu="handleContextMenu"
    @click="handleLeftClick"
  >
    <Handle
      type="target"
      :position="Position.Top"
      class="handle target-handle"
      :is-connectable="true"
      :is-connectable-start="true"
      :is-connectable-end="true"
    />
    
    <div class="node-header" :style="{ color: nodeStyle.text }">
      <span class="node-id">#{{ id }}</span>
      <span class="node-name">{{ data?.name || '未命名' }}</span>
    </div>
    <div v-if="data?._blockTitle" class="node-block-tag">{{ data._blockTitle }}</div>
    
    <div class="node-content">
      {{ data?.content || '无内容' }}
    </div>
    
    <div class="node-footer">
      <span class="branch-info" v-if="data?.branch_num">
        分支: {{ data.branch_num }}
      </span>
    </div>
    
    <Handle
      type="source"
      :position="Position.Bottom"
      class="handle source-handle"
      :is-connectable="true"
      :is-connectable-start="true"
      :is-connectable-end="true"
    />
    
    <Teleport to="body">
      <div
        v-if="showMenu"
        class="node-menu-overlay"
        @click="handleClickOutside"
      >
        <div
          class="node-menu"
          :style="{
            left: menuPosition.x + 'px',
            top: menuPosition.y + 'px'
          }"
        >
          <button class="menu-item view-btn" @click="handleViewDetails">
            <span class="menu-icon">👁️</span>
            查看详细内容
          </button>
          <button class="menu-item copy-btn" @click="handleCopy">
            <span class="menu-icon">📋</span>
            复制节点
          </button>
          <button class="menu-item delete-btn" @click="handleDelete">
            <span class="menu-icon">🗑️</span>
            删除节点
          </button>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
.dialogue-node {
  min-width: 200px;
  max-width: 280px;
  padding: 12px 16px;
  border-radius: 12px;
  border: 2px solid;
  background: rgba(30, 41, 59, 0.9);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
  cursor: pointer;
  transition: all 0.2s ease;
  position: relative;
}

.dialogue-node:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 25px rgba(0, 0, 0, 0.4);
}

.dialogue-node.selected {
  box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.5), 0 6px 25px rgba(0, 0, 0, 0.4);
}

.node-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  font-weight: 600;
}

.node-id {
  font-size: 0.75rem;
  opacity: 0.7;
}

.node-name {
  font-size: 0.9rem;
}

.node-block-tag {
  font-size: 0.65rem;
  color: rgba(255, 255, 255, 0.45);
  margin: -4px 0 8px;
  line-height: 1.3;
  max-height: 2.6em;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

.node-content {
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.85);
  line-height: 1.5;
  word-break: break-word;
}

.node-footer {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.branch-info {
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.5);
}

.handle {
  width: 18px;
  height: 18px;
  border: 3px solid #1a1a2e;
  border-radius: 50%;
  transition: all 0.2s ease;
  cursor: crosshair;
}

.target-handle {
  top: -9px;
  background: #ff69b4;
}

.target-handle:hover {
  transform: scale(1.3);
  background: #7b2cbf;
  box-shadow: 0 0 15px rgba(255, 105, 180, 0.8);
}

.source-handle {
  bottom: -9px;
  background: #32cd32;
}

.source-handle:hover {
  transform: scale(1.3);
  background: #7b2cbf;
  box-shadow: 0 0 15px rgba(50, 205, 50, 0.8);
}

.node-menu-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 9999;
}

.node-menu {
  position: fixed;
  transform: translateX(-50%);
  background: rgba(26, 26, 46, 0.98);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  padding: 8px 0;
  min-width: 160px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
  z-index: 10000;
}

.menu-item {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 10px 16px;
  border: none;
  background: transparent;
  color: rgba(255, 255, 255, 0.85);
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.2s ease;
  text-align: left;
}

.menu-item:hover {
  background: rgba(255, 255, 255, 0.1);
}

.view-btn:hover {
  color: #00d4ff;
}

.copy-btn:hover {
  color: #32cd32;
}

.delete-btn:hover {
  color: #ff4757;
  background: rgba(255, 71, 87, 0.15);
}

.menu-icon {
  font-size: 1rem;
}
</style>
