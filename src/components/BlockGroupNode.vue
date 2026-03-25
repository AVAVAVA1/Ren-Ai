<script setup>
import { inject } from 'vue'

const props = defineProps({
  id: { type: String, required: true },
  data: { type: Object, default: () => ({}) },
  selected: { type: Boolean, default: false }
})

const toggleBlock = inject('flowToggleBlock', () => {})
const openBlockContextMenu = inject('flowOpenBlockContextMenu', () => {})

function onHeaderContextMenu(e) {
  const gi = props.data?.groupIndex
  if (gi === undefined || gi === null) return
  e.preventDefault()
  e.stopPropagation()
  openBlockContextMenu(gi, e)
}
</script>

<template>
  <div
    class="block-group-root"
    :class="{ selected, 'is-collapsed': data?.collapsed }"
  >
    <div class="block-group-chrome">
      <div class="block-group-header" @pointerdown.stop @contextmenu="onHeaderContextMenu">
        <span class="block-title">{{ data?.dialogue_name || '区块' }}</span>
        <button
          type="button"
          class="collapse-btn"
          @click.stop="toggleBlock(data?.groupIndex)"
          @pointerdown.stop
        >
          {{ data?.collapsed ? '展开' : '收起' }}
        </button>
      </div>
      <div v-if="!data?.collapsed" class="block-group-hint">拖拽边框区域可整体移动本区块</div>
    </div>
  </div>
</template>

<style scoped>
.block-group-root {
  width: 100%;
  height: 100%;
  min-height: 48px;
  box-sizing: border-box;
  border: 3px solid #f5c400;
  border-radius: 12px;
  background: rgba(245, 196, 0, 0.06);
  box-shadow: 0 0 0 1px rgba(245, 196, 0, 0.25) inset, 0 8px 28px rgba(0, 0, 0, 0.35);
  position: relative;
  /* 不挡住子对话节点的点击与拖拽；仅标题栏可交互（与 FlowCanvas dragHandle 一致） */
  pointer-events: none;
}

.block-group-root.selected {
  border-color: #ffe066;
  box-shadow: 0 0 0 2px rgba(255, 224, 102, 0.45), 0 8px 32px rgba(245, 196, 0, 0.2);
}

.block-group-root.is-collapsed {
  background: rgba(245, 196, 0, 0.12);
}

.block-group-chrome {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 8px 10px 6px;
  pointer-events: none;
}

.block-group-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  pointer-events: auto;
  cursor: grab;
}

.block-group-header:active {
  cursor: grabbing;
}

.block-title {
  font-size: 0.78rem;
  font-weight: 600;
  color: #ffe066;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
  min-width: 0;
}

.collapse-btn {
  flex-shrink: 0;
  padding: 4px 10px;
  font-size: 0.7rem;
  border-radius: 6px;
  border: 1px solid rgba(245, 196, 0, 0.55);
  background: rgba(0, 0, 0, 0.35);
  color: #ffe066;
  cursor: pointer;
}

.collapse-btn:hover {
  background: rgba(245, 196, 0, 0.2);
}

.block-group-hint {
  font-size: 0.62rem;
  color: rgba(255, 255, 255, 0.35);
  pointer-events: none;
}
</style>
