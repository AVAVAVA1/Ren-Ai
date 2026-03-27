<script setup>
defineProps({
  activeTab: {
    type: String,
    default: 'flow'
  }
})

const emit = defineEmits(['navigate', 'import', 'export'])

const navItems = [
  { id: 'character', label: '人物卡' },
  { id: 'flow', label: '流程图' },
  { id: 'story', label: '故事生成' },
  { id: 'settings', label: '设置' }
]

function handleNavClick(id) {
  emit('navigate', id)
}

function handleImportClick() {
  emit('import')
}

function handleExportClick() {
  emit('export')
}
</script>

<template>
  <nav class="navbar">
    <div class="nav-brand">
      <span class="brand-text">RenAI Flow</span>
    </div>
    
    <div class="nav-links">
      <button
        v-for="item in navItems"
        :key="item.id"
        :class="['nav-item', { active: activeTab === item.id }]"
        @click="handleNavClick(item.id)"
      >
        {{ item.label }}
      </button>
    </div>
    
    <div class="nav-actions">
      <button class="action-btn import-btn" @click="handleImportClick">
        <span class="btn-icon">📁</span>
        导入JSON
      </button>
      <button class="action-btn export-btn" @click="handleExportClick">
        <span class="btn-icon">💾</span>
        导出JSON
      </button>
    </div>
  </nav>
</template>

<style scoped>
.navbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  height: 60px;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
}

.nav-brand {
  display: flex;
  align-items: center;
}

.brand-text {
  font-size: 1.5rem;
  font-weight: 700;
  background: linear-gradient(90deg, #00d4ff, #7b2cbf);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.nav-links {
  display: flex;
  gap: 8px;
}

.nav-item {
  padding: 10px 24px;
  border: none;
  background: transparent;
  color: rgba(255, 255, 255, 0.7);
  font-size: 0.95rem;
  cursor: pointer;
  border-radius: 8px;
  transition: all 0.3s ease;
  position: relative;
}

.nav-item:hover {
  color: #fff;
  background: rgba(255, 255, 255, 0.1);
}

.nav-item.active {
  color: #00d4ff;
  background: rgba(0, 212, 255, 0.15);
}

.nav-item.active::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 50%;
  transform: translateX(-50%);
  width: 30px;
  height: 3px;
  background: #00d4ff;
  border-radius: 3px;
}

.nav-actions {
  display: flex;
  gap: 10px;
}

.action-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  font-size: 0.85rem;
  cursor: pointer;
  transition: all 0.3s ease;
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

.btn-icon {
  font-size: 1rem;
}
</style>
