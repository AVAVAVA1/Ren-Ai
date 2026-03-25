<script setup>
import { ref } from 'vue'

const settings = ref({
  theme: 'dark',
  language: 'zh-CN',
  autoSave: true,
  snapToGrid: true,
  gridSize: 20
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
