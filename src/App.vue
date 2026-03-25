<script setup>
import { ref, nextTick } from 'vue'
import NavBar from './components/NavBar.vue'
import FlowCanvas from './components/FlowCanvas.vue'
import CharacterPage from './components/CharacterPage.vue'
import StoryPage from './components/StoryPage.vue'
import SettingsPage from './components/SettingsPage.vue'

const currentTab = ref('flow')
const flowCanvasRef = ref(null)
const fileInputRef = ref(null)

function handleNavigate(tab) {
  currentTab.value = tab
}

function handleImport() {
  fileInputRef.value?.click()
}

function handleFileSelect(event) {
  const file = event.target.files?.[0]
  if (file) {
    if (currentTab.value === 'flow') {
      flowCanvasRef.value?.handleImportJson(file)
    }
    event.target.value = ''
  }
}

function handleExport() {
  flowCanvasRef.value?.handleExportJson()
}

async function handleStoryOpenFlow(publicUrl) {
  currentTab.value = 'flow'
  await nextTick()
  flowCanvasRef.value?.loadFromPublicUrl(publicUrl)
}
</script>

<template>
  <div class="app-container">
    <NavBar
      @navigate="handleNavigate"
      @import="handleImport"
      @export="handleExport"
    />
    
    <input
      ref="fileInputRef"
      type="file"
      accept=".json"
      style="display: none"
      @change="handleFileSelect"
    />
    
    <main class="main-content">
      <FlowCanvas
        v-show="currentTab === 'flow'"
        ref="flowCanvasRef"
      />
      <CharacterPage v-show="currentTab === 'character'" />
      <StoryPage v-show="currentTab === 'story'" @open-flow="handleStoryOpenFlow" />
      <SettingsPage v-show="currentTab === 'settings'" />
    </main>
  </div>
</template>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
  background: #0f0f1a;
  color: #fff;
  overflow: hidden;
}

.app-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100vw;
}

.main-content {
  flex: 1;
  overflow: hidden;
}
</style>
