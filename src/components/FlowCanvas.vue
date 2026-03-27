<script setup>
import { ref, watch, onMounted, onUnmounted, computed, markRaw, provide } from 'vue'
import { VueFlow, useVueFlow, ConnectionMode } from '@vue-flow/core'
import { Background } from '@vue-flow/background'
import { Controls } from '@vue-flow/controls'
import { MiniMap } from '@vue-flow/minimap'
import DialogueNode from './DialogueNode.vue'
import BlockGroupNode from './BlockGroupNode.vue'
import NodeDetailModal from './NodeDetailModal.vue'
import dagre from 'dagre'

import '@vue-flow/core/dist/style.css'
import '@vue-flow/core/dist/theme-default.css'
import '@vue-flow/controls/dist/style.css'
import '@vue-flow/minimap/dist/style.css'

const emit = defineEmits(['export-data'])

const API_BASE_URL = 'http://localhost:8000'

const DEFAULT_BG_WORKFLOW_ID = '2037179226444533762'
const isGeneratingFlowBg = ref(false)
const isLaunchingPygame = ref(false)

/** 与 public/sources/dialogue 下 dialogue_*.json 一致：chapter_name、site、dialogues[] */
function isDialogueScriptImportShape(data) {
  const arr = Array.isArray(data) ? data : [data]
  if (arr.length === 0) return false
  return arr.every(
    (item) =>
      item &&
      typeof item === 'object' &&
      Object.prototype.hasOwnProperty.call(item, 'chapter_name') &&
      Object.prototype.hasOwnProperty.call(item, 'dialogues') &&
      Array.isArray(item.dialogues)
  )
}

const {
  onConnect,
  addEdges,
  removeEdges,
  project,
  vueFlowRef,
  screenToFlowCoordinate,
  onConnectStart,
  onConnectEnd
} = useVueFlow()

const nodes = ref([])
const edges = ref([])
const selectedNode = ref(null)
const showModal = ref(false)
const mousePosition = ref({ x: 0, y: 0 })
const nodeIdCounter = ref(0)
const dragStartInfo = ref(null)
const connectionMade = ref(false)
const isUpdating = ref(false)
const copiedNode = ref(null)
const showContextMenu = ref(false)
const contextMenuPosition = ref({ x: 0, y: 0 })
/** 非 null 表示当前菜单由区块标题右键打开，对应 flowGroups 下标 */
const contextMenuBlockGi = ref(null)
const hoveredNodeId = ref(null)

const nodeTypes = {
  dialogue: markRaw(DialogueNode),
  blockGroup: markRaw(BlockGroupNode)
}

const NODE_W = 240
const NODE_H = 100
const BLOCK_PAD_X = 20
const BLOCK_HEADER = 52
const BLOCK_PAD_BOTTOM = 28
const BLOCK_GAP_X = 56
const COLLAPSED_BLOCK_H = 52
const COLLAPSED_BLOCK_W = 280

const flowGroups = ref([])
/** 新建节点 / 粘贴默认落入的区块 */
const activeGroupIndex = ref(0)

/** 跨区块引用：g{组索引}:{该块内 originalId}，与同块内仅用 originalId 相对 */
const CROSS_BLOCK_REF = /^g(\d+):(.+)$/

function childRefToVueNodeId(childRef, sourceGroupIndex) {
  const s = String(childRef ?? '').trim()
  if (!s) return null
  const m = s.match(CROSS_BLOCK_REF)
  if (m) return `${m[1]}_${m[2]}`
  return `${sourceGroupIndex}_${s}`
}

function vueNodeIdToExportedRef(sourceNode, targetNode) {
  if (!sourceNode || !targetNode) return ''
  if (Number(sourceNode.groupId) === Number(targetNode.groupId)) {
    return String(targetNode.originalId)
  }
  return `g${targetNode.groupId}:${targetNode.originalId}`
}

function syncBlockTitlesToNodes() {
  nodes.value.forEach((n) => {
    if (n.type !== 'dialogue') return
    const g = flowGroups.value[n.groupId]
    n.data._blockTitle = g?.dialogue_name || `区块 ${n.groupId + 1}`
  })
}

function syncBlockGroupNodesMeta() {
  flowGroups.value.forEach((g, gi) => {
    const bn = nodes.value.find((n) => n.id === `block_${gi}`)
    if (bn && bn.data) {
      bn.data.dialogue_name = g.dialogue_name
      bn.data.collapsed = !!g.collapsed
    }
  })
}

function syncEdgesFromHiddenNodes() {
  edges.value.forEach((e) => {
    const s = nodes.value.find((n) => n.id === e.source)
    const t = nodes.value.find((n) => n.id === e.target)
    e.hidden = Boolean(s?.hidden || t?.hidden)
  })
}

function fitBlockParentBounds(gi) {
  const blockId = `block_${gi}`
  const parent = nodes.value.find((n) => n.id === blockId)
  if (!parent) return
  const g = flowGroups.value[gi]
  if (g?.collapsed) {
    parent.style = {
      ...parent.style,
      width: COLLAPSED_BLOCK_W,
      height: COLLAPSED_BLOCK_H
    }
    parent.width = COLLAPSED_BLOCK_W
    parent.height = COLLAPSED_BLOCK_H
    return
  }
  const children = nodes.value.filter(
    (n) => n.parentNode === blockId && n.type === 'dialogue' && !n.hidden
  )
  if (children.length === 0) {
    const ew = 280
    const eh = BLOCK_HEADER + 40
    parent.style = { ...parent.style, width: ew, height: eh }
    parent.width = ew
    parent.height = eh
    return
  }
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  children.forEach((n) => {
    minX = Math.min(minX, n.position.x)
    minY = Math.min(minY, n.position.y)
    maxX = Math.max(maxX, n.position.x + NODE_W)
    maxY = Math.max(maxY, n.position.y + NODE_H)
  })
  const pw = Math.max(280, maxX - minX + 2 * BLOCK_PAD_X)
  const ph = Math.max(BLOCK_HEADER + 60, maxY - minY + BLOCK_HEADER + BLOCK_PAD_BOTTOM)
  parent.style = { ...parent.style, width: pw, height: ph }
  parent.width = pw
  parent.height = ph
}

function toggleBlockCollapse(gi) {
  const g = flowGroups.value[gi]
  if (!g) return
  g.collapsed = !g.collapsed
  const blockId = `block_${gi}`
  const parent = nodes.value.find((n) => n.id === blockId)
  if (parent?.data) {
    parent.data.collapsed = g.collapsed
  }
  nodes.value.forEach((n) => {
    if (n.parentNode === blockId && n.type === 'dialogue') {
      n.hidden = g.collapsed
    }
  })
  if (g.collapsed) {
    if (parent) {
      parent.style = {
        ...parent.style,
        width: COLLAPSED_BLOCK_W,
        height: COLLAPSED_BLOCK_H
      }
      parent.width = COLLAPSED_BLOCK_W
      parent.height = COLLAPSED_BLOCK_H
    }
  } else {
    fitBlockParentBounds(gi)
  }
  syncEdgesFromHiddenNodes()
}

provide('flowToggleBlock', toggleBlockCollapse)

function openBlockContextMenuForDelete(gi, event) {
  if (gi === undefined || gi === null || !flowGroups.value[gi]) return
  if (!vueFlowRef.value) return
  const flowCoords = screenToFlowCoordinate({
    x: event.clientX,
    y: event.clientY
  })
  contextMenuPosition.value = {
    x: flowCoords.x,
    y: flowCoords.y,
    clientX: event.clientX,
    clientY: event.clientY
  }
  contextMenuBlockGi.value = gi
  showContextMenu.value = true
}

provide('flowOpenBlockContextMenu', openBlockContextMenuForDelete)

/** 收起状态删除区块后，后续组号、节点 id、连线端点整体减一 */
function compactIdsAfterRemovingGroup(removedGi) {
  const mapNodeId = (id) => {
    const dm = /^(\d+)_(.+)$/.exec(id)
    if (dm) {
      const g = Number(dm[1])
      if (g > removedGi) return `${g - 1}_${dm[2]}`
      return id
    }
    const bm = /^block_(\d+)$/.exec(id)
    if (bm) {
      const g = Number(bm[1])
      if (g > removedGi) return `block_${g - 1}`
      return id
    }
    return id
  }

  nodes.value = nodes.value.map((n) => {
    if (n.type === 'blockGroup') {
      const m = /^block_(\d+)$/.exec(n.id)
      if (!m) return n
      const k = Number(m[1])
      if (k <= removedGi) return n
      const nk = k - 1
      return {
        ...n,
        id: `block_${nk}`,
        data: { ...n.data, groupIndex: nk }
      }
    }
    if (n.type === 'dialogue' && n.groupId > removedGi) {
      const ng = n.groupId - 1
      let parentNode = n.parentNode
      if (parentNode) {
        const pm = /^block_(\d+)$/.exec(parentNode)
        if (pm) {
          const pk = Number(pm[1])
          if (pk > removedGi) parentNode = `block_${pk - 1}`
        }
      }
      return {
        ...n,
        id: `${ng}_${n.originalId}`,
        groupId: ng,
        parentNode
      }
    }
    return n
  })

  edges.value = edges.value.map((e) => {
    const s = mapNodeId(e.source)
    const t = mapNodeId(e.target)
    if (s === e.source && t === e.target) return e
    return {
      ...e,
      source: s,
      target: t,
      id: `e-${s}-${t}-${Math.random().toString(36).slice(2, 9)}`
    }
  })
}

function maxNumericOriginalInDialogueGroup(g) {
  let max = 0
  nodes.value.forEach((n) => {
    if (n.type === 'dialogue' && n.groupId === g) {
      const v = Number.parseInt(String(n.originalId).replace(/\D/g, '') || '0', 10)
      max = Math.max(max, v)
    }
  })
  return max
}

function remapEdgeEndpointsForIdChange(oldId, newId) {
  if (oldId === newId) return
  edges.value = edges.value.map((e) => {
    const s = e.source === oldId ? newId : e.source
    const t = e.target === oldId ? newId : e.target
    if (s === e.source && t === e.target) return e
    return {
      ...e,
      source: s,
      target: t,
      id: `e-${s}-${t}-${Math.random().toString(36).slice(2, 9)}`
    }
  })
}

/** 展开：只删块框，子对话节点移到画布坐标系（不再挂 parentNode）；并移除左侧对应流程区块、压缩后续组号 */
function deleteExpandedBlockGroup(gi) {
  const blockId = `block_${gi}`
  const block = nodes.value.find((n) => n.id === blockId)
  if (!block) return

  const L = flowGroups.value.length
  const mergeInto = gi > 0 ? gi - 1 : L > 1 ? gi + 1 : 0

  const bx = block.position.x
  const by = block.position.y
  const childIds = new Set(
    nodes.value
      .filter((n) => n.parentNode === blockId && n.type === 'dialogue')
      .map((n) => n.id)
  )

  const next = nodes.value
    .filter((n) => n.id !== blockId)
    .map((n) => {
      if (!childIds.has(n.id)) return n
      const detached = { ...n, hidden: false }
      delete detached.parentNode
      detached.position = {
        x: bx + n.position.x,
        y: by + n.position.y
      }
      detached.zIndex = Math.max(3, n.zIndex || 2)
      return detached
    })

  nodes.value = next

  const orphans = nodes.value.filter(
    (n) => n.type === 'dialogue' && n.groupId === gi && !n.parentNode
  )

  if (mergeInto !== gi) {
    let num = maxNumericOriginalInDialogueGroup(mergeInto)
    const gMeta = flowGroups.value[mergeInto]
    const title = gMeta?.dialogue_name || `区块 ${mergeInto + 1}`
    for (const o of orphans) {
      num += 1
      const newOid = String(num)
      const oldId = o.id
      const newId = `${mergeInto}_${newOid}`
      o.id = newId
      o.originalId = newOid
      o.groupId = mergeInto
      if (o.data) {
        o.data._blockTitle = title
      }
      remapEdgeEndpointsForIdChange(oldId, newId)
    }
  }

  flowGroups.value.splice(gi, 1)
  flowGroups.value.forEach((g, i) => {
    g.groupIndex = i
    g.id = `group_${i}`
  })

  if (flowGroups.value.length === 0) {
    flowGroups.value.push({
      id: 'group_0',
      groupIndex: 0,
      dialogue_name: '区块 1',
      site_description: '',
      collapsed: false
    })
  }

  if (L > 1) {
    compactIdsAfterRemovingGroup(gi)
  }

  let ai = activeGroupIndex.value
  if (ai === gi) {
    ai = Math.max(0, gi - 1)
  } else if (ai > gi) {
    ai -= 1
  }
  activeGroupIndex.value = Math.min(ai, Math.max(0, flowGroups.value.length - 1))

  updateNodeDataFromEdges()
  syncBlockTitlesToNodes()
  syncBlockGroupNodesMeta()
  syncEdgesFromHiddenNodes()
}

/** 收起：删除块及其中全部节点，并压缩后续区块索引 */
function deleteCollapsedBlockGroup(gi) {
  const blockId = `block_${gi}`
  const toRemove = new Set([blockId])
  nodes.value.forEach((n) => {
    if (n.type === 'dialogue' && n.groupId === gi) {
      toRemove.add(n.id)
    }
  })

  edges.value = edges.value.filter(
    (e) => !toRemove.has(e.source) && !toRemove.has(e.target)
  )
  nodes.value = nodes.value.filter((n) => !toRemove.has(n.id))

  flowGroups.value.splice(gi, 1)
  flowGroups.value.forEach((g, i) => {
    g.groupIndex = i
    g.id = `group_${i}`
  })

  compactIdsAfterRemovingGroup(gi)
  updateNodeDataFromEdges()
  syncBlockTitlesToNodes()
  syncBlockGroupNodesMeta()
  syncEdgesFromHiddenNodes()
  activeGroupIndex.value = Math.min(
    activeGroupIndex.value,
    Math.max(0, flowGroups.value.length - 1)
  )
}

function handleContextMenuDeleteBlock() {
  const gi = contextMenuBlockGi.value
  if (gi == null || !flowGroups.value[gi]) {
    closeContextMenu()
    return
  }
  const collapsed = !!flowGroups.value[gi].collapsed
  if (collapsed) {
    deleteCollapsedBlockGroup(gi)
  } else {
    deleteExpandedBlockGroup(gi)
  }
  closeContextMenu()
}

watch(
  flowGroups,
  () => {
    syncBlockTitlesToNodes()
    syncBlockGroupNodesMeta()
  },
  { deep: true }
)

const duplicateIds = computed(() => {
  const idCounts = new Map()
  nodes.value.forEach(node => {
    const count = idCounts.get(node.id) || 0
    idCounts.set(node.id, count + 1)
  })
  
  const duplicates = []
  idCounts.forEach((count, id) => {
    if (count > 1) {
      duplicates.push(id)
    }
  })
  return duplicates
})

const hasIdConflicts = computed(() => duplicateIds.value.length > 0)

const CARD_STORAGE_KEY = 'characters_data'

/** 人物卡写入或跨标签页同步后递增，用于刷新「流程 vs 人物卡」对比 */
const flowCardSyncKey = ref(0)

function bumpFlowCardSync() {
  flowCardSyncKey.value++
}

function loadCharacterCardNameSet() {
  try {
    const raw = localStorage.getItem(CARD_STORAGE_KEY)
    if (!raw) return new Set()
    const arr = JSON.parse(raw)
    if (!Array.isArray(arr)) return new Set()
    const set = new Set()
    for (const c of arr) {
      const n = (c?.data?.name ?? '').trim()
      if (n) set.add(n)
    }
    return set
  } catch {
    return new Set()
  }
}

/** 当前加载的流程里出现的说话者（去重，不含「旁白」与空名） */
const flowCharacterNames = computed(() => {
  const names = new Set()
  for (const n of nodes.value) {
    if (n.type !== 'dialogue') continue
    const raw = (n.data?.name ?? '').trim()
    if (!raw || raw === '旁白') continue
    names.add(raw)
  }
  return Array.from(names).sort((a, b) => a.localeCompare(b, 'zh-Hans-CN'))
})

/** 流程中有、本地人物卡 data.name 中未出现的名字 */
const flowNamesMissingFromCards = computed(() => {
  void flowCardSyncKey.value
  const cardNames = loadCharacterCardNameSet()
  return flowCharacterNames.value.filter((name) => !cardNames.has(name))
})

function generateNodeId() {
  nodeIdCounter.value++
  return `node_${Date.now()}_${nodeIdCounter.value}`
}

function createDefaultNodeData() {
  return {
    name: '新角色',
    content: '新对话内容',
    background: '',
    character: '',
    music: '',
    sound: '',
    transition: '',
    menu: [],
    setOrChangeFlag: '',
    checkFlag: {},
    branch_num: 0,
    parent_id: '',
    children: []
  }
}

function applyDagreLayout(flowNodes, flowEdges, offsetX = 0, offsetY = 0) {
  const g = new dagre.graphlib.Graph()

  g.setGraph({
    rankdir: 'TB',
    nodesep: 72,
    ranksep: 100,
    marginx: 40,
    marginy: 40
  })

  g.setDefaultEdgeLabel(() => ({}))

  flowNodes.forEach((node) => {
    g.setNode(node.id, { width: NODE_W, height: NODE_H })
  })

  flowEdges.forEach((edge) => {
    g.setEdge(edge.source, edge.target)
  })

  dagre.layout(g)

  const hw = NODE_W / 2
  const hh = NODE_H / 2
  flowNodes.forEach((node) => {
    const nodeWithPosition = g.node(node.id)
    if (nodeWithPosition) {
      node.position = {
        x: nodeWithPosition.x - hw + offsetX,
        y: nodeWithPosition.y - hh + offsetY
      }
    }
  })

  return flowNodes
}

/** 块内 Dagre（自上而下），并把坐标归一化到父节点内容区（含标题栏留白） */
function computeGroupLocalLayout(flowNodes, internalEdges) {
  if (!flowNodes.length) {
    return { parentW: 280, parentH: BLOCK_HEADER + 36 }
  }
  applyDagreLayout(flowNodes, internalEdges, 0, 0)
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  flowNodes.forEach((n) => {
    minX = Math.min(minX, n.position.x)
    minY = Math.min(minY, n.position.y)
    maxX = Math.max(maxX, n.position.x + NODE_W)
    maxY = Math.max(maxY, n.position.y + NODE_H)
  })
  const shiftX = BLOCK_PAD_X - minX
  const shiftY = BLOCK_HEADER - minY
  flowNodes.forEach((n) => {
    n.position.x += shiftX
    n.position.y += shiftY
  })
  maxX += shiftX
  maxY += shiftY
  const parentW = Math.max(280, maxX + BLOCK_PAD_X)
  const parentH = maxY + BLOCK_PAD_BOTTOM
  return { parentW, parentH }
}

function pickBlockGroupIndexAtFlowXY(x, y) {
  const blocks = nodes.value.filter((n) => n.type === 'blockGroup')
  for (const b of blocks) {
    const w = Number.parseFloat(b.style?.width) || COLLAPSED_BLOCK_W
    const h = Number.parseFloat(b.style?.height) || COLLAPSED_BLOCK_H
    if (
      x >= b.position.x &&
      x <= b.position.x + w &&
      y >= b.position.y &&
      y <= b.position.y + h
    ) {
      return b.data?.groupIndex ?? 0
    }
  }
  return Math.min(
    Math.max(0, activeGroupIndex.value),
    Math.max(0, flowGroups.value.length - 1)
  )
}

function convertJsonToFlow(jsonData) {
  const isArray = Array.isArray(jsonData)
  const dialogues = isArray ? jsonData : [jsonData]

  flowGroups.value = dialogues.map((dialogue, groupIndex) => ({
    id: `group_${groupIndex}`,
    groupIndex,
    dialogue_name: dialogue.dialogue_name || `区块 ${groupIndex + 1}`,
    site_description: dialogue.site_description || '',
    collapsed: false
  }))

  const dialogueNodes = []
  let currentOffsetX = 40

  dialogues.forEach((dialogue, groupIndex) => {
    const content = dialogue.dialogue_content || []
    const flowNodes = []

    let maxId = 0
    content.forEach((item) => {
      const idMatch = item.id?.toString().match(/(\d+)/)
      if (idMatch) {
        maxId = Math.max(maxId, parseInt(idMatch[1], 10))
      }
    })
    if (maxId > nodeIdCounter.value) {
      nodeIdCounter.value = maxId
    }

    const blockTitle = dialogue.dialogue_name || `区块 ${groupIndex + 1}`
    const blockId = `block_${groupIndex}`

    content.forEach((item) => {
      const cf = item.checkFlag
      const checkFlagNorm =
        cf && typeof cf === 'object' && !Array.isArray(cf) ? { ...cf } : {}

      flowNodes.push({
        id: `${groupIndex}_${item.id}`,
        originalId: item.id.toString(),
        groupId: groupIndex,
        type: 'dialogue',
        position: { x: 0, y: 0 },
        parentNode: blockId,
        zIndex: 2,
        width: NODE_W,
        height: NODE_H,
        hidden: false,
        draggable: true,
        data: {
          name: item.name,
          content: item.content,
          background: item.background || '',
          character: item.character || '',
          music: item.music || '',
          sound: item.sound || '',
          transition: item.transition || '',
          menu: item.menu || [],
          setOrChangeFlag: item.setOrChangeFlag || '',
          checkFlag: checkFlagNorm,
          branch_num: item.branch_num,
          parent_id: item.parent_id,
          children: item.children || [],
          _blockTitle: blockTitle
        }
      })
    })

    const internalEdges = []
    const prefix = `${groupIndex}_`
    content.forEach((item) => {
      const srcId = `${groupIndex}_${item.id}`
      ;(item.children || []).forEach((childRef) => {
        const tid = childRefToVueNodeId(childRef, groupIndex)
        if (tid && tid.startsWith(prefix)) {
          internalEdges.push({ source: srcId, target: tid })
        }
      })
    })

    const { parentW, parentH } = computeGroupLocalLayout(flowNodes, internalEdges)

    dialogueNodes.push({
      id: blockId,
      type: 'blockGroup',
      position: { x: currentOffsetX, y: 28 },
      draggable: true,
      selectable: true,
      dragHandle: '.block-group-header',
      zIndex: 0,
      width: parentW,
      height: parentH,
      style: {
        width: parentW,
        height: parentH
      },
      data: {
        groupIndex,
        dialogue_name: blockTitle,
        collapsed: false
      }
    })
    dialogueNodes.push(...flowNodes)
    currentOffsetX += parentW + BLOCK_GAP_X
  })

  const nodeIdSet = new Set(dialogueNodes.map((n) => n.id))
  const allEdges = []

  dialogues.forEach((dialogue, groupIndex) => {
    const content = dialogue.dialogue_content || []
    content.forEach((item) => {
      const srcId = `${groupIndex}_${item.id}`
      if (!nodeIdSet.has(srcId)) return
      ;(item.children || []).forEach((childRef) => {
        const tid = childRefToVueNodeId(childRef, groupIndex)
        if (!tid || !nodeIdSet.has(tid)) return
        allEdges.push({
          id: `e-${srcId}-${tid}-${Math.random().toString(36).slice(2, 8)}`,
          source: srcId,
          target: tid,
          type: 'smoothstep',
          animated: true,
          style: { stroke: '#00d4ff', strokeWidth: 2 },
          markerEnd: {
            type: 'arrowclosed',
            color: '#00d4ff'
          }
        })
      })
    })
  })

  nodes.value = dialogueNodes
  edges.value = allEdges
  activeGroupIndex.value = 0
  updateNodeDataFromEdges()
  syncBlockTitlesToNodes()
  syncBlockGroupNodesMeta()
  syncEdgesFromHiddenNodes()
}

function ensureFlowGroupsCoverNodes() {
  let maxG = -1
  nodes.value.forEach((n) => {
    maxG = Math.max(maxG, n.groupId)
  })
  while (maxG >= 0 && flowGroups.value.length <= maxG) {
    const k = flowGroups.value.length
    flowGroups.value.push({
      id: `group_${k}`,
      groupIndex: k,
      dialogue_name: `区块 ${k + 1}`,
      site_description: '',
      collapsed: false
    })
  }
}

function convertFlowToJson() {
  updateNodeDataFromEdges()
  ensureFlowGroupsCoverNodes()

  const dialogues = []

  for (let gi = 0; gi < flowGroups.value.length; gi++) {
    const gMeta = flowGroups.value[gi]
    let compNodes = nodes.value.filter((n) => n.groupId === gi)
    compNodes = [...compNodes].sort((a, b) => {
      const pa = parseInt(String(a.originalId).replace(/\D/g, '') || '0', 10)
      const pb = parseInt(String(b.originalId).replace(/\D/g, '') || '0', 10)
      if (pa !== pb) return pa - pb
      return String(a.originalId).localeCompare(String(b.originalId))
    })

    const content = compNodes.map((node) => {
      const rawCf = node.data.checkFlag
      const checkFlagData =
        rawCf && typeof rawCf === 'object' && !Array.isArray(rawCf) ? rawCf : {}

      const childrenVue = node.data.children || []
      const childrenExported = []
      const checkFlagOutput = {}
      const menuArr = Array.isArray(node.data.menu) ? node.data.menu : []

      childrenVue.forEach((childVueId, childIndex) => {
        const tn = nodes.value.find((n) => n.id === childVueId)
        if (!tn) return
        const expKey = vueNodeIdToExportedRef(node, tn)
        childrenExported.push(expKey)
        const co = String(tn.originalId)
        let v =
          checkFlagData[co] !== undefined
            ? checkFlagData[co]
            : checkFlagData[expKey] !== undefined
              ? checkFlagData[expKey]
              : checkFlagData[childVueId] !== undefined
                ? checkFlagData[childVueId]
                : ''
        const menuFlag = menuArr[childIndex]?.flag
        if (
          (v === undefined || v === null || String(v).trim() === '') &&
          menuFlag !== undefined &&
          menuFlag !== null &&
          String(menuFlag).trim() !== ''
        ) {
          v = String(menuFlag).trim()
        }
        checkFlagOutput[expKey] = v === undefined || v === null ? '' : String(v)
      })

      let parentExported = ''
      if (node.data.parent_id) {
        const pn = nodes.value.find((n) => n.id === node.data.parent_id)
        if (pn) parentExported = vueNodeIdToExportedRef(node, pn)
      }

      return {
        id: node.originalId || node.id,
        name: node.data.name,
        content: node.data.content,
        background: node.data.background || '',
        character: node.data.character || '',
        music: node.data.music || '',
        sound: node.data.sound || '',
        transition: node.data.transition || '',
        menu: node.data.menu || [],
        setOrChangeFlag: node.data.setOrChangeFlag || '',
        checkFlag: checkFlagOutput,
        branch_num: node.data.branch_num || 0,
        parent_id: parentExported,
        children: childrenExported
      }
    })

    dialogues.push({
      dialogue_name: gMeta.dialogue_name || `区块 ${gi + 1}`,
      site_description: gMeta.site_description || '',
      dialogue_content: content
    })
  }

  return dialogues.length === 1 ? dialogues[0] : dialogues
}

function updateNodeDataFromEdges() {
  if (isUpdating.value) return
  isUpdating.value = true
  
  nodes.value.forEach((node) => {
    if (node.type === 'blockGroup') return
    node.data.children = []
    node.data.parent_id = ''
    node.data.branch_num = 0
  })
  
  edges.value.forEach(edge => {
    const sourceNode = nodes.value.find(n => n.id === edge.source)
    const targetNode = nodes.value.find(n => n.id === edge.target)
    
    if (sourceNode) {
      if (!sourceNode.data.children) {
        sourceNode.data.children = []
      }
      if (!sourceNode.data.children.includes(edge.target)) {
        sourceNode.data.children.push(edge.target)
      }
      sourceNode.data.branch_num = sourceNode.data.children.length
    }
    
    if (targetNode) {
      targetNode.data.parent_id = edge.source
    }
  })
  
  isUpdating.value = false
}

function handleShowMenu(event) {
  const full = nodes.value.find((n) => n.id === event.nodeId)
  selectedNode.value =
    full ||
    ({
      id: event.nodeId,
      data: event.data,
      groupId: 0,
      originalId: String(event.data?.id ?? event.nodeId)
    })
  showModal.value = true
}

function handleDeleteNode(nodeId) {
  const victim = nodes.value.find((n) => n.id === nodeId)
  const gi = victim?.groupId
  edges.value = edges.value.filter((e) => e.source !== nodeId && e.target !== nodeId)
  nodes.value = nodes.value.filter((n) => n.id !== nodeId)
  updateNodeDataFromEdges()
  if (gi !== undefined) {
    fitBlockParentBounds(gi)
    syncEdgesFromHiddenNodes()
  }
}

function closeModal() {
  showModal.value = false
  selectedNode.value = null
}

function handleUpdateNode(updateData) {
  const { oldId, newId, name, content, background, character, music, sound, transition, menu, setOrChangeFlag, checkFlag } = updateData
  
  const node = nodes.value.find(n => n.id === oldId)
  if (!node) return
  
  node.data.name = name
  node.data.content = content
  node.data.background = background
  node.data.character = character
  node.data.music = music
  node.data.sound = sound
  node.data.transition = transition
  node.data.menu = menu || []
  node.data.setOrChangeFlag = setOrChangeFlag
  node.data.checkFlag = checkFlag
  
  if (oldId !== newId) {
    const existingNode = nodes.value.find(
      (n) => n.originalId === newId && n.id !== oldId && n.groupId === node.groupId
    )
    if (existingNode) {
      alert(`节点ID "${newId}" 已存在，无法修改`)
      return
    }
    
    edges.value.forEach(edge => {
      if (edge.source === oldId) {
        edge.source = newId
        edge.id = `e-${newId}-${edge.target}-${Date.now()}`
      }
      if (edge.target === oldId) {
        edge.target = newId
        edge.id = `e-${edge.source}-${newId}-${Date.now()}`
      }
    })
    
    nodes.value.forEach(n => {
      if (n.data.children) {
        const idx = n.data.children.indexOf(oldId)
        if (idx !== -1) {
          n.data.children[idx] = newId
        }
      }
      if (n.data.parent_id === oldId) {
        n.data.parent_id = newId
      }
    })
    
    node.originalId = newId
  }
  
  updateNodeDataFromEdges()
}

function addNewNodeAtPosition(x, y) {
  if (flowGroups.value.length === 0) {
    flowGroups.value.push({
      id: 'group_0',
      groupIndex: 0,
      dialogue_name: '区块 1',
      site_description: '',
      collapsed: false
    })
    activeGroupIndex.value = 0
  }
  const gi = pickBlockGroupIndexAtFlowXY(x, y)
  if (flowGroups.value[gi]?.collapsed) {
    toggleBlockCollapse(gi)
  }
  activeGroupIndex.value = gi
  const blockId = `block_${gi}`
  const orig = generateNodeId().replace(/^node_/, 'n')
  const vid = `${gi}_${orig}`
  const g = flowGroups.value[gi]
  const siblings = nodes.value.filter(
    (n) => n.parentNode === blockId && n.type === 'dialogue' && !n.hidden
  )
  let px = BLOCK_PAD_X
  let py = BLOCK_HEADER + 16
  if (siblings.length) {
    const maxY = Math.max(...siblings.map((s) => s.position.y + NODE_H))
    py = maxY + 36
  }
  const newNode = {
    id: vid,
    originalId: orig,
    groupId: gi,
    type: 'dialogue',
    position: { x: px, y: py },
    parentNode: blockId,
    zIndex: 2,
    width: NODE_W,
    height: NODE_H,
    hidden: false,
    draggable: true,
    data: {
      ...createDefaultNodeData(),
      _blockTitle: g?.dialogue_name || `区块 ${gi + 1}`
    }
  }

  nodes.value.push(newNode)
  fitBlockParentBounds(gi)
  updateNodeDataFromEdges()
  return vid
}

function handleCopyNode(event) {
  copiedNode.value = {
    id: event.nodeId,
    data: { ...event.data }
  }
}

function pasteNodeAtPosition(x, y) {
  if (!copiedNode.value) return

  if (flowGroups.value.length === 0) {
    flowGroups.value.push({
      id: 'group_0',
      groupIndex: 0,
      dialogue_name: '区块 1',
      site_description: '',
      collapsed: false
    })
    activeGroupIndex.value = 0
  }
  const gi = pickBlockGroupIndexAtFlowXY(x, y)
  if (flowGroups.value[gi]?.collapsed) {
    toggleBlockCollapse(gi)
  }
  activeGroupIndex.value = gi
  const blockId = `block_${gi}`
  const orig = generateNodeId().replace(/^node_/, 'n')
  const vid = `${gi}_${orig}`
  const g = flowGroups.value[gi]
  const src = copiedNode.value.data
  const siblings = nodes.value.filter(
    (n) => n.parentNode === blockId && n.type === 'dialogue' && !n.hidden
  )
  let px = BLOCK_PAD_X
  let py = BLOCK_HEADER + 16
  if (siblings.length) {
    const maxY = Math.max(...siblings.map((s) => s.position.y + NODE_H))
    py = maxY + 36
  }
  const newNode = {
    id: vid,
    originalId: orig,
    groupId: gi,
    type: 'dialogue',
    position: { x: px, y: py },
    parentNode: blockId,
    zIndex: 2,
    width: NODE_W,
    height: NODE_H,
    hidden: false,
    draggable: true,
    data: {
      ...createDefaultNodeData(),
      name: src.name,
      content: src.content,
      background: src.background || '',
      character: src.character || '',
      music: src.music || '',
      sound: src.sound || '',
      transition: src.transition || '',
      menu: src.menu ? src.menu.map((item) => ({ ...item })) : [],
      setOrChangeFlag: src.setOrChangeFlag || '',
      checkFlag: src.checkFlag ? { ...src.checkFlag } : {},
      _blockTitle: g?.dialogue_name || `区块 ${gi + 1}`
    }
  }

  nodes.value.push(newNode)
  fitBlockParentBounds(gi)
  updateNodeDataFromEdges()
}

function addFlowBlock() {
  const gi = flowGroups.value.length
  flowGroups.value.push({
    id: `group_${gi}`,
    groupIndex: gi,
    dialogue_name: `新区块 ${gi + 1}`,
    site_description: '',
    collapsed: false
  })
  activeGroupIndex.value = gi

  let nx = 40
  const blocks = nodes.value.filter((n) => n.type === 'blockGroup')
  if (blocks.length > 0) {
    nx =
      Math.max(
        ...blocks.map((b) => b.position.x + (Number.parseFloat(b.style?.width) || 280))
      ) + BLOCK_GAP_X
  }

  const blockId = `block_${gi}`
  const blockNode = {
    id: blockId,
    type: 'blockGroup',
    position: { x: nx, y: 28 },
    draggable: true,
    selectable: true,
    dragHandle: '.block-group-header',
    zIndex: 0,
    width: 300,
    height: 200,
    style: { width: 300, height: 200 },
    data: {
      groupIndex: gi,
      dialogue_name: flowGroups.value[gi].dialogue_name,
      collapsed: false
    }
  }
  const childNode = {
    id: `${gi}_0`,
    originalId: '0',
    groupId: gi,
    type: 'dialogue',
    position: { x: BLOCK_PAD_X, y: BLOCK_HEADER + 12 },
    parentNode: blockId,
    zIndex: 2,
    width: NODE_W,
    height: NODE_H,
    hidden: false,
    draggable: true,
    data: {
      ...createDefaultNodeData(),
      _blockTitle: flowGroups.value[gi].dialogue_name
    }
  }
  nodes.value.push(blockNode, childNode)
  fitBlockParentBounds(gi)
  updateNodeDataFromEdges()
  syncBlockTitlesToNodes()
  syncBlockGroupNodesMeta()
}

function handleContextMenu(event) {
  if (event.target.closest('.dialogue-node')) {
    return
  }
  if (event.target.closest('.vue-flow__node')) {
    return
  }

  event.preventDefault()

  const flowContainer = vueFlowRef.value
  if (!flowContainer) return

  const flowCoords = screenToFlowCoordinate({
    x: event.clientX,
    y: event.clientY
  })

  contextMenuBlockGi.value = null
  contextMenuPosition.value = {
    x: flowCoords.x,
    y: flowCoords.y,
    clientX: event.clientX,
    clientY: event.clientY
  }
  showContextMenu.value = true
}

function closeContextMenu() {
  showContextMenu.value = false
  contextMenuBlockGi.value = null
}

function handleContextMenuPaste() {
  if (copiedNode.value) {
    pasteNodeAtPosition(contextMenuPosition.value.x, contextMenuPosition.value.y)
  }
  closeContextMenu()
}

function handleContextMenuNewNode() {
  addNewNodeAtPosition(contextMenuPosition.value.x, contextMenuPosition.value.y)
  closeContextMenu()
}

function handleNodeMouseEnter(nodeId) {
  hoveredNodeId.value = nodeId
}

function handleNodeMouseLeave() {
  hoveredNodeId.value = null
}

function handleKeyDown(event) {
  if (event.key === 'Tab') {
    event.preventDefault()
    
    const flowContainer = vueFlowRef.value
    if (!flowContainer) return
    
    const rect = flowContainer.getBoundingClientRect()
    const centerX = rect.width / 2
    const centerY = rect.height / 2
    
    const flowCoords = screenToFlowCoordinate({
      x: rect.left + centerX,
      y: rect.top + centerY
    })
    
    addNewNodeAtPosition(flowCoords.x, flowCoords.y)
  }
  
  if ((event.ctrlKey || event.metaKey) && event.key === 'c') {
    if (hoveredNodeId.value) {
      const node = nodes.value.find(n => n.id === hoveredNodeId.value)
      if (node) {
        copiedNode.value = {
          id: node.id,
          data: { ...node.data }
        }
      }
    }
  }
  
  if ((event.ctrlKey || event.metaKey) && event.key === 'v') {
    if (copiedNode.value) {
      const flowContainer = vueFlowRef.value
      if (!flowContainer) return
      
      const rect = flowContainer.getBoundingClientRect()
      const flowCoords = screenToFlowCoordinate({
        x: mousePosition.value.x,
        y: mousePosition.value.y
      })
      
      pasteNodeAtPosition(flowCoords.x - 100, flowCoords.y - 50)
    }
  }
}

function handleMouseMove(event) {
  mousePosition.value = { x: event.clientX, y: event.clientY }
}

async function loadDefaultData() {
  /** 默认不加载超大 JSON，避免首屏卡顿；大文件请用导航栏「导入」 */
  const sampleData = {
    dialogue_name: '示例对话',
    site_description: '',
    dialogue_content: [
      {
        id: '0',
        name: '旁白',
        content: '开始',
        branch_num: 1,
        parent_id: '',
        children: ['1']
      },
      {
        id: '1',
        name: '旁白',
        content: '结束',
        branch_num: 0,
        parent_id: '0',
        children: []
      }
    ]
  }
  convertJsonToFlow(sampleData)
}

async function loadDemoLargeFile() {
  try {
    const response = await fetch('/sources/strctured_json/test1111.json')
    if (!response.ok) return
    const data = await response.json()
    convertJsonToFlow(data)
  } catch (e) {
    console.warn('示例大文件未找到或解析失败', e)
  }
}

/** 从 public 下的 URL 加载（如 /sources/strctured_json/renai_xxx.json） */
async function loadFromPublicUrl(url) {
  const path = String(url || '').trim()
  if (!path) {
    console.warn('loadFromPublicUrl: 空路径')
    return
  }
  try {
    const response = await fetch(path)
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }
    const data = await response.json()
    if (isDialogueScriptImportShape(data)) {
      const payload = Array.isArray(data) ? data : [data]
      const res = await fetch(`${API_BASE_URL}/api/story/import-dialogue-for-flow`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dialogue_results: payload, persist: false })
      })
      const rawText = await res.text()
      let parsed
      try {
        parsed = JSON.parse(rawText)
      } catch {
        throw new Error(rawText || `HTTP ${res.status}`)
      }
      if (!res.ok) {
        const msg =
          typeof parsed.detail === 'string'
            ? parsed.detail
            : Array.isArray(parsed.detail)
              ? parsed.detail.map((d) => d.msg || d).join('; ')
            : rawText
        throw new Error(msg || `HTTP ${res.status}`)
      }
      convertJsonToFlow(parsed.dialogues)
      return
    }
    convertJsonToFlow(data)
  } catch (e) {
    console.error('从 URL 加载流程 JSON 失败:', e)
    alert('加载流程图失败: ' + (e?.message || String(e)))
  }
}

function handleImportJson(file) {
  const reader = new FileReader()
  reader.onload = async (e) => {
    try {
      const jsonData = JSON.parse(e.target.result)
      if (isDialogueScriptImportShape(jsonData)) {
        const payload = Array.isArray(jsonData) ? jsonData : [jsonData]
        const res = await fetch(`${API_BASE_URL}/api/story/import-dialogue-for-flow`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ dialogue_results: payload, persist: false })
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
        convertJsonToFlow(data.dialogues)
        return
      }
      convertJsonToFlow(jsonData)
    } catch (error) {
      console.error('Failed to parse/import JSON:', error)
      alert(error?.message || 'JSON 文件格式错误或导入失败（对话剧本导入需后端已启动）')
    }
  }
  reader.readAsText(file)
}

function handleExportJson() {
  const jsonData = convertFlowToJson()
  const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  const fileName = Array.isArray(jsonData) ? 'dialogues.json' : `${jsonData.dialogue_name || 'flow'}.json`
  a.download = fileName
  a.click()
  URL.revokeObjectURL(url)
}

async function handleGenerateFlowBackgrounds() {
  if (flowGroups.value.length === 0) {
    alert('请先导入或编辑流程图后再生成背景。')
    return
  }
  const structuredJson = convertFlowToJson()
  isGeneratingFlowBg.value = true
  try {
    const controller = new AbortController()
    const timeoutMs = 60 * 60 * 1000
    const tid = setTimeout(() => controller.abort(), timeoutMs)
    const response = await fetch(`${API_BASE_URL}/api/runninghub/generate-flow-backgrounds`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({
        structured_json: structuredJson,
        workflow_id: DEFAULT_BG_WORKFLOW_ID
      })
    })
    clearTimeout(tid)
    const rawText = await response.text()
    let parsed
    try {
      parsed = JSON.parse(rawText)
    } catch {
      throw new Error(rawText || `HTTP ${response.status}`)
    }
    if (!response.ok) {
      const msg =
        typeof parsed.detail === 'string'
          ? parsed.detail
          : Array.isArray(parsed.detail)
            ? parsed.detail.map((d) => d.msg || d).join('; ')
            : rawText
      throw new Error(msg || `HTTP ${response.status}`)
    }
    convertJsonToFlow(parsed.dialogues)
    alert(parsed.message || '背景已生成并已写回流程节点。')
  } catch (e) {
    console.error('一键生成背景失败:', e)
    alert('生成背景失败: ' + (e?.message || String(e)))
  } finally {
    isGeneratingFlowBg.value = false
  }
}

async function handleRunFlowPygame() {
  if (flowGroups.value.length === 0) {
    alert('请先导入或编辑流程图。')
    return
  }
  const structuredJson = convertFlowToJson()
  isLaunchingPygame.value = true
  try {
    const response = await fetch(`${API_BASE_URL}/api/play/run-flow-pygame`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ structured_json: structuredJson })
    })
    const rawText = await response.text()
    let parsed
    try {
      parsed = JSON.parse(rawText)
    } catch {
      throw new Error(rawText || `HTTP ${response.status}`)
    }
    if (!response.ok) {
      const msg =
        typeof parsed.detail === 'string'
          ? parsed.detail
          : Array.isArray(parsed.detail)
            ? parsed.detail.map((d) => d.msg || d).join('; ')
            : rawText
      throw new Error(msg || `HTTP ${response.status}`)
    }
    alert(parsed.message || '已启动 Pygame 预览。')
  } catch (e) {
    console.error('启动 Pygame 失败:', e)
    alert('启动失败: ' + (e?.message || String(e)))
  } finally {
    isLaunchingPygame.value = false
  }
}

function handleEdgeDoubleClick(event) {
  const edgeId = event.edge?.id || event.id
  if (edgeId) {
    edges.value = edges.value.filter(e => e.id !== edgeId)
    updateNodeDataFromEdges()
  }
}

function relayoutNodes() {
  ensureFlowGroupsCoverNodes()
  let currentOffsetX = 40

  for (let gi = 0; gi < flowGroups.value.length; gi++) {
    const blockId = `block_${gi}`
    const parent = nodes.value.find((n) => n.id === blockId)
    if (!parent) continue

    const g = flowGroups.value[gi]
    parent.position.x = currentOffsetX
    parent.position.y = 28

    if (g?.collapsed) {
      parent.style = {
        ...parent.style,
        width: COLLAPSED_BLOCK_W,
        height: COLLAPSED_BLOCK_H
      }
      parent.width = COLLAPSED_BLOCK_W
      parent.height = COLLAPSED_BLOCK_H
      if (parent.data) parent.data.collapsed = true
      currentOffsetX += COLLAPSED_BLOCK_W + BLOCK_GAP_X
      continue
    }

    const flowNodes = nodes.value.filter(
      (n) => n.parentNode === blockId && n.type === 'dialogue' && !n.hidden
    )
    const ids = new Set(flowNodes.map((n) => n.id))
    const internalEdges = edges.value.filter(
      (e) => ids.has(e.source) && ids.has(e.target)
    )

    const { parentW, parentH } = computeGroupLocalLayout(flowNodes, internalEdges)
    parent.style = { ...parent.style, width: parentW, height: parentH }
    parent.width = parentW
    parent.height = parentH
    if (parent.data) parent.data.collapsed = false
    currentOffsetX += parentW + BLOCK_GAP_X
  }
  syncEdgesFromHiddenNodes()
}

onConnectStart((params) => {
  const { nodeId, handleType } = params
  
  connectionMade.value = false
  
  if (handleType === 'target') {
    const existingEdge = edges.value.find(e => e.target === nodeId)
    if (existingEdge) {
      dragStartInfo.value = {
        type: 'target',
        edgeId: existingEdge.id
      }
    } else {
      dragStartInfo.value = null
    }
  } else {
    dragStartInfo.value = null
  }
})

onConnectEnd(() => {
  if (dragStartInfo.value && !connectionMade.value) {
    edges.value = edges.value.filter(e => e.id !== dragStartInfo.value.edgeId)
    updateNodeDataFromEdges()
  }
  
  dragStartInfo.value = null
  connectionMade.value = false
})

onConnect((params) => {
  connectionMade.value = true
  
  const existingEdge = edges.value.find(
    e => e.source === params.source && e.target === params.target
  )
  if (existingEdge) return
  
  if (dragStartInfo.value && dragStartInfo.value.type === 'target') {
    edges.value = edges.value.filter(e => e.id !== dragStartInfo.value.edgeId)
  }
  
  const newEdge = {
    id: `e-${params.source}-${params.target}-${Date.now()}`,
    source: params.source,
    target: params.target,
    type: 'smoothstep',
    animated: true,
    style: { stroke: '#00d4ff', strokeWidth: 2 },
    markerEnd: {
      type: 'arrowclosed',
      color: '#00d4ff'
    }
  }
  
  edges.value = [...edges.value, newEdge]
  updateNodeDataFromEdges()
  
  dragStartInfo.value = null
})

function onCharacterStorageEvent() {
  bumpFlowCardSync()
}

function onWindowStorage(e) {
  if (e.key === CARD_STORAGE_KEY) bumpFlowCardSync()
}

onMounted(() => {
  loadDefaultData()
  document.addEventListener('keydown', handleKeyDown)
  document.addEventListener('mousemove', handleMouseMove)
  window.addEventListener('renai-characters-storage', onCharacterStorageEvent)
  window.addEventListener('storage', onWindowStorage)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeyDown)
  document.removeEventListener('mousemove', handleMouseMove)
  window.removeEventListener('renai-characters-storage', onCharacterStorageEvent)
  window.removeEventListener('storage', onWindowStorage)
})

defineExpose({
  handleImportJson,
  handleExportJson,
  convertJsonToFlow,
  convertFlowToJson,
  relayoutNodes,
  loadDemoLargeFile,
  loadFromPublicUrl
})
</script>

<template>
  <div class="flow-container" tabindex="0" @contextmenu="handleContextMenu">
    <aside class="blocks-panel">
      <div class="blocks-panel-head">
        <h3 class="blocks-panel-title">流程区块</h3>
        <button type="button" class="add-block-btn" @click="addFlowBlock">＋ 区块</button>
      </div>
      <p class="blocks-hint">数组每项对应一个区块；当前选中的区块用于 Tab / 粘贴新建节点。</p>
      <div v-if="flowGroups.length === 0" class="blocks-empty">
        请导入流程图 JSON、或对话剧本 JSON（chapter_name / site / dialogues，与 dialogue_*.json 相同），也可点击「区块」添加
      </div>
      <div
        v-for="(g, idx) in flowGroups"
        :key="g.id"
        class="block-card"
        :class="{ 'block-card-active': activeGroupIndex === idx }"
        @click="activeGroupIndex = idx"
      >
        <div class="block-card-head">
          <span class="block-index">#{{ idx + 1 }}</span>
          <span class="block-node-count">{{ nodes.filter((n) => n.groupId === idx).length }} 节点</span>
          <button
            type="button"
            class="block-side-toggle"
            @click.stop="toggleBlockCollapse(idx)"
          >
            {{ g.collapsed ? '展开' : '收起' }}
          </button>
        </div>
        <label class="block-label">dialogue_name</label>
        <input v-model="g.dialogue_name" class="block-input" type="text" @click.stop />
        <label class="block-label">site_description</label>
        <textarea
          v-model="g.site_description"
          class="block-textarea"
          rows="4"
          placeholder="场景与地点说明…"
          @click.stop
        />
      </div>

      <section class="flow-cast-panel" aria-label="流程中的角色">
        <h4 class="flow-cast-title">流中的角色</h4>
        <p class="flow-cast-sub">不含「旁白」；与本地人物卡「姓名」对比。</p>
        <div v-if="flowCharacterNames.length === 0" class="flow-cast-empty">
          暂无角色名（无对话节点或仅有旁白）
        </div>
        <div v-else class="flow-cast-tags">
          <span v-for="name in flowCharacterNames" :key="name" class="flow-cast-tag">{{ name }}</span>
        </div>
        <div
          v-if="flowNamesMissingFromCards.length"
          class="flow-cast-missing"
          role="status"
        >
          <span class="flow-cast-missing-label">以下在流程中有，人物卡中没有：</span>
          <span class="flow-cast-missing-value">{{ flowNamesMissingFromCards.join('、') }}</span>
        </div>
        <p v-else-if="flowCharacterNames.length" class="flow-cast-ok">
          人物卡已覆盖当前流中的全部角色名。
        </p>
      </section>
    </aside>

    <div class="flow-main">
    <div v-if="hasIdConflicts" class="id-conflict-warning">
      <span class="warning-icon">⚠️</span>
      <span class="warning-text">节点ID冲突: {{ duplicateIds.join(', ') }}</span>
    </div>
    
    <div v-if="flowGroups.length > 1" class="flow-groups-info">
      <span class="groups-label">{{ flowGroups.length }} 个区块 · 跨块连线导出为 g组号:节点ID</span>
    </div>
    
    <VueFlow
      v-model:nodes="nodes"
      v-model:edges="edges"
      :node-types="nodeTypes"
      :default-viewport="{ zoom: 1, x: 0, y: 0 }"
      :min-zoom="0.2"
      :max-zoom="4"
      :snap-to-grid="false"
      :connection-line-style="{ stroke: '#00d4ff', strokeWidth: 2 }"
      :connection-line-type="'smoothstep'"
      :connection-mode="ConnectionMode.Loose"
      :nodes-connectable="true"
      :nodes-draggable="true"
      :edges-updatable="true"
      fit-view-on-init
      class="vue-flow-wrapper"
      @edge-double-click="handleEdgeDoubleClick"
    >
      <Background pattern-color="rgba(255, 255, 255, 0.1)" :gap="20" />
      <Controls />
      <MiniMap />
      
      <template #node-dialogue="nodeProps">
        <DialogueNode
          v-bind="nodeProps"
          @show-menu="handleShowMenu"
          @delete="handleDeleteNode"
          @copy="handleCopyNode"
          @mouseenter="handleNodeMouseEnter(nodeProps.id)"
          @mouseleave="handleNodeMouseLeave"
        />
      </template>
    </VueFlow>
    
    <div class="help-tip">
      <span>区块内自上而下排版，多块横排；拖区块顶部标题栏整体移动；标题栏右键可删除区块（展开删框并移除侧栏项、节点并入邻块；收起删块内全部）；TAB/粘贴落在指针下或当前选中块</span>
      <button class="relayout-btn" type="button" @click="relayoutNodes" title="按区块重新整理布局">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 12a9 9 0 11-9-9c2.52 0 4.93 1 6.74 2.74L21 8"/>
          <path d="M21 3v5h-5"/>
        </svg>
        按区块整理
      </button>
      <button class="relayout-btn demo-heavy-btn" type="button" @click="loadDemoLargeFile" title="可能明显卡顿，仅作联调">
        加载大示例 JSON
      </button>
      <button
        class="relayout-btn generate-bg-btn"
        type="button"
        :disabled="isGeneratingFlowBg || flowGroups.length === 0"
        title="按各区块 site_description 调用 RunningHub 出图，保存到 public/pic_bg/时间戳/，并写回节点背景路径"
        @click="handleGenerateFlowBackgrounds"
      >
        {{ isGeneratingFlowBg ? '生成背景中…' : '一键生成背景' }}
      </button>
      <button
        class="relayout-btn run-pygame-btn"
        type="button"
        :disabled="isLaunchingPygame || isGeneratingFlowBg || flowGroups.length === 0"
        title="将当前画布导出为结构化 JSON，由本机后端启动 pygame_play 预览（需已 pip install pygame）"
        @click="handleRunFlowPygame"
      >
        {{ isLaunchingPygame ? '启动中…' : '运行剧本(Pygame)' }}
      </button>
    </div>
    </div>
    
    <Teleport to="body">
      <NodeDetailModal
        v-if="showModal && selectedNode"
        :node="selectedNode"
        :flow-nodes="nodes"
        @close="closeModal"
        @update="handleUpdateNode"
      />
      
      <div
        v-if="showContextMenu"
        class="context-menu-overlay"
        @click="closeContextMenu"
      >
        <div
          class="context-menu"
          :style="{
            left: contextMenuPosition.clientX + 'px',
            top: contextMenuPosition.clientY + 'px'
          }"
        >
          <button class="context-menu-item" @click="handleContextMenuPaste" :disabled="!copiedNode">
            <span class="menu-icon">📌</span>
            粘贴节点
          </button>
          <div class="context-menu-divider"></div>
          <button class="context-menu-item" @click="handleContextMenuNewNode">
            <span class="menu-icon">➕</span>
            创建新节点
          </button>
          <template v-if="contextMenuBlockGi !== null">
            <div class="context-menu-divider" />
            <p class="context-menu-block-hint">
              {{
                flowGroups[contextMenuBlockGi]?.collapsed
                  ? '收起中：将删除本块及其中全部节点。'
                  : '展开中：删除块框与左侧对应区块；对话节点保留在画布上，并并入相邻区块（用于导出与侧栏）。'
              }}
            </p>
            <button type="button" class="context-menu-item context-menu-item-danger" @click="handleContextMenuDeleteBlock">
              <span class="menu-icon">🗑</span>
              删除区块
            </button>
          </template>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
.flow-container {
  display: flex;
  width: 100%;
  height: calc(100vh - 60px);
  background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
  position: relative;
  overflow: hidden;
}

.blocks-panel {
  width: 288px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 12px;
  border-right: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(15, 15, 26, 0.95);
  overflow-y: auto;
  z-index: 5;
}

.blocks-panel-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.blocks-panel-title {
  margin: 0;
  font-size: 0.95rem;
  color: rgba(255, 255, 255, 0.9);
}

.add-block-btn {
  padding: 6px 10px;
  font-size: 0.75rem;
  border-radius: 8px;
  border: 1px solid rgba(0, 212, 255, 0.4);
  background: rgba(0, 212, 255, 0.15);
  color: #00d4ff;
  cursor: pointer;
  white-space: nowrap;
}

.add-block-btn:hover {
  background: rgba(0, 212, 255, 0.28);
}

.blocks-hint {
  margin: 0;
  font-size: 0.7rem;
  line-height: 1.4;
  color: rgba(255, 255, 255, 0.35);
}

.blocks-empty {
  font-size: 0.8rem;
  color: rgba(255, 255, 255, 0.4);
  padding: 12px 4px;
  text-align: center;
}

.flow-cast-panel {
  margin-top: auto;
  padding-top: 14px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  flex-shrink: 0;
}

.flow-cast-title {
  margin: 0 0 4px 0;
  font-size: 0.82rem;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.88);
}

.flow-cast-sub {
  margin: 0 0 10px 0;
  font-size: 0.65rem;
  line-height: 1.35;
  color: rgba(255, 255, 255, 0.38);
}

.flow-cast-empty {
  font-size: 0.72rem;
  color: rgba(255, 255, 255, 0.35);
  padding: 6px 0;
}

.flow-cast-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 8px;
  margin-bottom: 10px;
}

.flow-cast-tag {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 500;
  color: #e0f7ff;
  background: rgba(0, 212, 255, 0.14);
  border: 1px solid rgba(0, 212, 255, 0.35);
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.flow-cast-missing {
  padding: 8px 10px;
  border-radius: 8px;
  background: rgba(255, 193, 7, 0.08);
  border: 1px solid rgba(255, 193, 7, 0.35);
  font-size: 0.68rem;
  line-height: 1.45;
  color: rgba(255, 255, 255, 0.82);
}

.flow-cast-missing-label {
  display: block;
  margin-bottom: 4px;
  color: #ffc107;
  font-weight: 600;
}

.flow-cast-missing-value {
  color: rgba(255, 224, 130, 0.95);
  word-break: break-all;
}

.flow-cast-ok {
  margin: 0;
  font-size: 0.68rem;
  color: rgba(46, 204, 113, 0.85);
  line-height: 1.4;
}

.block-card {
  padding: 10px;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(26, 26, 46, 0.6);
  cursor: pointer;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.block-card:hover {
  border-color: rgba(0, 212, 255, 0.25);
}

.block-card-active {
  border-color: rgba(0, 212, 255, 0.55);
  box-shadow: 0 0 0 1px rgba(0, 212, 255, 0.2);
}

.block-card-head {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 6px;
  margin-bottom: 8px;
}

.block-side-toggle {
  font-size: 0.65rem;
  padding: 3px 8px;
  border-radius: 6px;
  border: 1px solid rgba(0, 212, 255, 0.35);
  background: rgba(0, 212, 255, 0.12);
  color: #00d4ff;
  cursor: pointer;
}

.block-side-toggle:hover {
  background: rgba(0, 212, 255, 0.22);
}

.demo-heavy-btn {
  border-color: rgba(255, 193, 7, 0.45) !important;
  color: #ffc107 !important;
  background: rgba(255, 193, 7, 0.08) !important;
}

.block-index {
  font-size: 0.75rem;
  font-weight: 600;
  color: #7b2cbf;
}

.block-node-count {
  font-size: 0.7rem;
  color: rgba(255, 255, 255, 0.4);
}

.block-label {
  display: block;
  font-size: 0.65rem;
  color: rgba(255, 255, 255, 0.45);
  margin-bottom: 4px;
}

.block-input,
.block-textarea {
  width: 100%;
  margin-bottom: 8px;
  padding: 6px 8px;
  border-radius: 6px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(0, 0, 0, 0.25);
  color: rgba(255, 255, 255, 0.9);
  font-size: 0.8rem;
  font-family: inherit;
}

.block-textarea {
  resize: vertical;
  min-height: 64px;
  line-height: 1.4;
}

.flow-main {
  flex: 1;
  position: relative;
  min-width: 0;
  min-height: 0;
}

.id-conflict-warning {
  position: absolute;
  top: 10px;
  right: 10px;
  padding: 10px 16px;
  background: rgba(255, 71, 87, 0.9);
  border: 2px solid #ff4757;
  border-radius: 8px;
  color: #fff;
  font-size: 0.85rem;
  z-index: 100;
  display: flex;
  align-items: center;
  gap: 8px;
  box-shadow: 0 4px 15px rgba(255, 71, 87, 0.4);
}

.warning-icon {
  font-size: 1rem;
}

.warning-text {
  font-weight: 500;
}

.flow-groups-info {
  position: absolute;
  top: 10px;
  left: 10px;
  padding: 8px 16px;
  background: rgba(0, 212, 255, 0.2);
  border: 1px solid rgba(0, 212, 255, 0.4);
  border-radius: 8px;
  color: #00d4ff;
  font-size: 0.85rem;
  z-index: 100;
}

.groups-label {
  font-weight: 500;
}

.vue-flow-wrapper {
  width: 100%;
  height: 100%;
}

.help-tip {
  position: absolute;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  padding: 8px 16px;
  background: rgba(26, 26, 46, 0.9);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 20px;
  font-size: 0.8rem;
  color: rgba(255, 255, 255, 0.6);
  z-index: 10;
  display: flex;
  align-items: center;
  gap: 16px;
}

.relayout-btn {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  background: rgba(0, 212, 255, 0.2);
  border: 1px solid rgba(0, 212, 255, 0.4);
  border-radius: 12px;
  color: #00d4ff;
  font-size: 0.75rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.relayout-btn:hover {
  background: rgba(0, 212, 255, 0.3);
  border-color: rgba(0, 212, 255, 0.6);
  transform: scale(1.05);
}

.relayout-btn svg {
  flex-shrink: 0;
}

.generate-bg-btn {
  background: rgba(123, 44, 191, 0.25);
  border-color: rgba(179, 102, 233, 0.45);
  color: #d4a5ff;
}

.generate-bg-btn:hover:not(:disabled) {
  background: rgba(123, 44, 191, 0.38);
  border-color: rgba(179, 102, 233, 0.65);
}

.generate-bg-btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
  transform: none;
}

.run-pygame-btn {
  background: rgba(46, 213, 115, 0.2);
  border-color: rgba(46, 213, 115, 0.45);
  color: #7bed9f;
}

.run-pygame-btn:hover:not(:disabled) {
  background: rgba(46, 213, 115, 0.32);
  border-color: rgba(46, 213, 115, 0.65);
}

.run-pygame-btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
  transform: none;
}

.context-menu-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 9999;
}

.context-menu {
  position: fixed;
  background: rgba(26, 26, 46, 0.98);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  padding: 8px 0;
  min-width: 160px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
  z-index: 10000;
}

.context-menu-item {
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

.context-menu-item:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
}

.context-menu-item:disabled {
  color: rgba(255, 255, 255, 0.3);
  cursor: not-allowed;
}

.context-menu-divider {
  height: 1px;
  background: rgba(255, 255, 255, 0.1);
  margin: 4px 0;
}

.context-menu-block-hint {
  margin: 0;
  padding: 6px 16px 4px;
  font-size: 0.72rem;
  line-height: 1.35;
  color: rgba(255, 255, 255, 0.45);
}

.context-menu-item-danger {
  color: #ff6b6b;
}

.context-menu-item-danger:hover:not(:disabled) {
  background: rgba(255, 59, 48, 0.15);
  color: #ff8787;
}

:deep(.vue-flow__edge-path) {
  stroke: #00d4ff;
  stroke-width: 2;
}

:deep(.vue-flow__edge.animated path) {
  stroke-dasharray: 5;
  animation: flowDash 0.5s linear infinite;
}

:deep(.vue-flow__edge:hover .vue-flow__edge-path) {
  stroke: #ff4757;
  stroke-width: 3;
}

:deep(.vue-flow__edge.selected .vue-flow__edge-path) {
  stroke: #7b2cbf;
  stroke-width: 3;
}

@keyframes flowDash {
  to {
    stroke-dashoffset: -10;
  }
}

:deep(.vue-flow__controls) {
  background: rgba(26, 26, 46, 0.9);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

:deep(.vue-flow__controls-button) {
  background: transparent;
  border: none;
  color: rgba(255, 255, 255, 0.7);
  fill: currentColor;
}

:deep(.vue-flow__controls-button:hover) {
  background: rgba(255, 255, 255, 0.1);
  color: #00d4ff;
}

:deep(.vue-flow__minimap) {
  background: rgba(26, 26, 46, 0.9);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
}

:deep(.vue-flow__background) {
  background: transparent;
}

:deep(.vue-flow__handle) {
  width: 16px;
  height: 16px;
  background: #00d4ff;
  border: 3px solid #1a1a2e;
  border-radius: 50%;
  transition: all 0.2s ease;
  cursor: crosshair;
}

:deep(.vue-flow__handle:hover) {
  transform: scale(1.4);
  background: #7b2cbf;
  box-shadow: 0 0 15px rgba(0, 212, 255, 0.8);
}

:deep(.vue-flow__handle.source) {
  background: #32cd32;
}

:deep(.vue-flow__handle.source:hover) {
  background: #7b2cbf;
}

:deep(.vue-flow__handle.target) {
  background: #ff69b4;
}

:deep(.vue-flow__handle.target:hover) {
  background: #7b2cbf;
}

:deep(.vue-flow__handle.connecting) {
  background: #7b2cbf;
  box-shadow: 0 0 15px rgba(123, 44, 191, 0.8);
}

:deep(.vue-flow__handle.valid) {
  background: #32cd32;
  box-shadow: 0 0 15px rgba(50, 205, 50, 0.8);
}
</style>
