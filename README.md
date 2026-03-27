# Vue 3

Vue 3 是一个用于构建用户界面的渐进式 JavaScript 框架。它具备以下几个特点：

- **高性能**：Vue 3 使用 Proxy 实现了更快的数据响应。
- **Composition API**：提供了新的组织和组合逻辑的方式。
- **更小的体积**：Vue 3 的核心库比 Vue 2 更小，加载更快。

## 安装
你可以通过 npm 安装 Vue 3：

```bash
npm install vue@next
```

---

# Vite

Vite 是一个新一代的前端构建工具，具有极速的冷启动和热更新功能。主要优势包括：

- **即时启动**：基于原生 ES 模块，Vite 可以在几乎没有打包的情况下启动应用。
- **快速热更新**：文件更新后，Vite 使用热模块替换进行快速更新。

## 安装
通过 npm 安装 Vite：

```bash
npm install vite
```

---

# Vue Flow

Vue Flow 是一个用于创建流程图或图表的库。它提供了简单的 API，易于使用。

## 安装
通过 npm 安装 Vue Flow：

```bash
npm install @vue-flow/vue-flow
```

## 使用示例

```javascript
import { createApp } from 'vue';
import VueFlow from '@vue-flow/vue-flow';

const app = createApp({});
app.use(VueFlow);
app.mount('#app');
```

---

可以访问 [Vue 3 官方文档](https://v3.vuejs.org/) 以及 [Vite 官方文档](https://vitejs.dev/) 了解更多信息。