# 部署说明（可放在 GitHub 供协作者/使用者参考）

## `.env` 文件怎么办？

| 场景 | 做法 |
|------|------|
| **本机 / 自有服务器** | **不要**把真实 `.env` 推送到 GitHub。仓库里只保留 **`server/.env.example`**（无密钥模板）。在服务器上：`cd server` 后执行 `copy .env.example .env`（Windows）或 `cp .env.example .env`（Linux/macOS），再编辑 `.env` 填入密钥。 |
| **GitHub Actions / CI** | 在仓库 **Settings → Secrets and variables → Actions** 里添加与 `.env` 同名的 **Secrets**，在 workflow 里用 `env:` 注入，或生成临时 `.env` 文件（步骤里 `echo`/`run` 写入，且勿 `cat` 到日志）。 |
| **云平台（Railway、Render、Fly.io、自有 K8s 等）** | 使用平台提供的 **Environment Variables / Secrets** 面板逐项配置，**不要**把 `.env` 打进镜像或提交仓库。若运行时仍依赖文件，可在启动命令前由脚本从环境变量生成 `server/.env`（仅部署环境使用）。 |

本项目后端通过 `python-dotenv` 读取 **`server/.env`**（见 `server/app/services/const.py`）。  
根目录与 `server/` 下的 `.env` 已在 `.gitignore` 中忽略，避免误提交。

---

## 架构说明（部署时要心里有数）

- **前端**：Vue 3 + Vite，构建产物在 **`dist/`**。
- **后端**：FastAPI，在 **`server/`** 目录下用 Uvicorn 启动。
- 当前前端部分请求里 **API 基地址写为 `http://localhost:8000`**。若你部署到公网且前后端不同域，需要自行改为你的 API 地址，或通过 **同源反向代理**（见下）让浏览器只访问一个域名。

---

## 方式一：自有服务器（常见：Nginx + 静态 + 反代 API）

适合：一台 Linux/Windows 服务器，有域名或 IP。

1. **安装依赖并构建前端**（在仓库根目录）  
   `npm ci` → `npm run build`  
   得到 `dist/`。

2. **准备 Python 环境**（在 `server/`）  
   ```bash
   python -m venv venv
   # Windows: venv\Scripts\activate
   # Linux/macOS: source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **配置密钥**  
   `server/.env`：从 `.env.example` 复制并填写（见上文「`.env` 文件怎么办？」）。

4. **启动后端**（工作目录为 `server/`）  
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

5. **Nginx（示例思路）**  
   - 根路径 `location /` 指向构建好的 `dist/`（`index.html` + 静态资源）。  
   - `location /api/`（以及若前端使用 `/sources/`、`/pic_bg/` 等静态资源与后端写入路径一致时）**反代**到 `http://127.0.0.1:8000`。  
   具体 `proxy_pass`、WebSocket/SSE（故事生成流式接口）需按需开启缓冲与超时。

6. **前端 API 地址**  
   若浏览器只访问你的域名（同源），应把前端的 API 基地址改为 **空字符串（走相对路径 `/api/...`）** 或 **你的 HTTPS 域名**，否则仍会请求 `localhost:8000` 导致用户浏览器失败。

---

## 方式二：仅内网/本机演示（双进程）

1. 终端 A：`cd server` → 激活 venv → `uvicorn app.main:app --reload --port 8000`  
2. 终端 B：仓库根目录 `npm run dev`（Vite 开发服务器，已代理 `/api` 到 8000）。

同样需要配置好 `server/.env`。

---

## 方式三：Docker（可选，需自行维护 Dockerfile）

仓库若未附带官方镜像，可自行编写多阶段构建：Node 构建 `dist` → Python 镜像安装 `server/requirements.txt` → 用进程管理器同时起静态服务与 Uvicorn，或仅起 Uvicorn 并由外层 Nginx 提供静态文件。**密钥一律用构建/运行时的环境变量或挂载只读 secret 文件，不要写进镜像层。**

---

## 检查清单（发布到 GitHub 前）

- [ ] `server/.env` **未**被提交（仅保留 `server/.env.example`）。  
- [ ] README 或本文已说明：**复制 `.env.example` → `.env` 再填写**。  
- [ ] 协作者知悉：密钥只在各环境本地或平台 Secrets 中配置，Issue/PR 中勿粘贴真实 Key。

---

## 与 Pygame 预览相关

`/api/play/run-flow-pygame` 会在**运行后端的机器**上拉起本机 Pygame，需要该环境已安装 `pygame`（已在 `requirements.txt` 中），且通常**不适合**无桌面/无显示的服务器；若部署在纯 Linux 无头服务器，该功能可能不可用或需额外配置显示/虚拟帧缓冲。
