# 启动后端服务器
Write-Host "启动后端服务器..."
cd server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
