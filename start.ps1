# ============================================
# 一键启动前后端；先按 backend/README.md 和 frontend/README.md 准备依赖。
# 用法：在项目根目录打开 PowerShell，运行：
#   .\start.ps1
# 脚本会打开浏览器；两个 PowerShell 窗口分别显示后端和前端日志。
# 关闭：直接关掉弹出的两个 PowerShell 窗口。
# ============================================
$backendDir = Join-Path $PSScriptRoot 'backend'
$frontendDir = Join-Path $PSScriptRoot 'frontend'
$python = Join-Path $backendDir '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    throw '未找到 backend/.venv；请先按 backend/README.md 创建虚拟环境并安装依赖。'
}
if (-not (Test-Path -LiteralPath (Join-Path $frontendDir 'node_modules'))) {
    throw '未找到 frontend/node_modules；请先在 frontend/ 执行 npm ci。'
}

# 工作目录决定相对模型、索引与配置路径；保留可见终端便于查看进度和错误。
Start-Process powershell -WorkingDirectory $backendDir -WindowStyle Normal -ArgumentList '-NoExit', '-NoProfile', '-Command', '.\.venv\Scripts\python.exe scripts\serve_api.py'
Start-Process powershell -WorkingDirectory $frontendDir -WindowStyle Normal -ArgumentList '-NoExit', '-NoProfile', '-Command', 'npm run dev'
Write-Host "后端: http://127.0.0.1:8000"
Write-Host "前端: http://localhost:5173"
Start-Sleep -Seconds 1
Start-Process "http://localhost:5173"
