# Nightingale AI - 集成指南

## 项目概述
这是一个完整的医疗AI系统，包含Next.js前端和Python FastAPI后端，支持患者和医生的不同界面。

## 环境配置

### 1. 环境变量设置
在Vercel项目设置中添加以下环境变量：

\`\`\`
NEXT_PUBLIC_API_URL=http://localhost:8000
\`\`\`

对于生产环境，将URL更改为您的实际后端服务地址。

### 2. 后端启动
在项目根目录运行Python后端：

\`\`\`bash
# 安装依赖
pip install -r backend/requirements.txt

# 启动FastAPI服务器
cd backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
\`\`\`

### 3. 前端启动
Next.js前端会自动运行在v0环境中。

## 功能特性

### 认证系统
- **注册**: `/auth/signup` - 支持患者和医生角色
- **登录**: `/auth/login` - 基于角色的重定向
- **角色区分**: 患者和医生有不同的仪表板

### 患者功能
- **仪表板**: `/patient/dashboard` - 查看个人健康记录
- **搜索功能**: 基于历史记录的智能问答
- **隐私保护**: 所有数据自动脱敏处理
- **记录查看**: 查看已审核的就诊总结

### 医生功能
- **总览界面**: `/doctor/dashboard` - 管理所有患者
- **审核系统**: 审核和批准患者记录
- **会话管理**: 创建和管理咨询会话

### API端点
所有API调用通过Next.js代理路由到FastAPI后端：

- `POST /api/auth/login` - 用户登录
- `POST /api/auth/signup` - 用户注册
- `POST /api/consent/record` - 记录患者同意
- `POST /api/session/start` - 开始新会话
- `POST /api/asr/ingest` - 语音转录数据摄取
- `POST /api/summarize/{sessionId}` - 生成会话总结
- `GET /api/patient/{patientId}/qa` - 患者问答搜索

## 数据库
使用SQLite数据库 (`nightingale.db`)，包含以下表：
- `users` - 用户账户
- `patients` - 患者信息
- `sessions` - 咨询会话
- `segments` - 语音片段
- `summaries` - 会话总结

## 安全特性
- **数据脱敏**: 自动识别和脱敏个人健康信息
- **访问控制**: 基于角色的权限管理
- **隐私保护**: 符合医疗隐私标准
- **审核流程**: 医生审核后患者才能查看记录

## 测试账户
创建测试账户时：
- 用户名包含 "doctor" 或 "dr" 的会被识别为医生
- 其他用户默认为患者角色

## 故障排除
1. **网络错误**: 确保FastAPI后端在 `http://localhost:8000` 运行
2. **CORS问题**: 后端需要配置允许前端域名的跨域请求
3. **认证问题**: 检查localStorage中的token是否正确存储
