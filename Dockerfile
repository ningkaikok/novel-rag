# 多阶段构建（P3 工程化 / M5 第一项）：
#   stage 1  Node 构建前端产物
#   stage 2  Python 运行时（uv 装锁定依赖）+ 前端 dist，FastAPI 单端口托管
#
# 构建：docker build -t novel-rag .
# 运行：推荐用 docker compose up（带 pgvector、模型缓存卷和健康检查）

# ---------------------------------------------------------------- 前端构建
FROM node:22-alpine AS frontend-build
WORKDIR /app/frontend
# 先拷 package*.json 再 npm ci：依赖没变时这一层直接命中缓存
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------- 运行时
FROM python:3.12-slim

# uv 二进制从官方镜像拷贝，不引入额外构建层
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # 不在容器里跑开发工具链
    UV_NO_DEV=1 \
    # HuggingFace 模型缓存位置（compose 挂卷持久化，避免每次重建都下 1.1GB 重排器）
    HF_HOME=/cache/hf \
    PYTHONUNBUFFERED=1

# 先装依赖再拷代码：代码变更不触发昂贵的依赖层重建
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ src/
COPY backend/ backend/
COPY --from=frontend-build /app/frontend/dist frontend/dist
# ingest 等入口需要 data/novels 目录存在（compose 里挂载真实数据卷）
RUN mkdir -p data/novels

EXPOSE 8000
# lifespan 里会连库/建表；数据库由 compose 的 depends_on + healthcheck 保证先就绪
CMD ["uv", "run", "--no-dev", "python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
