#!/usr/bin/env bash
# 一键跑通：构建 → 语料 → 模型 → 索引 → 启动 → 健康检查 → 冒烟
#
# 用法（仓库根目录）：
#   bash scripts/docker_verify.sh           # 全流程
#   bash scripts/docker_verify.sh --down    # 跑完停掉容器

set -euo pipefail
cd "$(dirname "$0")/.."

DOWN=0
[[ "${1:-}" == "--down" ]] && DOWN=1

log()  { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
info() { printf '       %s\n' "$*"; }
die()  { printf '\n[FAIL] %s\n' "$*" >&2; exit 1; }

if [[ ! -f .env ]]; then
  cp .env.example .env
  info ".env 不存在，已从 .env.example 复制（未填 Key 时以 MOCK 模式运行）"
fi

log "1/8 构建镜像"
docker compose build

log "2/8 生成演示语料"
docker compose run --rm backend python -m scripts.gen_corpus --force

log "3/8 准备向量模型"
docker compose run --rm backend python -m scripts.download_models --embedding-only

log "4/8 构建索引"
docker compose run --rm backend python -m scripts.ingest --full

log "5/8 启动服务"
docker compose up -d

log "6/8 等待后端健康检查"
ready=0
for i in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    ready=1
    info "backend ready (${i} 次探测)"
    break
  fi
  sleep 2
done
[[ $ready -eq 1 ]] || die "backend 120 秒内未就绪，查看 docker compose logs backend"

log "7/8 容器网络：frontend → backend"
docker compose exec -T frontend curl -fsS http://backend:8000/api/health >/dev/null \
  || die "frontend 无法访问 backend:8000"
info "OK"

log "8/8 端到端冒烟（nginx :8080）"

HEALTH=$(curl -fsS http://127.0.0.1:8080/api/health)
echo "$HEALTH" | grep -Eq '"status": *"ok"' || die "health 状态异常"
echo "$HEALTH" | grep -Eq '"vector_backend": *"qdrant"' || die "向量后端不是 qdrant"
info "health ok / qdrant / langgraph=$(echo "$HEALTH" | grep -Eo '"langgraph": *[a-z]+' | awk -F: '{print $2}' | tr -d ' ')"

curl -fsS http://127.0.0.1:8080/api/users | grep -q '"erin"' || die "/api/users 异常"
info "users ok"

ANSWER=$(curl -fsS -X POST http://127.0.0.1:8080/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"一线城市出差住宿标准是多少？","user_id":"carol","stream":false}')
echo "$ANSWER" | grep -Eq '"route": *"kb"' || die "知识问答路由异常"
echo "$ANSWER" | grep -q '600' || die "回答里没有 600"
info "kb: route=kb, answer contains 600"

DENIED=$(curl -fsS -X POST http://127.0.0.1:8080/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"经销商首批进货金额门槛是多少？","user_id":"dave","stream":false}')
echo "$DENIED" | grep -q '抱歉' || die "越权问题没有拒答"
info "acl: 市场部被拒答"

TOOL=$(curl -fsS -X POST http://127.0.0.1:8080/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"订单 ORD-2026-1001 到哪了？","user_id":"carol","stream":false}')
echo "$TOOL" | grep -Eq '"route": *"tool"' || die "工具路由异常"
echo "$TOOL" | grep -q 'SF1234567890' || die "工具结果里没有运单号"
info "tool: route=tool, 返回运单号"

SSE_FILE=$(mktemp)
trap 'rm -f "$SSE_FILE"' EXIT
curl -sN --max-time 90 -X POST http://127.0.0.1:8080/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"售后工单多久闭环？","user_id":"carol","stream":true}' \
  -o "$SSE_FILE" || true

TOKENS=$(grep -c '^event: token' "$SSE_FILE" || true)
[[ "$TOKENS" -gt 0 ]] || die "SSE 没有 token 事件（收到 $(wc -c < "$SSE_FILE") 字节）"
grep -q '^event: done' "$SSE_FILE" || die "SSE 没有 done 事件"
grep -q '^event: trace' "$SSE_FILE" || die "SSE 没有 trace 事件"
info "sse: ${TOKENS} 个 token 事件，trace / done 正常"

curl -fsS http://127.0.0.1:8080/ | grep -q '<div id="root">' || die "nginx 未托管前端页面"
info "web: dist 正常"

log "完成"
echo "       前端  http://localhost:8080"
echo "       API   http://localhost:8000/docs"

if [[ $DOWN -eq 1 ]]; then
  log "停止容器"
  docker compose down
fi
