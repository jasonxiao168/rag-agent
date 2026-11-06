.PHONY: help install ingest dev web test eval docker-build docker-up docker-down clean

help:
	@echo "make install   安装后端与前端依赖"
	@echo "make ingest    构建/重建知识库索引"
	@echo "make dev       启动 FastAPI（:8000）"
	@echo "make web       启动前端开发服务器（:5173）"
	@echo "make test      运行单元测试"
	@echo "make eval      运行评测并生成报告"
	@echo "make docker-up 一键启动后端 + 前端"

install:
	python3 -m pip install -r backend/requirements.txt
	cd frontend && npm install

ingest:
	cd backend && python3 -m scripts.gen_corpus && python3 -m scripts.ingest

dev:
	cd backend && uvicorn app.main:app --reload --port 8000

web:
	cd frontend && npm run dev

test:
	cd backend && MOCK_MODE=1 python3 -m unittest discover -s tests -v

eval:
	cd backend && python3 -m scripts.evaluate --compare
	cd backend && python3 -m scripts.evaluate_multiturn

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-verify:
	bash scripts/docker_verify.sh

docker-down:
	docker compose down

clean:
	rm -rf data/index data/sessions data/tickets.json
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -rf frontend/dist
