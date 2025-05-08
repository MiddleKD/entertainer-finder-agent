run-up:
	docker compose up

run-down:
	docker compose down

dev-server:
	uv run uvicorn main:app --host 127.0.0.1 --port 8000 --reload --app-dir src

dev-mcp:
	uv run src/mcp_server.py --port 8001

dev-preprocess:
	uv run src/preprocess.py

dev-qdrant:
	docker run -p 6333:6333 -p 6334:6334 --rm \
		-v $(PWD)/datas/qdrant:/qdrant/storage \
		--name qdrant_container \
		qdrant/qdrant 

dev-n8n:
	docker run -it --rm --name n8n -p 5678:5678 --network host \
		-v $(PWD)/datas/n8n_data:/home/node/.n8n \
		--name n8n_container \
		docker.n8n.io/n8nio/n8n

dev-benchmark:
	PYTHONPATH=. uv run benchmark/benchmark.py

test:
	PYTHONPATH=$(PWD)/src pytest tests/ -v

format:
	@echo "Running isort..."
	isort .
	@echo "Running black..."
	black .

.PHONY: run-up run-down dev-server dev-mcp dev-preprocess dev-qdrant dev-n8n dev-benchmark test format
