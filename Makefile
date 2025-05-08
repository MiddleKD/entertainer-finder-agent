run_server:
	uv run uvicorn main:app --host 127.0.0.1 --port 8000 --reload --app-dir src

run_mcp:
	uv run src/mcp_server.py --port 8001

preprocess:
	uv run src/preprocess.py

qdrant:
	docker run -p 6333:6333 -p 6334:6334 --rm \
		-v $(PWD)/datas/qdrant:/qdrant/storage \
		--name qdrant_container \
		qdrant/qdrant 

n8n:
	docker run -it --rm --name n8n -p 5678:5678 --network host \
		-v $(PWD)/datas/n8n_data:/home/node/.n8n \
		--name n8n_container \
		docker.n8n.io/n8nio/n8n

benchmark:
	PYTHONPATH=. uv run benchmark/benchmark.py

test:
	PYTHONPATH=$(PWD)/src pytest tests/ -v

format:
	@echo "Running isort..."
	isort .
	@echo "Running black..."
	black .

.PHONY: run down qdrant test format n8n benchmark
