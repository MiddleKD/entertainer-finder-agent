run:
	uvicorn main:app --host 127.0.0.1 --port 8000 --reload

preprocess:
	uv run src/preprocess.py

qdrant:
	docker run -p 6333:6333 -p 6334:6334 \
		-v $(PWD)/datas/qdrant:/qdrant/storage \
		qdrant/qdrant 

n8n:
	docker run -it --rm --name n8n -p 5678:5678 --network host -v n8n_data:/home/node/.n8n docker.n8n.io/n8nio/n8n

test:
	PYTHONPATH=$(PWD)/src pytest tests/ -v

format:
	@echo "Running isort..."
	isort .
	@echo "Running black..."
	black .

.PHONY: run qdrant test format n8n 