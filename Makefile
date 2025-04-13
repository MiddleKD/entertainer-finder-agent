run:
	uvicorn main:app --host 127.0.0.1 --port 8000 --reload

qdrant:
	docker run -p 6333:6333 -p 6334:6334 \
		-v $(pwd)/datas/qdrant:/qdrant/storage \
		qdrant/qdrant 

test:
	PYTHONPATH=$(PWD)/src pytest tests/ -v

.PHONY: run qdrant test 