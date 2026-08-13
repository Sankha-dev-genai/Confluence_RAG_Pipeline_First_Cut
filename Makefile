.PHONY: install ingest app eval test lint docker-build docker-run clean

install:      ## install runtime + dev deps
	pip install -r requirements.txt -r requirements-dev.txt

ingest:       ## run the full ingestion pipeline for the active SOURCE
	python main.py

reindex:      ## rebuild embeddings + FAISS from existing chunks
	python main.py --steps chunk embed index

app:          ## launch the Streamlit demo
	streamlit run streamlit_app.py

eval:         ## run retrieval evaluation
	python scripts/run_evaluation.py

test:         ## run unit tests
	pytest -q

lint:         ## static checks
	ruff check .

docker-build:
	docker build -t knowledge-rag .

docker-run:
	docker run --rm -p 8501:8501 --env-file .env knowledge-rag

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache

benchmark-generate:  ## generate the synthetic benchmark corpus
	python scripts/generate_synthetic_corpus.py --services 40 --questions-per-service 2

benchmark-build:     ## embed + index the benchmark corpus (needs the model)
	python scripts/build_benchmark.py

benchmark:           ## dense vs hybrid comparison on the benchmark
	python scripts/run_benchmark.py
