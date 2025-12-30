uv run uvicorn main:app 

npx inngest-cli@latest dev -u http://127.0.0.1:8000

docker run -d --name qdrantRagDb -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant