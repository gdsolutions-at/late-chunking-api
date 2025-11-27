# code taken from https://github.com/jina-ai/late-chunking/tree/main
start with

    uv run fastapi dev late_chunking_api.py --host localhost --port 8888
# deployment via docker
https://docs.astral.sh/uv/guides/integration/fastapi/#migrating-an-existing-fastapi-project

## example curl
curl --location 'http://localhost:8888/chunk' \
--header 'Content-Type: application/json' \
--data '
{
    "doc_id": "test-2", "chunk_mode": "semantic","chunk_size": 200,
    "text": "Ich bin ein lustiger Text mitvielen Absätzen."
  }