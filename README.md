## Injection Molding Chatbot - A RAG Based Application

### v1.0.0

- Configuration is supported through yaml files
  - models can be selected in the configuration file --> base-config.yaml 

- Load couments
- Chunking
- Embedding
- Vectorstore
- Retrieval
- Rerank
- LLM response

### Steps to create the project

1. **Set up the development environment**: 
    - uv init --python 3.12 --no-workspace
    - **Edit pyproject.toml to include required libraries in dependencies[]**
    - uv sync

Usage:

python .\src\main.py --config "configs/base-config-nvidia.yaml" --ingest -q "What is Anisotropy?"