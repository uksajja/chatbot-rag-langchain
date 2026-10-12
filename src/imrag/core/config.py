from __future__ import annotations
from typing import Any, Optional, Literal
from pathlib import Path
from pydantic import BaseModel, Field   
import copy
import yaml

class ConfigLoader:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.config = self.load_config()

    def load_config(self) -> Config:
        return Config(**self.load_config_data())

    def load_config_data(self) -> dict[str, Any]:
        with open(self.config_path, "r") as f:
            config_data = yaml.safe_load(f)
        return config_data or {}

class ConfigManager:
    def __init__(
        self,
        base_config_path: str | Path,
        override_config_path: str | Path | list[str | Path] | None = None,
    ):
        self.base_config_path = Path(base_config_path)
        if isinstance(override_config_path, (str, Path)):
            self.override_config_paths = [Path(override_config_path)]
        else:
            self.override_config_paths = [Path(path) for path in (override_config_path or [])]
        self.config = self.load_and_merge_configs()

    def load_and_merge_configs(self) -> Config:
        merged_dict = ConfigLoader(self.base_config_path).load_config_data()
        for override_path in self.override_config_paths:
            override_dict = ConfigLoader(override_path).load_config_data()
            merged_dict = self.merge_dicts(merged_dict, override_dict)
        return Config(**merged_dict)

    @staticmethod
    def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = copy.deepcopy(base)
        for key, value in override.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = ConfigManager.merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

class AppConfig(BaseModel):
    name: str="imrag"
    env: str="dev"

class IngestionConfig(BaseModel):
    """Configuration for the ingestion stage."""
    chunk_cache: str = Field(..., description="Path to store chunked data.")
    batch_size: int = Field(10, description="Number of documents to process in a batch.")

class ComponentConfig(BaseModel):
    """Base class for component configuration."""

    type: str = Field(..., description="Type of the component.")
    params: Optional[dict] = Field(
        default_factory=dict, description="Parameters for the component."
    )

class LoadingConfig(BaseModel):
    data_dir: str = "data/raw"
    recursive: bool = True
    on_error: Literal["skip", "raise"] = "skip"
    loaders: dict[str, ComponentConfig] = Field(default_factory=dict)

class ChunkingConfig(ComponentConfig):
    pass

class EmbeddingConfig(ComponentConfig):
    pass

class VectorStoreConfig(ComponentConfig):
    pass

class LLMConfig(ComponentConfig):
    pass

class PromptConfig(BaseModel):
    path: str = "configs/system_prompts.yaml"

class RetrieverConfig(ComponentConfig):
    pass

class GuardrailsConfig(BaseModel):
    min_sources: int = 1

class Config(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    ingestion: IngestionConfig
    loading: LoadingConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vectorstore: VectorStoreConfig
    retrieval: RetrieverConfig
    reranker: ComponentConfig | None = None
    llm: LLMConfig
    prompt: PromptConfig
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
