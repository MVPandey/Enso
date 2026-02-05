"""Configuration for the application."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration for the application."""

    model_config = SettingsConfigDict(env_file='.env')

    huggingface_api_token: str | None = Field(description='The API token for the Hugging Face API.', default=None)
    kaggle_api_token: str | None = Field(description='The API token for the Kaggle API.', default=None)
    data_dir: Path = Field(default=Path('data'), description='Directory where datasets are stored.')


config = Config()  # type: ignore[missing-argument]
