"""Validated data contracts for nutrition document ingestion."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

class ExtractedPage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: str = Field(min_length=1, max_length=500)
    source: str = Field(min_length=1, max_length=500)
    title: str = Field(min_length=1, max_length=300)
    page: int | None = Field(default=None, ge=1)
    text: str = Field(min_length=1, max_length=2_000_000)

    @field_validator("source_id", "source", "title", "text")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

class NutritionChunk(BaseModel):
    """Portable chunk record; embeddings are added in the retrieval module."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    chunk_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    domain: Literal["nutrition"] = "nutrition"
    source_id: str = Field(min_length=1, max_length=500)
    source: str = Field(min_length=1, max_length=500)
    title: str = Field(min_length=1, max_length=300)
    page: int | None = Field(default=None, ge=1)
    chunk_number: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=4000)
    char_count: int = Field(ge=1, le=4000)
    token_estimate: int = Field(ge=1)
    metadata_version: Literal["1"] = "1"

    @field_validator("text")
    @classmethod
    def strip_chunk_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("chunk text must not be blank")
        return value

class IngestionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    files_processed: int = Field(ge=0)
    pages_with_text: int = Field(ge=0)
    empty_pages: int = Field(ge=0)
    chunks_written: int = Field(ge=0)
    output_path: str
    domain: Literal["nutrition"] = "nutrition"
