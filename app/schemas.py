"""Pydantic models for API responses.

Requests mostly come in as multipart form-data (files + fields), which FastAPI
handles directly, so most of these are response shapes.
"""

from __future__ import annotations

from pydantic import BaseModel


class EnrollResponse(BaseModel):
    person_id: int
    name: str
    faces_enrolled: int
    message: str


class Match(BaseModel):
    box: list[int]                 # [x1, y1, x2, y2]
    name: str                      # matched person or "Unknown"
    person_id: int | None
    distance: float | None         # cosine distance to nearest person
    confidence: float | None       # 0..1 display value
    det_score: float               # detector confidence


class RecognizeResponse(BaseModel):
    faces_found: int
    threshold: float
    matches: list[Match]


class PersonOut(BaseModel):
    id: int
    name: str
    image_count: int
    thumbnail: str | None          # URL to a thumbnail image


class ConfigOut(BaseModel):
    threshold: float
    model: str
    metric: str
