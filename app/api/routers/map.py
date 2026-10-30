from fastapi import APIRouter, Query

from app.api.map_payload import build_map_payload

router = APIRouter()


@router.get("/api/map")
def get_map(dataset: str = Query(..., description="Dataset id: dataset-a | dataset-b")):
    """Static geometry + optional per-station anomaly list (JSON under app/core/data/)."""
    return build_map_payload(dataset)
