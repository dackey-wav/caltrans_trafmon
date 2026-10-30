"""Shared CLI parsing helpers."""


def split_locations(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]
