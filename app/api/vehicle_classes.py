"""Vehicle class labels and colors (inherited from the original traffic dataset)."""

from __future__ import annotations

VEHICLE_CLASS_LABELS: dict[int, str] = {
    1: "motorcycle",
    2: "car+trailer",
    3: "truck",
    5: "bus",
    7: "car",
    8: "truck+trailer",
    9: "tractor+trailer",
    10: "unknown-10",
    11: "LCV",
}

VEHICLE_CLASS_COLORS: dict[int, str] = {
    1: "#999999",
    2: "#56B4E9",
    3: "#D55E00",
    5: "#CC79A7",
    7: "#009E73",
    8: "#F0E442",
    9: "#0072B2",
    10: "#E69F00",
    11: "#882255",
}

FALLBACK_CLASS_COLOR = "#AAAAAA"


def class_label(cls_id: int) -> str:
    return VEHICLE_CLASS_LABELS.get(cls_id, f"cls{cls_id}")


def class_color(cls_id: int) -> str:
    return VEHICLE_CLASS_COLORS.get(cls_id, FALLBACK_CLASS_COLOR)


def api_class_metadata() -> dict[str, dict[str, str]]:
    labels = {str(k): v for k, v in VEHICLE_CLASS_LABELS.items()}
    colors = {str(k): v for k, v in VEHICLE_CLASS_COLORS.items()}
    return {"labels": labels, "colors": colors}
