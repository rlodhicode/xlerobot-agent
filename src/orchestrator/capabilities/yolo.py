import os
import io
import base64
import logging
import cv2
import numpy as np
from typing import Any
from PIL import Image as PILImage
from .camera import get_frame, frame_to_base64

logger = logging.getLogger(__name__)

_YOLO_SERVER = os.getenv("YOLO_SERVER", "http://192.168.50.42:8081")
_EV_BATTERY_QUERIES = ["Bolt", "BusBar", "InteriorScrew", "Nut", "OrangeCover", "Screw", "Screw Hole"]


def _frame_to_b64_jpeg(frame_rgb: np.ndarray, quality: int = 90) -> str:
    """Encode RGB frame as base64 JPEG for sending over the wire."""
    img = PILImage.fromarray(frame_rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_queries(target_object: str) -> list[str]:
    if not target_object:
        return _EV_BATTERY_QUERIES
    requested = {lbl.strip().lower() for lbl in target_object.split(",") if lbl.strip()}
    matched = [q for q in _EV_BATTERY_QUERIES if q.lower() in requested]
    return matched if matched else _EV_BATTERY_QUERIES


def _run_yolo_detection(
    frame_rgb: np.ndarray,
    target_object: str,
    confidence_threshold: float = 0.20,
) -> tuple[list[dict[str, Any]], str, str]:
    try:
        import requests
    except ImportError:
        return [], "", "requests library not installed. Run: pip install requests"

    try:
        queries = _build_queries(target_object)
        payload = {
            "frame_b64": _frame_to_b64_jpeg(frame_rgb),
            "queries": queries,
            "debug": False,
        }
        resp = requests.post(f"{_YOLO_SERVER}/infer", json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        err = f"YOLO server request failed: {exc}"
        logger.error(err)
        return [], "", err

    if data.get("error"):
        return [], "", data["error"]

    candidates_2d = data["candidates_2d"]
    detections: list[dict[str, Any]] = []
    h, w = frame_rgb.shape[:2]
    annotated = frame_rgb.copy()
    det_idx = 0

    for query, preds in candidates_2d.items():
        for box, prob in zip(preds["boxes"], preds["probs"]):
            if prob < confidence_threshold:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            label_id = query.lower().replace(" ", "_")
            detections.append({
                "id": f"{label_id}_{det_idx}",
                "label": query,
                "confidence": round(prob, 3),
                "bbox_px": [x1, y1, x2, y2],
                "x": round(cx / w - 0.5, 4),
                "y": round(cy / h - 0.5, 4),
                "z": None,
                "note": "x/y are normalised image coords; wire depth for metric 3-D.",
            })
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated, f"{query} {prob:.2f}", (x1, max(y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
            det_idx += 1

    return detections, frame_to_base64(annotated), ""


def yolo_base_camera_fn(target_object: str = "", confidence_threshold: float = 0.25, **kwargs: Any) -> dict[str, Any]:
    frame, err = get_frame("base")
    if frame is None:
        return {"capability": "yolo_base_camera", "detected": [], "count": 0, "frame_b64": "", "error": err}
    detections, annotated_b64, yolo_err = _run_yolo_detection(frame, target_object, confidence_threshold)
    return {
        "capability": "yolo_base_camera",
        "camera": "base",
        "target": target_object or "(all EV battery classes)",
        "detected": detections,
        "count": len(detections),
        "frame_b64": annotated_b64 or frame_to_base64(frame),
        "error": yolo_err or err,
    }


def yolo_wrist_camera_fn(target_object: str = "", confidence_threshold: float = 0.25, **kwargs: Any) -> dict[str, Any]:
    frame, err = get_frame("wrist")
    if frame is None:
        return {"capability": "yolo_wrist_camera", "detected": [], "count": 0, "frame_b64": "", "error": err}
    detections, annotated_b64, yolo_err = _run_yolo_detection(frame, target_object, confidence_threshold)
    return {
        "capability": "yolo_wrist_camera",
        "camera": "wrist",
        "target": target_object or "(all EV battery classes)",
        "detected": detections,
        "count": len(detections),
        "frame_b64": annotated_b64 or frame_to_base64(frame),
        "error": yolo_err or err,
    }
