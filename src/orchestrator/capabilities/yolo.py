import os
import logging
import cv2
import numpy as np
from typing import Any
from .camera import get_frame, frame_to_base64

logger = logging.getLogger(__name__)


_ARPA_VISION_ROOT    = "/home/xle/arpa_vision"
_ARPA_VISION_WEIGHTS = "/home/xle/arpa_vision/arpa_vision/scripts/yolov8x-worldv2_best.pt"
_EV_BATTERY_QUERIES  = ["Bolt", "BusBar", "InteriorScrew", "Nut", "OrangeCover", "Screw", "Screw Hole"]

_yolo_model = None
_yolo_error: str = ""


def _get_yolo_model():
    global _yolo_model, _yolo_error
    if _yolo_model is not None:
        return _yolo_model, ""
    if _yolo_error:
        return None, _yolo_error
    try:
        import sys
        if _ARPA_VISION_ROOT not in sys.path:
            sys.path.insert(0, _ARPA_VISION_ROOT)
        from arpa_vision.scripts.BoundingBoxDetectors import YOLO_WORLD

        weight_path = os.getenv("YOLO_MODEL", _ARPA_VISION_WEIGHTS)
        _yolo_model = YOLO_WORLD(weight_file_path=weight_path)
        logger.info("YOLO_WORLD loaded on CPU: %s", weight_path)

        return _yolo_model, ""
    except Exception as exc:
        _yolo_error = f"YOLO_WORLD load failed: {exc}"
        logger.error(_yolo_error)
        return None, _yolo_error


def _build_queries(target_object: str) -> list[str]:
    if not target_object:
        return _EV_BATTERY_QUERIES
    requested = {lbl.strip().lower() for lbl in target_object.split(",") if lbl.strip()}
    matched = [q for q in _EV_BATTERY_QUERIES if q.lower() in requested]
    # If any requested labels matched, return the matched ones.
    # If none matched (invalid queries), default to all available queries.
    return matched if matched else _EV_BATTERY_QUERIES


def _run_yolo_detection(
    frame_rgb: np.ndarray,
    target_object: str,
    confidence_threshold: float = 0.20,
) -> tuple[list[dict[str, Any]], str, str]:
    model, err = _get_yolo_model()
    if model is None:
        return [], "", err
    try:
        queries = _build_queries(target_object)
        candidates_2d = model.predict(frame_rgb, queries=queries, debug=False)

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
    except Exception as exc:
        err = f"YOLO inference failed: {exc}"
        logger.error(err)
        return [], "", err


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
