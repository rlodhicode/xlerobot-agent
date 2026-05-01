"""
yolo_server.py  —  run this on the laptop
    python yolo_server.py
    python yolo_server.py --host 0.0.0.0 --port 8081 --weights /path/to/weights.pt
"""
import argparse
import base64
import logging
import os
import sys

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_ARPA_VISION_ROOT    = r"C:\Users\rayaa\Downloads\arpa_vision"
_ARPA_VISION_WEIGHTS = r"C:\Users\rayaa\Downloads\arpa_vision\arpa_vision\scripts\yolov8x-worldv2_best.pt"

app = FastAPI(title="YOLO Inference Server")

_model = None


def _load_model(weight_path: str):
    global _model
    if _ARPA_VISION_ROOT not in sys.path:
        sys.path.insert(0, _ARPA_VISION_ROOT)
    from arpa_vision.scripts.BoundingBoxDetectors import YOLO_WORLD
    _model = YOLO_WORLD(weight_file_path=weight_path)
    logger.info("YOLO_WORLD loaded from %s", weight_path)


class InferRequest(BaseModel):
    frame_b64: str          # base64-encoded JPEG/PNG
    queries: list[str]
    debug: bool = False


class InferResponse(BaseModel):
    candidates_2d: dict     # {label: {boxes: [...], probs: [...]}}
    error: str = ""


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        img_bytes = base64.b64decode(req.frame_b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)       # BGR
        if frame is None:
            raise ValueError("Could not decode image")
        candidates = _model.predict(frame, queries=req.queries, debug=req.debug)
        return InferResponse(candidates_2d=candidates)
    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        return InferResponse(candidates_2d={}, error=str(exc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",    default="0.0.0.0")
    parser.add_argument("--port",    type=int, default=8081)
    parser.add_argument("--weights", default=os.getenv("YOLO_MODEL", _ARPA_VISION_WEIGHTS))
    args = parser.parse_args()

    _load_model(args.weights)
    uvicorn.run(app, host=args.host, port=args.port)
