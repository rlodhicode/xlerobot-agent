import os
import io
import base64
import logging
import numpy as np
from typing import Any
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

_camera_base = None
_camera_wrist = None
_camera_base_error: str = ""
_camera_wrist_error: str = ""

ZMQ_HOST: str = os.getenv("ZMQ_SERVER_HOST", "localhost")
ZMQ_PORT: int = int(os.getenv("ZMQ_SERVER_PORT", "5555"))


def _init_zmq_camera(camera_name: str) -> tuple[Any, str]:
    try:
        from lerobot.cameras.zmq.camera_zmq import ZMQCamera
        from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig

        config = ZMQCameraConfig(
            server_address=ZMQ_HOST,
            port=ZMQ_PORT,
            camera_name=camera_name,
            timeout_ms=1000,
            warmup_s=1,
        )
        cam = ZMQCamera(config)
        cam.connect(warmup=True)
        logger.info("ZMQ camera '%s' connected at %s:%s.", camera_name, ZMQ_HOST, ZMQ_PORT)
        return cam, ""
    except Exception as exc:
        err = f"ZMQ camera '{camera_name}' init failed: {exc}"
        logger.warning(err)
        return None, err


def get_frame(camera_name: str) -> tuple[np.ndarray | None, str]:
    """Return (frame_rgb, error) for the named camera ('base' or 'wrist')."""
    global _camera_base, _camera_base_error, _camera_wrist, _camera_wrist_error

    if camera_name == "base":
        if _camera_base is None:
            _camera_base, _camera_base_error = _init_zmq_camera("base")
        if _camera_base is not None:
            try:
                return _camera_base.read(), ""
            except Exception as exc:
                return None, f"Base camera read failed: {exc}"
        return None, _camera_base_error

    if camera_name == "wrist":
        if _camera_wrist is None:
            _camera_wrist, _camera_wrist_error = _init_zmq_camera("wrist")
        if _camera_wrist is not None:
            try:
                return _camera_wrist.read(), ""
            except Exception as exc:
                return None, f"Wrist camera read failed: {exc}"
        return None, _camera_wrist_error

    return None, f"Unknown camera name: '{camera_name}'. Use 'base' or 'wrist'."


def frame_to_base64(frame_rgb: np.ndarray) -> str:
    """Encode (H, W, 3) uint8 RGB array as a base64 PNG string (for UI display)."""
    img = PILImage.fromarray(frame_rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def frame_to_base64_vlm(frame_rgb: np.ndarray, max_dim: int = 400, jpeg_quality: int = 85) -> str:
    """Resize + JPEG-encode a frame for VLM submission (minimises token cost)."""
    h, w = frame_rgb.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = PILImage.fromarray(frame_rgb.astype(np.uint8), mode="RGB")
        img = img.resize((new_w, new_h), PILImage.LANCZOS)
    else:
        img = PILImage.fromarray(frame_rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
