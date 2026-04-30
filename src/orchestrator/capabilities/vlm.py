import os
import base64
import logging
from typing import Any
from .camera import get_frame, frame_to_base64, frame_to_base64_vlm

logger = logging.getLogger(__name__)

# TODO: generalize or dynamically inject different world model conditioning for different tasks
_VLM_PREAMBLE_BASE = (
    "CONTEXT: You are analysing an image from the fixed base (overview) camera "
    "of a robot manipulation workstation. The workspace contains EV battery components: "
    "screws, nuts, bolts, orange plastic covers, bus bars, and small containers/trays "
    "used to collect picked parts. The robot arm may appear at the edge of the frame. "
    "Answer concisely and literally \n\nQUESTION: "
)

_VLM_PREAMBLE_WRIST = (
    "CONTEXT: You are analysing an image from the wrist-mounted camera on a robot arm. "
    "The black ribbed cylindrical structures visible are the robot's parallel-jaw gripper "
    "fingers (end-effector). A successfully grasped object will appear between or pressed "
    "against those gripper fingers. The workspace contains EV battery components: screws, "
    "nuts, bolts, and small plastic containers. "
    "Answer concisely and literally \n\nQUESTION: "
)


def _run_vlm_description(frame_b64: str, question: str) -> tuple[str, str]:
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel, Part

        project  = os.getenv("VERTEX_PROJECT_ID", "")
        location = os.getenv("VERTEX_LOCATION", "us-central1")
        vlm_model_name = os.getenv("VLM_MODEL", "gemini-2.5-flash-lite")

        if not project:
            return "", "VERTEX_PROJECT_ID not set in .env"

        vertexai.init(project=project, location=location)
        model = GenerativeModel(vlm_model_name)

        image_bytes = base64.b64decode(frame_b64)
        image_part = Part.from_data(data=image_bytes, mime_type="image/jpeg")
        text_part = question or (
            "Describe this robot workspace image in detail. "
            "List all visible objects, their approximate positions relative to each other, "
            "and anything relevant to a pick-and-place robot task."
        )

        response = model.generate_content([image_part, text_part])
        return response.text.strip(), ""
    except ImportError:
        err = "vertexai SDK not installed. Run: pip install google-cloud-aiplatform"
        logger.error(err)
        return "", err
    except Exception as exc:
        err = f"VLM inference failed: {exc}"
        logger.error(err)
        return "", err


def observe_with_base_camera_fn(question: str = "", **kwargs: Any) -> dict[str, Any]:
    frame, err = get_frame("base")
    if frame is None:
        return {"capability": "observe_with_base_camera", "question": question,
                "description": "", "frame_b64": "", "error": err}
    vlm_b64 = frame_to_base64_vlm(frame)
    enriched = _VLM_PREAMBLE_BASE + (question or "Describe the full scene.")
    description, vlm_err = _run_vlm_description(vlm_b64, enriched)
    return {
        "capability": "observe_with_base_camera",
        "camera": "base",
        "question": question,
        "description": description,
        "frame_b64": frame_to_base64(frame),
        "error": vlm_err or err,
    }


def observe_with_wrist_camera_fn(question: str = "", **kwargs: Any) -> dict[str, Any]:
    frame, err = get_frame("wrist")
    if frame is None:
        return {"capability": "observe_with_wrist_camera", "question": question,
                "description": "", "frame_b64": "", "error": err}
    vlm_b64 = frame_to_base64_vlm(frame)
    enriched = _VLM_PREAMBLE_WRIST + (question or "Describe what is in or near the gripper.")
    description, vlm_err = _run_vlm_description(vlm_b64, enriched)
    return {
        "capability": "observe_with_wrist_camera",
        "camera": "wrist",
        "question": question,
        "description": description,
        "frame_b64": frame_to_base64(frame),
        "error": vlm_err or err,
    }


def observe_with_both_cameras_fn(question: str = "", **kwargs: Any) -> dict[str, Any]:
    base_frame, base_err = get_frame("base")
    wrist_frame, wrist_err = get_frame("wrist")

    descriptions: list[str] = []
    combined_err = " | ".join(filter(None, [base_err, wrist_err]))

    if base_frame is not None:
        enriched = _VLM_PREAMBLE_BASE + (question or "Describe the full scene.")
        desc, vlm_err = _run_vlm_description(frame_to_base64_vlm(base_frame), enriched)
        if vlm_err:
            combined_err = " | ".join(filter(None, [combined_err, f"Base VLM: {vlm_err}"]))
        else:
            descriptions.append(f"Base camera: {desc}")

    if wrist_frame is not None:
        enriched = _VLM_PREAMBLE_WRIST + (question or "Describe what is in or near the gripper.")
        desc, vlm_err = _run_vlm_description(frame_to_base64_vlm(wrist_frame), enriched)
        if vlm_err:
            combined_err = " | ".join(filter(None, [combined_err, f"Wrist VLM: {vlm_err}"]))
        else:
            descriptions.append(f"Wrist camera: {desc}")

    return {
        "capability": "observe_with_both_cameras",
        "question": question,
        "description": "\n\n".join(descriptions),
        "base_frame_b64":  frame_to_base64(base_frame)  if base_frame  is not None else "",
        "wrist_frame_b64": frame_to_base64(wrist_frame) if wrist_frame is not None else "",
        "error": combined_err,
    }
