from dataclasses import dataclass, field
from typing import Any, Callable
from langchain_core.tools import tool

# Import all the capability implementation functions
from .vlm import observe_with_base_camera_fn, observe_with_wrist_camera_fn, observe_with_both_cameras_fn
from .yolo import yolo_base_camera_fn, yolo_wrist_camera_fn
from .vla_policy import start_vla_policy_fn, stop_vla_policy_fn, wait_fn, POLICIES


@dataclass
class Capability:
    id: str
    description: str
    doc: str
    fn: Callable[..., dict[str, Any]]
    required_args: list[str] = field(default_factory=list)
    optional_args: list[str] = field(default_factory=list)


REGISTRY: dict[str, Capability] = {
    "observe_with_base_camera": Capability(
        id="observe_with_base_camera",
        description="Capture the base camera frame and ask the VLM a question.",
        doc="""Capability: observe_with_base_camera
        --------------------------------------
        Purpose:
        Captures the base RealSense/ZMQ camera frame and sends it to Gemini Vision
        along with your question. Returns a natural-language answer.

        The raw frame is returned as frame_b64 for the UI only — it is NOT in the
        agent context window (automatically stripped).

        Required args:  (none)
        Optional args:
        question (str) — Specific question to guide the VLM response.
                        e.g. "Is a screw visible on the table?"
                        Leave empty for a full scene description.

        Returns:
        description: VLM answer
        frame_b64:   raw base camera frame (UI only)
        error:       non-empty if something failed

        Use for: "what do you see?", "is the table clear?", "describe the workspace"
        Do NOT use for precise object coordinates — use yolo_base_camera instead.
        """,
        fn=observe_with_base_camera_fn,
        optional_args=["question"],
    ),
    "observe_with_wrist_camera": Capability(
        id="observe_with_wrist_camera",
        description=(
            "Capture the wrist (end-effector) camera frame and ask the VLM a question. "
            "Use to inspect grasp quality or confirm what is in the gripper."
        ),
        doc="""Capability: observe_with_wrist_camera
        ---------------------------------------
        Purpose:
        Captures the wrist ZMQ camera frame and sends it to Gemini Vision.
        Ideal for post-pick grasp verification.

        Required args:  (none)
        Optional args:
        question (str) — e.g. "Is the screw firmly grasped?", "What is in the gripper?"

        Returns:
        description: VLM answer
        frame_b64:   raw wrist camera frame (UI only)
        error:       non-empty if something failed

        Use for: grasp confirmation, close-up object inspection.
        """,
        fn=observe_with_wrist_camera_fn,
        optional_args=["question"],
    ),
    "observe_with_both_cameras": Capability(
        id="observe_with_both_cameras",
        description=(
            "Capture both base and wrist frames and ask the VLM a question about both. "
            "Use when full context (scene + end-effector state) is needed."
        ),
        doc="""Capability: observe_with_both_cameras
        ---------------------------------------
        Purpose:
        Captures both camera frames and runs the VLM twice (once per frame).
        Returns a combined natural-language description.

        Required args:  (none)
        Optional args:
        question (str) — Applies to both cameras, prefixed with the camera name.

        Returns:
        description: combined VLM answer (base then wrist)
        base_frame_b64:  base camera frame (UI only)
        wrist_frame_b64: wrist camera frame (UI only)
        error:       non-empty if something failed

        Use for: comprehensive state checks mid-task.
        """,
        fn=observe_with_both_cameras_fn,
        optional_args=["question"],
    ),
    "yolo_base_camera": Capability(
        id="yolo_base_camera",
        description="Capture the base camera frame and run YOLO open-vocabulary detection.",
        doc="""Capability: yolo_base_camera
        ------------------------------
        Purpose:
        Captures the base camera frame and runs YOLO_WORLD inference to locate objects.
        Returns bounding boxes with normalised pixel-space positions.

        Required args:  (none)
        Optional args:
        target_object        (str,   default "")    — object label to detect, e.g. "Screw".
                                                        Leave empty to detect all EV battery classes.
                                                        Comma-separate for multiple: "Screw,Nut".
        confidence_threshold (float, default 0.25)  — minimum confidence [0, 1].

        EV battery classes: Bolt, BusBar, InteriorScrew, Nut, OrangeCover, Screw, Screw Hole

        Returns:
        detected:  list of { id, label, confidence, bbox_px, x, y, z }
                    x/y are normalised image coords (z is None until depth is wired).
        count:     number of detections
        frame_b64: annotated frame with bounding boxes (UI only)
        error:     non-empty if something failed
        """,
        fn=yolo_base_camera_fn,
        optional_args=["target_object", "confidence_threshold"],
    ),
    "yolo_wrist_camera": Capability(
        id="yolo_wrist_camera",
        description="Capture the wrist camera frame and run YOLO detection.",
        doc="""Capability: yolo_wrist_camera
        -------------------------------
        Purpose:
        Captures the wrist camera frame and runs YOLO_WORLD inference.
        Useful for close-range object checks or post-pick confirmation.

        Required args:  (none)
        Optional args:
        target_object        (str,   default "")    — label(s) to detect.
        confidence_threshold (float, default 0.25)  — minimum confidence [0, 1].

        Returns:
        detected:  list of { id, label, confidence, bbox_px, x, y, z }
                    x/y are normalised image coords (z is None until depth is wired).
        count:     number of detections
        frame_b64: annotated frame with bounding boxes (UI only)
        error:     non-empty if something failed
        """,
        fn=yolo_wrist_camera_fn,
        optional_args=["target_object", "confidence_threshold"],
    ),
    "start_vla_policy": Capability(
        id="start_vla_policy",
        description=(
            "Launch the async VLA inference client using a predefined policy ID from the registry. "
            f"Available policy IDs: {list(POLICIES.keys())}."
        ),
        doc=(
            "Capability: start_vla_policy\n"
            "------------------------------\n"
            "Purpose:\n"
            "  Looks up the policy_id in policies.yaml and dispatches the lerobot async\n"
            "  inference robot client as a background subprocess.\n\n"
            "  Available policies (from policies.yaml):\n"
            + "\n".join(
                f"    {pid}: task=\"{cfg['task']}\", type={cfg['policy_type']}, typical_wait_time={cfg.get('typical_wait_time', 30)}s"
                for pid, cfg in POLICIES.items()
            )
            + "\n\n"
            "Required args:\n"
            "  policy_id (str) — ID from the list above, e.g. 'screw_picking'.\n\n"
            "Returns:\n"
            "  status:           \"SUCCESS\" or \"FAILURE\"\n"
            "  message:          confirmation with task description\n"
            "  pid:              OS process ID of the launched client\n"
            "  typical_wait_time: recommended seconds to wait before observing\n\n"
            "IMPORTANT: This call BLOCKS until the model is loaded and action chunks are\n"
            "flowing (or until the 5-minute timeout). On SUCCESS the robot is already moving.\n"
            "Use the returned typical_wait_time as your first wait duration, then observe\n"
            "to verify success.  Repeat wait+observe until done, then call stop_vla_policy."
        ),
        fn=start_vla_policy_fn,
        required_args=["policy_id"],
    ),
    "stop_vla_policy": Capability(
        id="stop_vla_policy",
        description="Terminate the currently running VLA inference client.",
        doc="""Capability: stop_vla_policy
        -----------------------------
        Purpose:
        Terminates the background async inference robot client (SIGTERM, then SIGKILL
        if it does not exit within 5 seconds).

        Required args:  (none)

        Returns:
        status:  "SUCCESS" or "WARNING" (if nothing was running)
        message: confirmation string
        """,
        fn=stop_vla_policy_fn,
    ),
    "wait": Capability(
        id="wait",
        description="Pause agent execution for N seconds to allow physical operations to complete.",
        doc="""Capability: wait
        -----------------
        Purpose:
        Block the agent for the given number of seconds.
        Use after start_vla_policy to give the robot time to execute the policy.

        Required args:
        seconds (int) — number of seconds to wait, e.g. 20.

        Returns:
        status:  "SUCCESS"
        message: confirmation string
        """,
        fn=wait_fn,
        required_args=["seconds"],
    ),
}


def list_capabilities() -> dict[str, Any]:
    return {
        "capabilities": [
            {
                "id": cap.id,
                "description": cap.description,
                "required_args": cap.required_args,
                "optional_args": cap.optional_args,
            }
            for cap in REGISTRY.values()
        ]
    }


def read_capability(capability_id: str) -> dict[str, Any]:
    cap = REGISTRY.get(capability_id)
    if cap is None:
        return {"error": f"Unknown capability: {capability_id!r}. Available: {list(REGISTRY.keys())}"}
    return {
        "id": cap.id,
        "description": cap.description,
        "doc": cap.doc.strip(),
        "required_args": cap.required_args,
        "optional_args": cap.optional_args,
    }


def run_capability(capability_id: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    cap = REGISTRY.get(capability_id)
    if cap is None:
        return {"error": f"Unknown capability: {capability_id!r}. Available: {list(REGISTRY.keys())}"}
    args = args or {}
    missing = [k for k in cap.required_args if k not in args]
    if missing:
        return {
            "error": f"Missing required args for {capability_id!r}: {missing}",
            "hint": f"Call read_capability('{capability_id}') to see the full contract.",
        }
    try:
        return cap.fn(**args)
    except Exception as exc:
        return {"error": str(exc), "capability_id": capability_id}

# ---------------------------------------------------------------------------
# LangChain tool wrappers
# ---------------------------------------------------------------------------


@tool
def list_capabilities_tool(**kwargs: Any) -> dict[str, Any]:
    """List all available robot capabilities with their required and optional args."""
    return list_capabilities()


@tool
def read_capability_tool(capability_id: str, **kwargs: Any) -> dict[str, Any]:
    """Read the full documentation and argument contract for one capability before using it."""
    return read_capability(capability_id)


@tool
def run_capability_tool(
    capability_id: str,
    args: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute a robot capability with validated arguments."""
    merged: dict[str, Any] = {}
    if isinstance(args, dict):
        merged.update(args)
    if isinstance(params, dict):
        merged.update(params)
    skip_keys = {"v__args", "type"}
    wrapper_keys = {"kwargs", "parameters", "payload", "input"}
    for key, value in kwargs.items():
        if key in skip_keys:
            continue
        if key in wrapper_keys and isinstance(value, dict):
            merged.update(value)
        else:
            merged[key] = value
    return run_capability(capability_id, merged or None)


TOOLS = [
    list_capabilities_tool,
    read_capability_tool,
    run_capability_tool,
]
