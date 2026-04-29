"""Robot capability registry.

Capabilities exposed to the agent:

  observe_with_base_camera   — capture base frame, call VLM with a question.
  observe_with_wrist_camera  — capture wrist frame, call VLM with a question.
  observe_with_both_cameras  — capture both frames, call VLM with a question.

  yolo_base_camera           — run YOLO on the base camera frame.
  yolo_wrist_camera          — run YOLO on the wrist camera frame.

  start_vla_policy           — launch the async inference client from the policy
                               registry (policies.yaml).  Requires only policy_id.
  stop_vla_policy            — terminate the running inference client.

  wait                       — pause agent execution for N seconds.

Camera backend:
  Both cameras are served by the lerobot ZMQ camera-server.
  Configure via .env:
    ZMQ_SERVER_HOST  (default: localhost)
    ZMQ_SERVER_PORT  (default: 5555)

VLA inference:
  Configure via .env:
    VLA_INFERENCE_SERVER  (default: 192.168.50.42:8080)
    ROBOT_PORT            (default: /dev/ttyACM0)
    ROBOT_ID              (default: left_arm)

YOLO backend:
  Uses arpa_vision YOLO_WORLD with EV battery weights by default.
  Override weight path via YOLO_MODEL= in .env.

VLM backend:
  Uses Gemini via Vertex AI.
  VERTEX_PROJECT_ID and GOOGLE_APPLICATION_CREDENTIALS must be set.
"""