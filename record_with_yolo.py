"""
Leader-follower teleoperation recording with per-episode YOLO screw detection.

Usage:

    python record_with_yolo.py \\
        --robot.type=so101_follower \\
        --robot.port=/dev/ttyACM0 \\
        --robot.id=left_arm \\
        --robot.cameras='{
            realsense: {type: intelrealsense, serial_number_or_name: "Intel RealSense D435", fps: 30, width: 640, height: 480},
            wrist:     {type: opencv, index_or_path: 2, fps: 30, width: 640, height: 480}
        }' \\
        --teleop.type=so101_leader \\
        --teleop.port=/dev/ttyACM1 \\
        --teleop.id=bmw_leader_arm \\
        --dataset.repo_id=<hf_user>/<dataset_name> \\
        --dataset.single_task="Pick up screw and place in hole" \\
        --dataset.num_episodes=50 \\
        --dataset.fps=30 \\
        --dataset.push_to_hub=false

Headless controls (type + Enter):
    Enter / s  → save / end episode early
    r          → discard and re-record
    q          → stop recording

Between episodes, the leader arm drives the follower arm in the background so you
can position it before pressing Enter to start recording.

YOLO feature:
  Before each episode, YOLO runs on the RealSense frame to detect screws.
  The result is stored as 'observation.yolo.screw_xyz' — a float32 array of
  shape (MAX_SCREWS * 3,) = (15,) containing [x0,y0,z0, x1,y1,z1, ...] for up
  to MAX_SCREWS detections, zero-padded. x/y are normalised image coords
  (-0.5…+0.5), z is metres from the camera. The array is stamped onto every
  frame of the episode and is compatible with lerobot training scripts.
"""

import logging
import select
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

import numpy as np

# Make lerobot importable
sys.path.insert(0, "/home/xle/lerobot/src")
# Make yolo_camera importable
sys.path.insert(0, str(Path(__file__).parent))

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import Robot, RobotConfig, make_robot_from_config, so_follower  # noqa: F401
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
    so_leader,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from yolo_camera import MAX_SCREWS, observe_screws

logger = logging.getLogger(__name__)

# ── YOLO feature definition ──────────────────────────────────────────────────
# Coordinate names used for the flat float32 array stored in each frame.
# Proper names ensure training scripts (e.g. lerobot_train) can introspect the
# feature without issues.
YOLO_FEATURE_KEY = "observation.yolo.screw_xyz"
YOLO_COORD_NAMES = [f"{ax}{i}" for i in range(MAX_SCREWS) for ax in ("x", "y", "z")]
YOLO_FEATURE = {
    "dtype": "float32",
    "shape": (MAX_SCREWS * 3,),
    "names": YOLO_COORD_NAMES,
}


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DatasetRecordConfig:
    repo_id: str
    single_task: str
    root: str | Path | None = None
    fps: int = 30
    episode_time_s: int | float = 60
    reset_time_s: int | float = 60
    num_episodes: int = 50
    video: bool = True
    push_to_hub: bool = True
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    vcodec: str = "libsvtav1"
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("Provide --dataset.single_task.")


@dataclass
class RecordWithYoloConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    teleop: TeleoperatorConfig | None = None
    display_data: bool = False
    display_ip: str | None = None
    display_port: int | None = None
    display_compressed_images: bool = False
    play_sounds: bool = True
    resume: bool = False
    yolo_target: str = "screw"
    yolo_confidence: float = 0.25

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            from lerobot.configs.policies import PreTrainedConfig
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            self.policy = None

        if self.teleop is None and self.policy is None:
            raise ValueError("Provide --teleop or --policy.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# ---------------------------------------------------------------------------
# YoloDatasetWrapper
#
# Solves two problems:
#
#   1. build_dataset_frame (called inside record_loop) iterates dataset.features
#      and for each float32 feature does values[name] for name in ft["names"].
#      The YOLO feature has names ["x0","y0",...] which do NOT exist in the flat
#      robot observation dict → KeyError.  The wrapper hides YOLO from .features
#      so build_dataset_frame skips it.
#
#   2. The YOLO value (constant per episode) must still land in every saved frame.
#      The wrapper injects it in add_frame before the real dataset stores the frame.
#
# The underlying LeRobotDataset was created WITH the YOLO feature registered, so
# it stores and indexes the data correctly.  lerobot_train loads the raw dataset
# directly and sees all features including YOLO — no special wrapper is needed
# at training time.
# ---------------------------------------------------------------------------

class YoloDatasetWrapper:
    """
    Thin proxy around LeRobotDataset for recording.

    - .features  → hides YOLO_FEATURE_KEY so build_dataset_frame doesn't crash.
    - .add_frame → injects the current episode's YOLO coords before saving.
    - Everything else → forwarded to the underlying dataset unchanged.
    """

    def __init__(self, dataset: LeRobotDataset):
        self._dataset = dataset
        self._yolo_xyz: np.ndarray = np.zeros(MAX_SCREWS * 3, dtype=np.float32)
        # Pre-compute the feature view without YOLO for build_dataset_frame
        self._robot_features = {
            k: v for k, v in dataset.features.items() if k != YOLO_FEATURE_KEY
        }

    def set_episode_yolo(self, xyz: np.ndarray) -> None:
        """Call once before each episode with the YOLO scan result."""
        self._yolo_xyz = xyz.copy()

    # ── Overrides ────────────────────────────────────────────────────────────

    @property
    def features(self) -> dict:
        """Return features WITHOUT YOLO so build_dataset_frame works correctly."""
        return self._robot_features

    def add_frame(self, frame: dict) -> None:
        """Inject this episode's YOLO coordinates, then delegate to real dataset."""
        frame[YOLO_FEATURE_KEY] = self._yolo_xyz.copy()
        return self._dataset.add_frame(frame)

    # ── Transparent proxy ────────────────────────────────────────────────────

    def __getattr__(self, name: str):
        return getattr(self._dataset, name)

    def __repr__(self) -> str:
        return f"YoloDatasetWrapper({self._dataset!r})"


# ---------------------------------------------------------------------------
# Background teleop loop
#
# Runs in a daemon thread while the operator positions the arm between episodes.
# Keeps leader → follower active so the arm can be moved to the start pose.
# ---------------------------------------------------------------------------

def _run_teleop_background(
    robot: Robot,
    teleop: Teleoperator,
    teleop_action_processor,
    robot_action_processor,
    stop_event: threading.Event,
    fps: int = 30,
) -> None:
    """Continuously apply leader → follower at `fps` Hz until stop_event is set."""
    while not stop_event.is_set():
        t0 = time.perf_counter()
        try:
            obs = robot.get_observation()
            act = teleop.get_action()
            act_processed = teleop_action_processor((act, obs))
            robot_action = robot_action_processor((act_processed, obs))
            robot.send_action(robot_action)
        except Exception as e:
            logger.debug(f"Background teleop step failed: {e}")
        precise_sleep(max(0.0, 1 / fps - (time.perf_counter() - t0)))


# ---------------------------------------------------------------------------
# Headless-compatible keyboard listener  (no X display required)
# ---------------------------------------------------------------------------

def _init_keyboard_listener():
    """
    Returns (listener, events, proceed_event).

    Tries lerobot's pynput listener first (when X display is present), then
    falls back to a stdin-based reader for SSH / headless environments.

    events keys:
        exit_early, rerecord_episode, stop_recording, waiting_for_proceed

    proceed_event is set when:
        - operator presses Enter (stdin mode) / any key (pynput mode)
        - exit_early or stop_recording is set
    """
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "waiting_for_proceed": False,
    }
    proceed_event = threading.Event()

    # Try pynput-based listener (requires X display)
    try:
        from lerobot.utils.control_utils import is_headless, init_keyboard_listener
        if not is_headless():
            listener, lerobot_events = init_keyboard_listener()

            class _Proxy:
                """Forward to lerobot events and also fire proceed_event."""
                def __getitem__(self, k):
                    return lerobot_events[k]

                def __setitem__(self, k, v):
                    lerobot_events[k] = v
                    if k in ("exit_early", "stop_recording") and v:
                        proceed_event.set()

                def get(self, k, default=None):
                    return lerobot_events.get(k, default)

            return listener, _Proxy(), proceed_event
    except Exception:
        pass

    # Headless stdin fallback
    print(
        "\n  Headless controls (type + Enter):\n"
        "    Enter / s  → save / end episode early\n"
        "    r          → discard and re-record\n"
        "    q          → stop recording\n"
        "    (any key)  → proceed (when waiting to start episode)\n"
    )

    # Drain buffered stdin
    while select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()

    class _StdinListener:
        def __init__(self):
            threading.Thread(target=self._run, daemon=True).start()

        def stop(self):
            pass

        def _run(self):
            try:
                for line in sys.stdin:
                    cmd = line.strip().lower()
                    if cmd in ("q", "quit"):
                        events["stop_recording"] = True
                        events["exit_early"] = True
                        proceed_event.set()
                        print("  → Stopping …")
                    elif events.get("waiting_for_proceed"):
                        proceed_event.set()
                    elif cmd in ("", "s"):
                        events["exit_early"] = True
                        print("  → Saving episode …")
                    elif cmd == "r":
                        events["rerecord_episode"] = True
                        events["exit_early"] = True
                        print("  → Re-recording …")
            except EOFError:
                pass

    return _StdinListener(), events, proceed_event


# ---------------------------------------------------------------------------
# record_loop  (unchanged lerobot pattern; YOLO injection is in the wrapper)
# ---------------------------------------------------------------------------

@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    dataset: YoloDatasetWrapper | LeRobotDataset | None = None,
    teleop: Teleoperator | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor=None,
    postprocessor=None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"dataset.fps ({dataset.fps}) != requested fps ({fps}).")

    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)

        if dataset is not None:
            # dataset.features on YoloDatasetWrapper excludes YOLO_FEATURE_KEY so
            # build_dataset_frame won't try to look it up in the obs dict.
            # dataset.add_frame (on the wrapper) injects YOLO before storing.
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        if policy is not None and preprocessor is not None and postprocessor is not None:
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )
            act_processed_policy: RobotAction = make_robot_action(action_values, dataset.features)
        elif policy is None and teleop is not None:
            act = teleop.get_action()
            act_processed_teleop = teleop_action_processor((act, obs))
        else:
            logger.info("No policy or teleop — skipping action.")
            continue

        if policy is not None and act_processed_policy is not None:
            action_values = act_processed_policy
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))
        else:
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        robot.send_action(robot_action_to_send)

        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)  # wrapper injects YOLO_FEATURE_KEY here

        if display_data:
            log_rerun_data(
                observation=obs_processed,
                action=action_values,
                compress_images=display_compressed_images,
            )

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(max(1 / fps - dt_s, 0.0))
        timestamp = time.perf_counter() - start_episode_t


# ---------------------------------------------------------------------------
# Main record function
# ---------------------------------------------------------------------------

@parser.wrap()
def record(cfg: RecordWithYoloConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )
    if cfg.display_data:
        init_rerun(session_name="recording_yolo", ip=cfg.display_ip, port=cfg.display_port)

    # Pre-load YOLO model before connecting any hardware.
    # On Jetson, model loading triggers CUDA/NvMap memory allocation that can
    # cause NvMap OOM errors (error 12).  Those errors stall the RealSense USB
    # pipeline if the camera is already open, causing async_read() timeouts
    # during recording.  Loading the model first lets memory settle before the
    # camera pipeline starts.
    from yolo_camera import get_yolo_model
    logger.info("Pre-loading YOLO model …")
    _yolo_model, _yolo_err = get_yolo_model()
    if _yolo_err:
        logger.warning(f"YOLO model pre-load failed (will retry at first scan): {_yolo_err}")
    else:
        logger.info("YOLO model ready.")

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Build robot features, then register YOLO in the underlying dataset features.
    # The wrapper will hide YOLO from build_dataset_frame during recording but the
    # real dataset stores and indexes it — lerobot_train sees it on load.
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )
    dataset_features[YOLO_FEATURE_KEY] = YOLO_FEATURE
    logger.info(f"Dataset features: {list(dataset_features.keys())}")

    raw_dataset: LeRobotDataset | None = None
    dataset: YoloDatasetWrapper | None = None
    listener = None

    try:
        if cfg.resume:
            raw_dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
            )
            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                raw_dataset.start_image_writer(
                    num_processes=cfg.dataset.num_image_writer_processes,
                    num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                )
            sanity_check_dataset_robot_compatibility(raw_dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
            raw_dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=(
                    cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras)
                ),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
            )

        dataset = YoloDatasetWrapper(raw_dataset)

        # Load policy if provided
        policy = None
        preprocessor = postprocessor = None
        if cfg.policy is not None:
            policy = make_policy(cfg.policy, ds_meta=raw_dataset.meta)
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=cfg.policy,
                pretrained_path=cfg.policy.pretrained_path,
                dataset_stats=rename_stats(raw_dataset.meta.stats, cfg.dataset.rename_map),
                preprocessor_overrides={
                    "device_processor": {"device": cfg.policy.device},
                    "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
                },
            )

        robot.connect()
        if teleop is not None:
            teleop.connect()

        listener, events, proceed_event = _init_keyboard_listener()

        with VideoEncodingManager(raw_dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:

                # ── Background teleop: leader drives follower while arm is being positioned ──
                teleop_stop = threading.Event()
                if teleop is not None:
                    bg_teleop = threading.Thread(
                        target=_run_teleop_background,
                        args=(robot, teleop, teleop_action_processor,
                              robot_action_processor, teleop_stop, cfg.dataset.fps),
                        daemon=True,
                        name="bg-teleop",
                    )
                    bg_teleop.start()

                print(
                    f"\n  Use leader arm to position follower, then press Enter to start "
                    f"episode {recorded_episodes + 1}/{cfg.dataset.num_episodes} "
                    f"(s=save early, r=re-record, q=quit) …"
                )
                proceed_event.clear()
                events["waiting_for_proceed"] = True
                proceed_event.wait()
                events["waiting_for_proceed"] = False

                # Stop background teleop before the recording loop takes over
                teleop_stop.set()
                if teleop is not None:
                    bg_teleop.join(timeout=0.5)

                if events["stop_recording"]:
                    break

                # ── Pre-episode: YOLO scan ───────────────────────────────────
                # YOLO inference on Jetson causes NvMap GPU memory pressure that
                # stalls the RealSense USB pipeline's _read_loop thread, making
                # async_read() time out even though the thread is alive.
                #
                # Fix: keep all cameras warm during inference by continuously
                # draining frames in a background thread.  This prevents the
                # pipeline's internal frame queue from backing up or stalling.
                log_say(f"Scanning for {cfg.yolo_target}s…", cfg.play_sounds)

                _cam_warm_stop = threading.Event()

                def _camera_keepalive():
                    while not _cam_warm_stop.is_set():
                        for _cam in robot.cameras.values():
                            try:
                                _cam.async_read()
                            except Exception:
                                pass
                        time.sleep(0.04)  # drain at ~25 Hz

                _cam_warm_thread = threading.Thread(
                    target=_camera_keepalive, daemon=True, name="cam-keepalive"
                )
                _cam_warm_thread.start()

                try:
                    yolo_frame_rgb = None
                    if "realsense" in robot.cameras:
                        try:
                            yolo_frame_rgb = robot.cameras["realsense"].async_read()
                        except Exception as e:
                            logger.warning(f"Could not grab realsense frame for YOLO: {e}")
                    result = observe_screws(
                        target_object=cfg.yolo_target,
                        confidence_threshold=cfg.yolo_confidence,
                        frame_rgb=yolo_frame_rgb,
                    )
                finally:
                    _cam_warm_stop.set()
                    _cam_warm_thread.join(timeout=1.0)

                dataset.set_episode_yolo(result["screw_xyz"])
                logger.info(
                    f"Episode {raw_dataset.num_episodes}: detected {result['count']} "
                    f"{cfg.yolo_target}(s). depth_status={result['depth_status']}  "
                    f"xyz={result['screw_xyz']}"
                )
                if result["error"]:
                    logger.warning(f"YOLO scan warning: {result['error']}")

                log_say(f"Recording episode {raw_dataset.num_episodes}", cfg.play_sounds)
                print(f"\n  Recording episode {recorded_episodes + 1}/{cfg.dataset.num_episodes} …")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=display_compressed_images,
                )

                if events["rerecord_episode"]:
                    log_say("Re-recording episode", cfg.play_sounds)
                    print("  Re-recording …")
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    raw_dataset.clear_episode_buffer()
                    continue

                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)
                    print("  Reset — move arm back to start position …")
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-recording episode", cfg.play_sounds)
                    print("  Re-recording …")
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    raw_dataset.clear_episode_buffer()
                    continue

                raw_dataset.save_episode()
                recorded_episodes += 1
                print(f"  Episode {recorded_episodes} saved.")

    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)
        if raw_dataset is not None:
            raw_dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()
        if listener:
            listener.stop()
        if cfg.dataset.push_to_hub and raw_dataset is not None:
            raw_dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
        log_say("Exiting", cfg.play_sounds)

    return raw_dataset


def main():
    record()


if __name__ == "__main__":
    main()
