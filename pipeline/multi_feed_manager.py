"""
pipeline/multi_feed_manager.py - Central Multi-Camera Stream & Actuation Coordinator
AIEcoLightswitcher Showcase Edition (Milestone 3 Implementation)

Central coordinator responsible for:
- Managing dynamic feed lifecycle: add_feed, remove_feed, enable_feed, disable_feed
- Centralized YOLOv8 inference dispatch across concurrent video feeds
- Dynamic per-feed Arduino GPIO pin routing (Pins 2–19)
- Per-feed manual overrides (AUTO, FORCE_ON, FORCE_OFF)
- Fault isolation: disconnected or lagging feeds never block or crash sibling feeds
- Thread-safe telemetry snapshot generation
"""

import copy
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from core.config import FeedConfig
from core.device import detect_best_device
from hardware.base_controller import BaseMultiPinHardwareController
from hardware.mock_controller import MockMultiPinHardwareController
from hardware.serial_controller import MultiPinHardwareController
from ingestion.base_stream import FeedStatus
from ingestion.stream_worker import StreamWorker
from pipeline.feed_pipeline import FeedPipeline

logger = logging.getLogger("pipeline.multi_feed_manager")

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class MultiFeedManager:
    """
    Central Multi-Camera Pipeline Coordinator for AIEcoLightswitcher.
    Orchestrates ingestion, detection, and hardware pin actuation across N video feeds.
    """

    def __init__(
        self,
        hardware: Optional[BaseMultiPinHardwareController] = None,
        model_path: str = "yolov8n.pt",
        device: Optional[str] = None
    ):
        self.hardware = hardware if hardware is not None else MockMultiPinHardwareController()
        self.model_path = model_path
        dev = device or detect_best_device()
        if isinstance(dev, (list, tuple)):
            self.device = str(dev[0])
            self.device_info = str(dev[1]) if len(dev) > 1 else str(dev[0])
        else:
            self.device = str(dev)
            self.device_info = str(dev)

        self._lock = threading.RLock()
        self._feeds: Dict[str, FeedConfig] = {}
        self._workers: Dict[str, StreamWorker] = {}
        self._pipelines: Dict[str, FeedPipeline] = {}
        self._overrides: Dict[str, str] = {}

        self._running = False
        self._eval_thread: Optional[threading.Thread] = None

        # Model instance (lazy loaded or initialized gracefully)
        self.model: Any = None
        self._init_model()

    def _init_model(self) -> None:
        """Initializes YOLOv8 model instance if available."""
        if YOLO is not None and os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
                logger.info(f"[multi_feed_manager] YOLOv8 model loaded from {self.model_path} on {self.device}")
            except Exception as exc:
                logger.warning(f"[multi_feed_manager] YOLO initialization deferred: {exc}")
                self.model = None
        else:
            self.model = None

    def add_feed(self, config: FeedConfig) -> bool:
        """
        Dynamically registers and initializes a new camera feed stream.
        Returns False if feed_id already exists.
        """
        with self._lock:
            if config.feed_id in self._feeds:
                logger.warning(f"[multi_feed_manager] Feed '{config.feed_id}' already registered.")
                return False

            cfg = copy.copy(config)
            override = (cfg.override_mode or "AUTO").upper().strip()
            self._feeds[cfg.feed_id] = cfg
            self._overrides[cfg.feed_id] = override

            pipeline = FeedPipeline(cfg)
            self._pipelines[cfg.feed_id] = pipeline
            self._workers[cfg.feed_id] = pipeline.worker

            if self._running and cfg.enabled:
                pipeline.start()

            logger.info(f"[multi_feed_manager] Added feed '{cfg.feed_id}' ({cfg.name}) mapped to Pin {cfg.pin}.")
            return True

    def remove_feed(self, feed_id: str) -> bool:
        """
        Dynamically tears down and unregisters a camera feed.
        Stops the ingestion worker and ensures the associated hardware pin is turned OFF.
        """
        with self._lock:
            if feed_id not in self._feeds:
                logger.warning(f"[multi_feed_manager] Cannot remove unknown feed '{feed_id}'.")
                return False

            cfg = self._feeds.pop(feed_id)
            self._overrides.pop(feed_id, None)
            self._workers.pop(feed_id, None)
            pipeline = self._pipelines.pop(feed_id, None)

            if pipeline:
                pipeline.stop()

            if self.hardware:
                try:
                    self.hardware.set_pin(cfg.pin, False)
                except Exception as exc:
                    logger.error(f"[multi_feed_manager] Error resetting pin {cfg.pin} on remove: {exc}")

            logger.info(f"[multi_feed_manager] Removed feed '{feed_id}' and reset Pin {cfg.pin}.")
            return True

    def enable_feed(self, feed_id: str) -> bool:
        """
        Enables an individual camera feed.
        Starts stream worker if the manager is currently running.
        """
        with self._lock:
            if feed_id not in self._feeds:
                return False

            cfg = self._feeds[feed_id]
            cfg.enabled = True
            pipeline = self._pipelines.get(feed_id)
            if pipeline:
                pipeline.update_config(enabled=True)
                if self._running:
                    pipeline.start()

            logger.info(f"[multi_feed_manager] Enabled feed '{feed_id}'.")
            return True

    def disable_feed(self, feed_id: str) -> bool:
        """
        Disables an individual camera feed.
        Stops stream worker and immediately drives the assigned hardware pin LOW.
        """
        with self._lock:
            if feed_id not in self._feeds:
                return False

            cfg = self._feeds[feed_id]
            cfg.enabled = False
            pipeline = self._pipelines.get(feed_id)
            if pipeline:
                pipeline.update_config(enabled=False)
                pipeline.stop()

            if self.hardware:
                try:
                    self.hardware.set_pin(cfg.pin, False)
                except Exception as exc:
                    logger.error(f"[multi_feed_manager] Error resetting pin {cfg.pin} on disable: {exc}")

            logger.info(f"[multi_feed_manager] Disabled feed '{feed_id}' and reset Pin {cfg.pin} to 0.")
            return True

    def set_feed_override(self, feed_id: str, mode: str) -> bool:
        """
        Applies a manual override ('AUTO', 'FORCE_ON', 'FORCE_OFF') to a specific feed.
        Actuates the hardware pin immediately if in FORCE_ON or FORCE_OFF mode.
        """
        with self._lock:
            if feed_id not in self._feeds:
                return False

            m = mode.upper().strip()
            if m not in ("AUTO", "FORCE_ON", "FORCE_OFF"):
                logger.warning(f"[multi_feed_manager] Invalid override mode '{mode}'.")
                return False

            self._overrides[feed_id] = m
            cfg = self._feeds[feed_id]
            cfg.override_mode = m

            pipeline = self._pipelines.get(feed_id)
            if pipeline:
                pipeline.update_config(override_mode=m)

            if self.hardware:
                if m == "FORCE_ON":
                    self.hardware.set_pin(cfg.pin, True)
                elif m == "FORCE_OFF":
                    self.hardware.set_pin(cfg.pin, False)

            logger.info(f"[multi_feed_manager] Feed '{feed_id}' override set to {m}.")
            return True

    def update_feed_config(self, feed_id: str, **kwargs) -> bool:
        """Dynamically updates configuration parameters for an existing feed."""
        with self._lock:
            if feed_id not in self._feeds:
                return False

            cfg = self._feeds[feed_id]
            for k, v in kwargs.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
                if k == "override_mode":
                    self._overrides[feed_id] = str(v).upper().strip()

            pipeline = self._pipelines.get(feed_id)
            if pipeline:
                pipeline.update_config(**kwargs)

            return True

    def get_feed_telemetry(self, feed_id: str) -> Dict[str, Any]:
        """
        Returns full telemetry snapshot for a specific feed.
        Thread-safe and conforming to the E2E verification test contract.
        """
        with self._lock:
            if feed_id not in self._feeds:
                return {}

            cfg = self._feeds[feed_id]
            pipeline = self._pipelines.get(feed_id)
            worker = self._workers.get(feed_id)

            if pipeline:
                t = pipeline.get_telemetry()
            else:
                _, w_telem = worker.get_latest_frame() if worker else (None, {})
                raw_status = w_telem.get("status", FeedStatus.OFFLINE)
                status = FeedStatus.DISABLED if not cfg.enabled else raw_status
                t = {
                    "feed_id": feed_id,
                    "name": cfg.name,
                    "pin": cfg.pin,
                    "enabled": cfg.enabled,
                    "override_mode": self._overrides.get(feed_id, "AUTO"),
                    "status": status,
                    "fps": w_telem.get("fps", 0.0),
                    "latency_ms": w_telem.get("latency_ms", 0.0),
                    "frame_id": w_telem.get("frame_id", 0)
                }

            # Inject actual hardware pin state
            pin_state = False
            if self.hardware:
                pin_state = self.hardware.get_pin(cfg.pin)

            t["pin_state"] = pin_state
            t["override_mode"] = self._overrides.get(feed_id, cfg.override_mode or "AUTO")
            return t

    def get_all_feeds_telemetry(self) -> Dict[str, Dict[str, Any]]:
        """Returns consolidated telemetry snapshots across all registered feeds."""
        with self._lock:
            return {fid: self.get_feed_telemetry(fid) for fid in list(self._feeds.keys())}

    def get_feed_frame(self, feed_id: str) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Retrieves raw or annotated frame from a specific feed."""
        with self._lock:
            pipeline = self._pipelines.get(feed_id)
            if pipeline:
                return pipeline.get_latest_frame()
            worker = self._workers.get(feed_id)
            if worker:
                return worker.get_latest_frame()
            return None, {}

    def get_feed_annotated_frame(self, feed_id: str) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Retrieves the latest annotated visual frame with detection bounding boxes."""
        with self._lock:
            pipeline = self._pipelines.get(feed_id)
            if pipeline:
                with pipeline._lock:
                    if pipeline._latest_annotated_frame is not None:
                        return pipeline._latest_annotated_frame.copy(), dict(pipeline._latest_telemetry)
                return pipeline.get_latest_frame()
            return None, {}

    def start(self) -> None:
        """
        Starts the multi-feed pipeline manager:
        Spawns ingestion workers and launches background evaluation thread.
        """
        with self._lock:
            if self._running:
                return
            self._running = True

            for fid, pipeline in self._pipelines.items():
                if self._feeds[fid].enabled:
                    pipeline.start()

            self._eval_thread = threading.Thread(
                target=self._evaluation_loop,
                name="MultiFeedManager-EvalThread",
                daemon=True
            )
            self._eval_thread.start()
            logger.info("[multi_feed_manager] Started MultiFeedManager evaluation thread.")

    def stop(self) -> None:
        """
        Gracefully terminates all feed streams and turns off all hardware pins.
        """
        with self._lock:
            if not self._running:
                return
            self._running = False

            for pipeline in self._pipelines.values():
                pipeline.stop()

            if self.hardware:
                try:
                    self.hardware.set_all(False)
                except Exception as exc:
                    logger.error(f"[multi_feed_manager] Error turning off all pins on stop: {exc}")

            logger.info("[multi_feed_manager] Stopped MultiFeedManager.")

        if self._eval_thread and self._eval_thread.is_alive():
            self._eval_thread.join(timeout=0.5)

    def _evaluation_loop(self) -> None:
        """
        Background evaluation loop.
        Sequentially dispatches inference across active feeds, calculates effective light,
        and routes hardware GPIO pin actuations.
        """
        while self._running:
            with self._lock:
                active_fids = [fid for fid, cfg in self._feeds.items() if cfg.enabled]

            for fid in active_fids:
                if not self._running:
                    break

                with self._lock:
                    pipeline = self._pipelines.get(fid)
                    cfg = self._feeds.get(fid)

                if pipeline is None or cfg is None:
                    continue

                try:
                    # Execute single processing step
                    _, telem, effective_light = pipeline.process_step(
                        model=self.model,
                        device=self.device
                    )

                    # Route result to hardware pin
                    override = self._overrides.get(fid, "AUTO")
                    if override == "FORCE_ON":
                        target_state = True
                    elif override == "FORCE_OFF":
                        target_state = False
                    else:
                        target_state = effective_light

                    if self.hardware:
                        self.hardware.set_pin(cfg.pin, target_state)

                except Exception as exc:
                    logger.error(f"[multi_feed_manager] Error evaluating feed '{fid}': {exc}")

            time.sleep(0.01)
