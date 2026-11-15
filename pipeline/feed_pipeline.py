"""
pipeline/feed_pipeline.py - Individual Camera Stream Processing Pipeline
AIEcoLightswitcher Showcase Edition (Milestone 3 Implementation)

Encapsulates the end-to-end processing pipeline for a single camera stream:
- Ingestion management via dedicated StreamWorker (non-blocking ring buffer)
- Spatial polygon ROI filtering via SpatialROIFilter
- Temporal presence debounce hysteresis smoothing via TemporalPresenceSmoother
- Power-saving duty cycle management via DutyCycleManager
- Annotated visual rendering with detection bounding boxes and status HUD
- Channel-isolated telemetry and effective pin output calculation
"""

import copy
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from core.config import FeedConfig
from core.duty_cycle import DutyCycleManager
from core.roi_filter import SpatialROIFilter
from core.smoother import TemporalPresenceSmoother
from ingestion.base_stream import FeedStatus
from ingestion.stream_worker import StreamWorker

logger = logging.getLogger("pipeline.feed_pipeline")

COCO_PERSON_CLASS_ID = 0
DEFAULT_CLASS_NAMES = {0: "Person"}


class FeedPipeline:
    """
    Encapsulates the end-to-end processing pipeline for an individual video feed.
    Operates in full isolation so slow or failing streams never degrade siblings.
    """

    def __init__(self, config: FeedConfig):
        # Shallow copy config to prevent pickling issues with source objects
        self.config = copy.copy(config)
        self.feed_id = config.feed_id
        self.name = config.name
        self.pin = config.pin

        self._lock = threading.RLock()
        self.worker = StreamWorker(
            source=config.source,
            feed_id=config.feed_id,
            name=config.name
        )

        poly = list(config.roi_polygon) if config.roi_polygon else None
        self.roi_filter = SpatialROIFilter(
            normalized_vertices=poly,
            enabled=config.roi_enabled
        )
        self.smoother = TemporalPresenceSmoother()
        self.duty_cycle = DutyCycleManager(
            eval_duration=config.eval_duration,
            sleep_duration=config.sleep_duration,
            enabled=True
        )

        # Telemetry & state caches
        self._latest_annotated_frame: Optional[np.ndarray] = None
        self._latest_telemetry: Dict[str, Any] = {}
        self._last_raw_occupants: int = 0
        self._last_roi_occupants: int = 0
        self._last_detections: Tuple[List[Dict[str, Any]], List[Dict[str, Any]]] = ([], [])
        self._last_latency_ms: float = 0.0
        self._max_cycle_occupants: int = 0
        self._fps_ema: float = 0.0
        self.effective_light: bool = False

    def start(self) -> None:
        """Starts the underlying StreamWorker thread if enabled."""
        with self._lock:
            if self.config.enabled:
                self.worker.start()

    def stop(self) -> None:
        """Stops the underlying StreamWorker thread."""
        with self._lock:
            self.worker.stop()

    @property
    def is_alive(self) -> bool:
        """Checks if the underlying StreamWorker is running."""
        return self.worker.is_alive

    @property
    def status(self) -> FeedStatus:
        """Returns the current operational status of the feed."""
        with self._lock:
            if not self.config.enabled:
                return FeedStatus.DISABLED
            return self.worker.status

    def update_config(self, **kwargs) -> None:
        """Updates feed configuration parameters dynamically."""
        with self._lock:
            for key, val in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, val)
                if key == "pin":
                    self.pin = val

            # Sync dependent component properties
            if "roi_enabled" in kwargs or "roi_polygon" in kwargs:
                poly = list(self.config.roi_polygon) if self.config.roi_polygon else None
                self.roi_filter.update_polygon(
                    normalized_vertices=poly,
                    enabled=self.config.roi_enabled
                )

            if "eval_duration" in kwargs or "sleep_duration" in kwargs:
                self.duty_cycle.update_durations(
                    eval_duration=self.config.eval_duration,
                    sleep_duration=self.config.sleep_duration
                )

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Returns the latest raw frame and telemetry from the worker."""
        return self.worker.get_latest_frame()

    def process_step(
        self,
        model: Any = None,
        device: str = "cpu"
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any], bool]:
        """
        Executes a single processing step:
        1. Retrieves frame from StreamWorker
        2. Evaluates duty cycle (evaluating vs sleeping)
        3. If evaluating: executes YOLOv8 inference and spatial ROI filtering
        4. Applies temporal presence hysteresis smoothing
        5. Computes effective pin state (incorporating override_mode)
        6. Renders visual annotations, ROI overlays, and telemetry HUD

        Returns:
            (annotated_frame, telemetry_dict, effective_light_state)
        """
        with self._lock:
            cfg = copy.copy(self.config)

        if not cfg.enabled:
            return None, self.get_telemetry(), False

        raw_frame, w_telem = self.worker.get_latest_frame()
        raw_status = w_telem.get("status", FeedStatus.OFFLINE)
        if raw_frame is None or raw_status != FeedStatus.ONLINE:
            telemetry = self.get_telemetry()
            return None, telemetry, self.effective_light

        now = time.time()
        h, w = raw_frame.shape[:2]

        # 1. Duty Cycle State Check
        cycle_state, cycle_countdown, _ = self.duty_cycle.update(now)

        detections: List[Dict[str, Any]] = []
        inside_dets: List[Dict[str, Any]] = []
        outside_dets: List[Dict[str, Any]] = []
        latency_ms = 0.0

        if cycle_state == "EVALUATING" or not self.duty_cycle.enabled:
            t0 = time.perf_counter()
            if model is not None:
                try:
                    results = model(
                        raw_frame,
                        device=device,
                        verbose=False,
                        conf=cfg.conf_threshold,
                        classes=[COCO_PERSON_CLASS_ID]
                    )
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    boxes = results[0].boxes
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        label = f"{DEFAULT_CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.2f}"
                        detections.append({
                            "xyxy": xyxy,
                            "conf": conf,
                            "cls_id": cls_id,
                            "label": label
                        })
                    del results
                except Exception as exc:
                    logger.debug(f"[feed_pipeline:{self.feed_id}] Inference error: {exc}")
                    latency_ms = 0.0

            raw_occupants = len(detections)
            inside_dets, outside_dets = self.roi_filter.filter_detections(detections, w, h)
            roi_occupants = len(inside_dets) if cfg.roi_enabled else raw_occupants

            self._last_raw_occupants = raw_occupants
            self._last_roi_occupants = roi_occupants
            self._last_detections = (inside_dets, outside_dets)
            self._last_latency_ms = latency_ms

            if roi_occupants > self._max_cycle_occupants:
                self._max_cycle_occupants = roi_occupants
        else:
            raw_occupants = self._last_raw_occupants
            roi_occupants = self._last_roi_occupants
            latency_ms = self._last_latency_ms
            inside_dets, outside_dets = self._last_detections

        # 2. Temporal Presence Smoothing
        raw_presence = (roi_occupants > 0)
        smoothed_presence, _ = self.smoother.update(raw_presence, now)

        # 3. Calculate Effective Light Output
        override = (cfg.override_mode or "AUTO").upper().strip()
        if override == "FORCE_ON":
            effective = True
        elif override == "FORCE_OFF":
            effective = False
        else:  # AUTO
            effective = smoothed_presence if cfg.enabled else False

        self.effective_light = effective

        # 4. Visual Rendering
        annotated_frame = raw_frame.copy()

        if cfg.roi_enabled:
            annotated_frame = self.roi_filter.draw_roi_overlay(
                annotated_frame, is_occupied=(roi_occupants > 0)
            )

        for det in inside_dets:
            xyxy = det["xyxy"]
            color = (0, 255, 0) if det.get("cls_id") == 0 else (255, 200, 0)
            cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            cv2.putText(
                annotated_frame, det.get("label", "Person"),
                (xyxy[0], max(20, xyxy[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            anchor = det.get("anchor")
            if anchor:
                cv2.circle(annotated_frame, anchor, 4, (0, 0, 255), -1)

        for det in outside_dets:
            xyxy = det["xyxy"]
            color = (130, 130, 130)
            cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 1)
            cv2.putText(
                annotated_frame, f"{det.get('label', 'Person')} (Outside ROI)",
                (xyxy[0], max(20, xyxy[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
            )

        if cycle_state == "SLEEPING":
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (25, 25, 25), -1)
            cv2.addWeighted(overlay, 0.35, annotated_frame, 0.65, 0, annotated_frame)
            cv2.putText(
                annotated_frame,
                f"POWER SAVING IDLE (Resumes in {cycle_countdown:.1f}s)",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2
            )

        # 5. FPS update
        worker_fps = w_telem.get("fps", 0.0)
        self._fps_ema = (0.85 * self._fps_ema + 0.15 * worker_fps) if self._fps_ema > 0 else worker_fps

        with self._lock:
            self._latest_annotated_frame = annotated_frame
            self._latest_telemetry = {
                "feed_id": self.feed_id,
                "name": self.name,
                "pin": self.pin,
                "enabled": cfg.enabled,
                "override_mode": override,
                "status": FeedStatus.ONLINE,
                "fps": round(self._fps_ema, 1),
                "latency_ms": round(latency_ms, 1),
                "frame_id": w_telem.get("frame_id", 0),
                "raw_occupants": raw_occupants,
                "roi_occupants": roi_occupants,
                "smoothed_presence": smoothed_presence,
                "effective_light": effective,
                "cycle_state": cycle_state,
                "cycle_countdown": round(cycle_countdown, 1),
                "reconnect_attempts": w_telem.get("reconnect_attempts", 0),
                "reconnect_countdown": w_telem.get("reconnect_countdown", 0.0)
            }

        return annotated_frame, dict(self._latest_telemetry), effective

    def get_telemetry(self) -> Dict[str, Any]:
        """Builds a complete snapshot of current feed telemetry."""
        with self._lock:
            cfg = self.config
            worker = self.worker
            _, w_telem = worker.get_latest_frame() if worker else (None, {})
            mode = (cfg.override_mode or "AUTO").upper().strip()
            raw_status = w_telem.get("status", FeedStatus.OFFLINE)
            status = FeedStatus.DISABLED if not cfg.enabled else raw_status

            return {
                "feed_id": self.feed_id,
                "name": cfg.name,
                "pin": cfg.pin,
                "enabled": cfg.enabled,
                "override_mode": mode,
                "status": status,
                "fps": w_telem.get("fps", round(self._fps_ema, 1)),
                "latency_ms": w_telem.get("latency_ms", round(self._last_latency_ms, 1)),
                "frame_id": w_telem.get("frame_id", 0),
                "raw_occupants": self._last_raw_occupants,
                "roi_occupants": self._last_roi_occupants,
                "smoothed_presence": self.smoother.current_state,
                "effective_light": self.effective_light,
                "cycle_state": self.duty_cycle.current_state,
                "reconnect_attempts": w_telem.get("reconnect_attempts", 0),
                "reconnect_countdown": w_telem.get("reconnect_countdown", 0.0)
            }
