"""
main.py - AIEcoLightswitcher AI Vision & Smart Energy Command Dashboard
Showcase & Interview Edition (Milestone 2 Implementation)

Modernized Bento-Grid Edge AI Command Dashboard featuring:
- Deep Charcoal / Dark Slate Bento-Grid surface hierarchy with 1px borders
- Top Header Ribbon (54px) with Brand, RTX 4050/CPU badge, live FPS, latency,
  animated pulsing duty-cycle badge, and serial link status
- Video Canvas HUD Card with tactical bounding boxes, dynamic polygon ROI,
  semi-transparent glassmorphic corner telemetry chips, and single duty-cycle sleep overlay
- 4 Bento Stat Tiles: Target Confidence, Smoothed Occupancy, Duty Cycle, Arduino Relay
- 4-Tab Control Station (390px):
  * Tab 1 (Live Controls): Presets, runtime sliders for Confidence, Eval & Sleep durations
  * Tab 2 (Spatial ROI): Master toggle, Foot vs Centroid anchor mode, Presets & Custom entry
  * Tab 3 (Hardware & Comms): COM auto-discovery + Baud selector, Camera stream ingestion, Overrides
  * Tab 4 (Event Log): 100-entry ring buffer, edge-triggered state logging, monospace terminal
- Decoupled multi-threaded pipeline: non-blocking 20ms UI loop with clean shutdown
"""

import copy
from collections import deque
from datetime import datetime
import os
import re
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctk

# Ensure parent directory is in Python path to import eco_engine
_PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from eco_engine import (
    AdvancedDetectionEngine,
    EcoVisionSystem,
    EngineConfig,
    HardwareController,
    MockHardwareController,
    MockVideoSource,
    detect_best_device,
    get_inference_device
)


# ============================================================================
# Design Tokens & Palette Hierarchy (Bento Theme)
# ============================================================================

BENTO_THEME = {
    # Surfaces
    "bg_root": "#0B0F17",              # Deep charcoal root canvas
    "card_surface": "#131822",         # Elevated dark slate card surface
    "card_surface_alt": "#161B22",     # Secondary card / chip tile surface
    "card_surface_subtle": "#0D1117",  # Recessed canvas / terminal background
    
    # Borders
    "border_subtle": "#21262D",        # Subtle 1px card perimeter
    "border_accent": "#30363D",        # Focused / active border
    "border_cyan": "#00F0FF",          # Electric Cyan accent outline
    
    # Accents & Status Indicators
    "accent_cyan": "#00F0FF",          # Electric Cyan (Active vision metrics)
    "accent_indigo": "#6366F1",        # Indigo (Branding & category tags)
    "status_active_mint": "#10B981",   # Vivid Mint (Occupancy active, relay ON)
    "status_sleep_amber": "#F59E0B",   # Amber (Duty-cycle power-saving sleep)
    "status_tamper_crimson": "#EF4444",# Crimson (Disconnect, alarm, relay OFF)
    
    # Typography
    "text_primary": "#F0F6FC",         # Crisp foreground text
    "text_secondary": "#8B949E",       # Muted slate secondary labels
    "text_muted": "#6E7681",           # Inactive subtext
}

# OpenCV BGR Color Equivalents for HUD Rendering
BGR_MINT = (129, 185, 16)      # #10B981 in BGR
BGR_CYAN = (255, 240, 0)       # #00F0FF in BGR
BGR_AMBER = (11, 158, 245)     # #F59E0B in BGR
BGR_CRIMSON = (68, 68, 239)    # #EF4444 in BGR
BGR_SLATE = (139, 116, 100)    # #64748B in BGR
BGR_CARD_BG = (34, 24, 19)     # #131822 in BGR
BGR_BORDER = (45, 38, 33)      # #21262D in BGR


# ============================================================================
# Event Logger (Thread-Safe 100-Entry Ring Buffer for Tab 4)
# ============================================================================

class EventLogger:
    """
    Fixed-capacity ring buffer capturing timestamped telemetry & system events:
    - Occupant entries and exits
    - Duty cycle state transitions
    - Serial TX commands ('1' and '0')
    - Runtime configuration and ROI adjustments
    """
    def __init__(self, maxlen: int = 100):
        self.buffer: deque = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.listeners: List[Callable[[Dict[str, Any]], None]] = []

    def log(self, category: str, message: str, level: str = "INFO") -> Dict[str, Any]:
        """Appends a new timestamped event and dispatches it to listeners."""
        now = time.time()
        time_str = datetime.fromtimestamp(now).strftime("%H:%M:%S.%f")[:-3]
        entry = {
            "time": time_str,
            "timestamp": now,
            "category": category.upper().strip(),  # VISION, CYCLE, SERIAL, CONFIG, SYSTEM
            "message": message.strip(),
            "level": level.upper().strip()
        }
        with self.lock:
            self.buffer.append(entry)

        for listener in self.listeners:
            try:
                listener(entry)
            except Exception:
                pass
        return entry

    def add_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Registers a callback function to receive newly logged events."""
        self.listeners.append(callback)

    def get_entries(self) -> List[Dict[str, Any]]:
        """Returns snapshot of current event history."""
        with self.lock:
            return list(self.buffer)

    def clear(self) -> None:
        """Clears all entries from ring buffer."""
        with self.lock:
            self.buffer.clear()


# ============================================================================
# HUD Renderer (Glassmorphic OpenCV Overlays & Telemetry Chips)
# ============================================================================

class HudRenderer:
    """
    Renders sleek computer vision HUD elements onto OpenCV frames:
    - Target bounding boxes with 4 corner brackets and translucent confidence pills
    - Dynamic polygon ROI zone with corner brackets and translucent mask
    - Semi-transparent glassmorphic corner telemetry chips (FPS, latency, occupants, relay)
    - Single duty-cycle sleep overlay banner (no double darkening)
    """

    @staticmethod
    def draw_chip(
        img: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        bg_color: Tuple[int, int, int] = BGR_CARD_BG,
        text_color: Tuple[int, int, int] = (255, 255, 255),
        border_color: Tuple[int, int, int] = BGR_BORDER,
        alpha: float = 0.82,
        pad: Tuple[int, int] = (8, 4),
        font_scale: float = 0.38,
        dot_color: Optional[Tuple[int, int, int]] = None
    ) -> None:
        """Draws a semi-transparent glassmorphic chip with a subtle 1px border."""
        x, y = pos
        font = cv2.FONT_HERSHEY_SIMPLEX
        thick = 1
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thick)
        px, py = pad
        dot_offset = 12 if dot_color else 0
        x2 = x + tw + px * 2 + dot_offset
        y2 = y + th + py * 2
        h, w = img.shape[:2]
        if x2 > w or y2 > h or x < 0 or y < 0:
            return

        sub = img[y:y2, x:x2]
        bg = np.full_like(sub, bg_color)
        cv2.addWeighted(bg, alpha, sub, 1.0 - alpha, 0, sub)
        cv2.rectangle(img, (x, y), (x2, y2), border_color, 1, cv2.LINE_AA)

        if dot_color:
            cv2.circle(img, (x + px + 4, y + py + th // 2), 4, dot_color, -1, cv2.LINE_AA)

        text_x = x + px + dot_offset
        text_y = y + th + py - 2
        cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thick, cv2.LINE_AA)

    @classmethod
    def render_hud_telemetry_chips(cls, frame: np.ndarray, telemetry: Dict[str, Any]) -> None:
        """Anchors 4 glassmorphic telemetry chips to the frame corners."""
        h, w = frame.shape[:2]
        if h < 100 or w < 200:
            return

        # 1. Top-Left Chip: Pipeline Mode & Compute Node
        is_eval = telemetry.get("cycle_state") == "EVALUATING"
        status_dot = BGR_MINT if is_eval else BGR_AMBER
        status_text = "LIVE" if is_eval else "SLEEP"
        device_str = str(telemetry.get("device_info", telemetry.get("device", "CPU")))
        if "cuda" in device_str.lower():
            dev_short = "CUDA:0 (RTX 4050)"
        else:
            dev_short = "CPU (HOST)"

        cls.draw_chip(
            frame, f"{status_text} | {dev_short}",
            pos=(12, 12),
            border_color=BGR_CYAN if is_eval else BGR_AMBER,
            text_color=(255, 255, 255),
            dot_color=status_dot
        )

        # 2. Top-Right Chip: FPS & Latency
        fps = float(telemetry.get("fps", 0.0))
        lat = float(telemetry.get("latency_ms", 0.0))
        fps_text = f"{fps:.1f} FPS | {lat:.1f}ms"
        (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cls.draw_chip(
            frame, fps_text,
            pos=(max(12, w - tw - 32), 12),
            border_color=BGR_MINT if fps > 20 else BGR_AMBER,
            text_color=(240, 240, 240)
        )

        # 3. Bottom-Left Chip: Occupancy Count Breakdown
        roi_occ = int(telemetry.get("roi_occupants", 0))
        raw_occ = int(telemetry.get("raw_occupants", roi_occ))
        if raw_occ > roi_occ:
            occ_text = f"ZONE: {roi_occ} ACTIVE ({raw_occ - roi_occ} PERIPHERY)"
        else:
            occ_text = f"ZONE: {roi_occ} OCCUPANT{'S' if roi_occ != 1 else ''}"
        cls.draw_chip(
            frame, occ_text,
            pos=(12, max(12, h - 34)),
            border_color=BGR_MINT if roi_occ > 0 else BGR_SLATE,
            text_color=(255, 255, 255),
            dot_color=BGR_MINT if roi_occ > 0 else None
        )

        # 4. Bottom-Right Chip: Serial Relay & Cycle Countdown
        relay_on = bool(telemetry.get("relay_state", False))
        cycle_state = str(telemetry.get("cycle_state", "EVALUATING"))
        countdown = float(telemetry.get("cycle_countdown", 0.0))
        relay_text = f"RELAY: {'ON [1]' if relay_on else 'OFF [0]'} | {cycle_state[:4]} {countdown:.1f}s"
        (rw, _), _ = cv2.getTextSize(relay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cls.draw_chip(
            frame, relay_text,
            pos=(max(12, w - rw - 32), max(12, h - 34)),
            border_color=BGR_MINT if relay_on else BGR_SLATE,
            text_color=BGR_MINT if relay_on else BGR_SLATE,
            dot_color=BGR_MINT if relay_on else None
        )

    @classmethod
    def draw_roi_vertex_brackets(
        cls,
        frame: np.ndarray,
        norm_vertices: List[Tuple[float, float]],
        is_occupied: bool = False,
        zone_name: str = "SPATIAL ROI"
    ) -> None:
        """
        Renders tactical corner brackets on each vertex of the normalized ROI polygon
        and attaches a zone identification status badge.
        """
        if not norm_vertices or len(norm_vertices) < 3:
            return

        h, w = frame.shape[:2]
        pts = np.array([[int(u * w), int(v * h)] for u, v in norm_vertices], dtype=np.int32)
        n = len(pts)
        color = BGR_MINT if is_occupied else BGR_CYAN
        bracket_len = 16

        for i in range(n):
            p_curr = pts[i].astype(float)
            p_next = pts[(i + 1) % n].astype(float)
            p_prev = pts[(i - 1) % n].astype(float)

            d_next = p_next - p_curr
            l_next = np.linalg.norm(d_next)
            if l_next > 1:
                u_next = d_next / l_next
                p1 = (p_curr + u_next * min(bracket_len, l_next / 2)).astype(int)
                cv2.line(frame, tuple(p_curr.astype(int)), tuple(p1), color, 2, cv2.LINE_AA)

            d_prev = p_prev - p_curr
            l_prev = np.linalg.norm(d_prev)
            if l_prev > 1:
                u_prev = d_prev / l_prev
                p2 = (p_curr + u_prev * min(bracket_len, l_prev / 2)).astype(int)
                cv2.line(frame, tuple(p_curr.astype(int)), tuple(p2), color, 2, cv2.LINE_AA)

        # Status Tag near uppermost vertex
        top_idx = int(np.argmin(pts[:, 1]))
        top_pt = pts[top_idx]
        status_str = f"⬡ {zone_name} [{'OCCUPIED' if is_occupied else 'MONITORING'}]"
        cls.draw_chip(
            frame, status_str,
            pos=(max(10, int(top_pt[0]) - 20), max(10, int(top_pt[1]) - 26)),
            border_color=color,
            text_color=(255, 255, 255),
            dot_color=BGR_MINT if is_occupied else None
        )

    @classmethod
    def draw_duty_sleep_overlay(cls, frame: np.ndarray, countdown: float) -> None:
        """
        Renders a centered glassmorphic power-saving sleep banner.
        Note: Avoids full-frame re-darkening since eco_engine already applies base dimming.
        """
        h, w = frame.shape[:2]
        bw, bh = min(440, w - 40), 54
        if bw < 100 or bh < 20:
            return

        bx1 = (w - bw) // 2
        by1 = 45
        bx2, by2 = bx1 + bw, by1 + bh

        sub = frame[by1:by2, bx1:bx2]
        if sub.shape[0] != bh or sub.shape[1] != bw:
            return

        bg = np.full_like(sub, BGR_CARD_BG)
        cv2.addWeighted(bg, 0.88, sub, 0.12, 0, sub)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), BGR_AMBER, 1, cv2.LINE_AA)

        # Pulsing Amber status dot
        cv2.circle(frame, (bx1 + 20, by1 + 20), 5, BGR_AMBER, -1, cv2.LINE_AA)

        # Banner Typography
        cv2.putText(
            frame, f"POWER-SAVING SLEEP — RESUMES IN {countdown:.1f}s",
            (bx1 + 34, by1 + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            frame, "Inference paused to conserve energy | Presence state locked",
            (bx1 + 34, by1 + 42),
            cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 160, 160), 1, cv2.LINE_AA
        )


# ============================================================================
# Coordinate Parser & Validator (Tab 2 Helper)
# ============================================================================

def parse_and_validate_roi(coord_text: str) -> Tuple[bool, Union[List[Tuple[float, float]], str]]:
    """
    Parses normalized (u, v) points from arbitrary coordinate strings.
    Validates: >= 3 vertices, even count of numbers, and u, v in [0.0, 1.0].
    """
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", coord_text)
    if len(nums) < 6:
        return False, "Polygon requires at least 3 vertices (6 coordinate values)."
    if len(nums) % 2 != 0:
        return False, "Coordinates must be in (u, v) pairs (even count)."
    vertices: List[Tuple[float, float]] = []
    for i in range(0, len(nums), 2):
        try:
            u = float(nums[i])
            v = float(nums[i + 1])
        except ValueError:
            return False, f"Invalid number format: '{nums[i]}', '{nums[i+1]}'."
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            return False, f"Coordinate ({u:.2f}, {v:.2f}) out of range [0.0, 1.0]."
        vertices.append((round(u, 3), round(v, 3)))
    if len(vertices) < 3:
        return False, "At least 3 valid polygon vertices required."
    return True, vertices


# ============================================================================
# Tooltip Helper Widget
# ============================================================================

class ToolTip:
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tw: Optional[tk.Toplevel] = None
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)

    def enter(self, event=None):
        if self.tw:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tw, text=self.text, justify='left',
            background="#161B22", foreground="#F0F6FC", relief='solid', borderwidth=1,
            font=("Segoe UI", 9, "normal"), padx=8, pady=4
        )
        label.pack(ipadx=1)

    def leave(self, event=None):
        if self.tw:
            self.tw.destroy()
            self.tw = None


# ============================================================================
# Main Application Dashboard: AIEcoLightSwitcher
# ============================================================================

class AIEcoLightSwitcher(ctk.CTk):
    """
    AIEcoLightswitcher Modernized Bento-Grid Dashboard.
    Integrates AdvancedDetectionEngine with CustomTkinter 6.0 dark telemetry layout.
    """

    def __init__(
        self,
        engine: Optional[AdvancedDetectionEngine] = None,
        headless: bool = False
    ):
        super().__init__()
        self.title("AIEcoLightswitcher — Edge AI Command Dashboard")
        self.geometry("1280x760")
        self.minsize(1120, 680)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.configure(fg_color=BENTO_THEME["bg_root"])

        self._headless = headless
        if self._headless:
            self.withdraw()

        # Lifecycle & Threading State
        self._is_shutting_down = False
        self._update_after_id: Optional[str] = None
        self._pulse_after_id: Optional[str] = None
        self._pulse_phase = 0
        self._last_telemetry: Dict[str, Any] = {}

        # Edge-Triggered Transition State Trackers for Event Logger
        self._last_log_roi_occupants = -1
        self._last_log_presence: Optional[bool] = None
        self._last_log_cycle: Optional[str] = None
        self._last_log_relay: Optional[bool] = None
        self._last_log_override: Optional[str] = None

        # Event Logger (100-entry ring buffer)
        self.event_logger = EventLogger(maxlen=100)

        # --- INITIALIZE CORE ENGINE ---
        if engine is not None:
            self.engine = engine
        else:
            model_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "yolov8n.pt"))
            if not os.path.exists(model_file):
                model_file = "yolov8n.pt"
            print(f"[App] Initializing AdvancedDetectionEngine (model: {model_file})...")
            self.engine = AdvancedDetectionEngine(source=0, model_path=model_file)
            self.engine.start()

        # Connect initial log entry
        self.event_logger.log("SYSTEM", f"Dashboard initialized on {self.engine.device_info}")

        # --- BUILD BENTO-GRID UI LAYOUT ---
        self._build_ui()

        # Clean shutdown protocol
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start background animation pulse loop (450ms)
        self._pulse_tick()

        # Start non-blocking decoupled UI update loop (20ms, ~50 FPS)
        self.update_loop()

    # ========================================================================
    # UI Layout Construction
    # ========================================================================

    def _build_ui(self):
        # Master Root Grid: Row 0 = Header Ribbon, Row 1 = Main Bento Body
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # 1. Top Bento Header Ribbon (54px)
        self._build_header_ribbon()

        # 2. Main Bento Body Container
        self.bento_body = ctk.CTkFrame(self, fg_color="transparent")
        self.bento_body.grid(row=1, column=0, padx=16, pady=(4, 14), sticky="nsew")
        self.bento_body.grid_rowconfigure(0, weight=1)
        self.bento_body.grid_columnconfigure(0, weight=1)  # Left Primary Vision & Stat Bento
        self.bento_body.grid_columnconfigure(1, weight=0)  # Right 4-Tab Control Station (390px)

        # 3. Left Section: Video Canvas HUD Card + 4 Stat Tiles
        self._build_left_section()

        # 4. Right Section: 390px Sidebar Card hosting CTkTabview
        self._build_right_section()

    def _build_header_ribbon(self):
        """Builds the 54px elevated top header ribbon."""
        self.header_frame = ctk.CTkFrame(
            self, height=54, fg_color=BENTO_THEME["card_surface"],
            border_width=1, border_color=BENTO_THEME["border_subtle"],
            corner_radius=12
        )
        self.header_frame.grid(row=0, column=0, padx=16, pady=(12, 6), sticky="ew")
        self.header_frame.grid_columnconfigure(0, weight=0)  # Brand group
        self.header_frame.grid_columnconfigure(1, weight=1)  # Spacer
        self.header_frame.grid_columnconfigure(2, weight=0)  # Telemetry chips

        # Brand Group (Left)
        brand_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        brand_frame.grid(row=0, column=0, padx=14, pady=8, sticky="w")

        lbl_icon = ctk.CTkLabel(
            brand_frame, text="⚡", font=ctk.CTkFont(size=18, weight="bold"),
            text_color=BENTO_THEME["accent_cyan"]
        )
        lbl_icon.pack(side="left", padx=(0, 6))

        lbl_title = ctk.CTkLabel(
            brand_frame, text="AIEcoLightswitcher",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            text_color=BENTO_THEME["text_primary"]
        )
        lbl_title.pack(side="left", padx=(0, 10))

        badge_cmd = ctk.CTkFrame(
            brand_frame, fg_color=BENTO_THEME["card_surface_alt"],
            border_width=1, border_color=BENTO_THEME["border_subtle"],
            corner_radius=5
        )
        badge_cmd.pack(side="left", padx=(0, 8))
        ctk.CTkLabel(
            badge_cmd, text="EDGE AI COMMAND",
            font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
            text_color=BENTO_THEME["accent_indigo"]
        ).pack(padx=6, pady=2)

        # Telemetry Chips Stack (Right)
        chips_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        chips_frame.grid(row=0, column=2, padx=14, pady=8, sticky="e")

        # 1. Compute Node Chip
        self.chip_compute = ctk.CTkFrame(
            chips_frame, fg_color=BENTO_THEME["card_surface_alt"],
            border_width=1, border_color=BENTO_THEME["border_subtle"],
            corner_radius=6
        )
        self.chip_compute.pack(side="left", padx=4)
        is_cuda = "cuda" in str(self.engine.device).lower()
        dev_label = "⚡ RTX 4050 GPU" if is_cuda else "💻 CPU HOST"
        self.lbl_header_compute = ctk.CTkLabel(
            self.chip_compute, text=dev_label,
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color=BENTO_THEME["accent_cyan"] if is_cuda else BENTO_THEME["text_secondary"]
        )
        self.lbl_header_compute.pack(padx=8, pady=3)
        ToolTip(self.chip_compute, f"Active Inference Compute Device:\n{self.engine.device_info}")

        # 2. Live FPS Chip
        self.chip_fps = ctk.CTkFrame(
            chips_frame, fg_color=BENTO_THEME["card_surface_alt"],
            border_width=1, border_color=BENTO_THEME["border_subtle"],
            corner_radius=6
        )
        self.chip_fps.pack(side="left", padx=4)
        self.lbl_header_fps = ctk.CTkLabel(
            self.chip_fps, text="FPS: 0.0",
            font=ctk.CTkFont(family="Consolas", size=10, weight="bold"),
            text_color=BENTO_THEME["text_primary"]
        )
        self.lbl_header_fps.pack(padx=8, pady=3)

        # 3. Latency Chip
        self.chip_latency = ctk.CTkFrame(
            chips_frame, fg_color=BENTO_THEME["card_surface_alt"],
            border_width=1, border_color=BENTO_THEME["border_subtle"],
            corner_radius=6
        )
        self.chip_latency.pack(side="left", padx=4)
        self.lbl_header_latency = ctk.CTkLabel(
            self.chip_latency, text="0.0 ms",
            font=ctk.CTkFont(family="Consolas", size=10, weight="bold"),
            text_color=BENTO_THEME["accent_cyan"]
        )
        self.lbl_header_latency.pack(padx=8, pady=3)

        # 4. Active Duty Cycle Badge with Pulsing Indicator Dot
        self.chip_duty = ctk.CTkFrame(
            chips_frame, fg_color=BENTO_THEME["card_surface_alt"],
            border_width=1, border_color=BENTO_THEME["border_subtle"],
            corner_radius=6
        )
        self.chip_duty.pack(side="left", padx=4)
        self.lbl_header_duty = ctk.CTkLabel(
            self.chip_duty, text="● EVALUATING (2.0s)",
            font=ctk.CTkFont(family="Consolas", size=10, weight="bold"),
            text_color=BENTO_THEME["status_active_mint"]
        )
        self.lbl_header_duty.pack(padx=8, pady=3)

        # 5. Serial Link Status Chip
        self.chip_serial = ctk.CTkFrame(
            chips_frame, fg_color=BENTO_THEME["card_surface_alt"],
            border_width=1, border_color=BENTO_THEME["border_subtle"],
            corner_radius=6
        )
        self.chip_serial.pack(side="left", padx=(4, 0))
        is_hw_conn = self.engine.hardware.is_connected
        port_desc = self.engine.hardware.port or "COM"
        self.lbl_header_serial = ctk.CTkLabel(
            self.chip_serial,
            text=f"● ARDUINO ONLINE ({port_desc})" if is_hw_conn else "○ ARDUINO OFFLINE",
            font=ctk.CTkFont(family="Consolas", size=10, weight="bold"),
            text_color=BENTO_THEME["status_active_mint"] if is_hw_conn else BENTO_THEME["text_muted"]
        )
        self.lbl_header_serial.pack(padx=8, pady=3)

    def _build_left_section(self):
        """Builds the primary left side: Video Canvas HUD card + 4 Stat Tiles."""
        left_container = ctk.CTkFrame(self.bento_body, fg_color="transparent")
        left_container.grid(row=0, column=0, padx=(0, 10), sticky="nsew")
        left_container.grid_rowconfigure(0, weight=1)  # Video Canvas
        left_container.grid_rowconfigure(1, weight=0)  # Stat Tiles
        left_container.grid_columnconfigure(0, weight=1)

        # ================= UPPER: VIDEO CANVAS HUD CARD =================
        self.video_card = ctk.CTkFrame(
            left_container, fg_color=BENTO_THEME["card_surface"],
            border_width=1, border_color=BENTO_THEME["border_subtle"],
            corner_radius=12
        )
        self.video_card.grid(row=0, column=0, pady=(0, 8), sticky="nsew")
        self.video_card.grid_rowconfigure(0, weight=0)  # Card Header Strip
        self.video_card.grid_rowconfigure(1, weight=1)  # Video Viewport
        self.video_card.grid_columnconfigure(0, weight=1)

        # Video Header Strip
        v_header = ctk.CTkFrame(self.video_card, fg_color="transparent", height=32)
        v_header.grid(row=0, column=0, padx=14, pady=(8, 4), sticky="ew")
        
        lbl_v_title = ctk.CTkLabel(
            v_header, text="LIVE VISION HUD & SPATIAL ROI",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=BENTO_THEME["text_secondary"]
        )
        lbl_v_title.pack(side="left")

        self.lbl_v_source = ctk.CTkLabel(
            v_header, text="SOURCE: WEBCAM 0",
            font=ctk.CTkFont(family="Consolas", size=10, weight="bold"),
            text_color=BENTO_THEME["accent_cyan"]
        )
        self.lbl_v_source.pack(side="right")

        # Recessed Video Viewport Container
        self.video_viewport = ctk.CTkFrame(
            self.video_card, fg_color=BENTO_THEME["card_surface_subtle"],
            corner_radius=8
        )
        self.video_viewport.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.video_viewport.grid_rowconfigure(0, weight=1)
        self.video_viewport.grid_columnconfigure(0, weight=1)

        # Video Label displaying CTkImage
        self.video_label = ctk.CTkLabel(self.video_viewport, text="")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # ================= LOWER: 4 BENTO STAT TILES =================
        self.stat_row = ctk.CTkFrame(left_container, fg_color="transparent", height=82)
        self.stat_row.grid(row=1, column=0, sticky="ew")
        self.stat_row.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="stat_col")

        # Tile 1: Target Confidence
        self.tile_conf = self._create_stat_tile(
            parent=self.stat_row, col=0,
            label="TARGET CONFIDENCE",
            initial_val=f"{int(self.engine.config.conf_threshold * 100)}%",
            val_color=BENTO_THEME["accent_cyan"],
            subtext="YOLOv8n Target Filter"
        )
        self.lbl_tile1_val = self.tile_conf._val_label

        # Tile 2: Smoothed Occupancy
        self.tile_occ = self._create_stat_tile(
            parent=self.stat_row, col=1,
            label="SMOOTHED OCCUPANCY",
            initial_val="VACANT (0)",
            val_color=BENTO_THEME["text_secondary"],
            subtext="Hysteresis M=3 / N=15"
        )
        self.lbl_tile2_val = self.tile_occ._val_label

        # Tile 3: Power Duty Cycle
        self.tile_cycle = self._create_stat_tile(
            parent=self.stat_row, col=2,
            label="POWER DUTY CYCLE",
            initial_val="EVALUATING",
            val_color=BENTO_THEME["status_active_mint"],
            subtext="2.0s Eval / 5.0s Sleep"
        )
        self.lbl_tile3_val = self.tile_cycle._val_label

        # Tile 4: Arduino Relay State
        self.tile_relay = self._create_stat_tile(
            parent=self.stat_row, col=3,
            label="ARDUINO RELAY",
            initial_val="STANDBY ('0')",
            val_color=BENTO_THEME["status_tamper_crimson"],
            subtext="Pin 13 LED Active"
        )
        self.lbl_tile4_val = self.tile_relay._val_label

    def _create_stat_tile(
        self, parent, col: int, label: str, initial_val: str,
        val_color: str, subtext: str
    ) -> ctk.CTkFrame:
        """Helper to construct a cohesive Bento Stat Tile."""
        tile = ctk.CTkFrame(
            parent, fg_color=BENTO_THEME["card_surface"],
            border_width=1, border_color=BENTO_THEME["border_subtle"],
            corner_radius=10, height=80
        )
        tile.grid(row=0, column=col, padx=(0 if col == 0 else 4, 0 if col == 3 else 4), sticky="nsew")

        lbl_top = ctk.CTkLabel(
            tile, text=label,
            font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
            text_color=BENTO_THEME["text_secondary"]
        )
        lbl_top.pack(anchor="w", padx=10, pady=(8, 2))

        val_lbl = ctk.CTkLabel(
            tile, text=initial_val,
            font=ctk.CTkFont(family="Consolas", size=13, weight="bold"),
            text_color=val_color
        )
        val_lbl.pack(anchor="w", padx=10, pady=(0, 2))

        lbl_sub = ctk.CTkLabel(
            tile, text=subtext,
            font=ctk.CTkFont(family="Segoe UI", size=8, weight="normal"),
            text_color=BENTO_THEME["text_muted"]
        )
        lbl_sub.pack(anchor="w", padx=10, pady=(0, 6))

        tile._val_label = val_lbl
        return tile

    def _build_right_section(self):
        """Builds the 390px sidebar card hosting the 4-Tab Control Station."""
        self.sidebar_card = ctk.CTkFrame(
            self.bento_body, width=390, fg_color=BENTO_THEME["card_surface"],
            border_width=1, border_color=BENTO_THEME["border_subtle"],
            corner_radius=12
        )
        self.sidebar_card.grid(row=0, column=1, sticky="nsew")
        self.sidebar_card.grid_propagate(False)

        # 4-Tab Control Station (CTkTabview)
        self.tabview = ctk.CTkTabview(
            self.sidebar_card,
            fg_color="transparent",
            segmented_button_fg_color=BENTO_THEME["bg_root"],
            segmented_button_selected_color=BENTO_THEME["card_surface_alt"],
            segmented_button_selected_hover_color=BENTO_THEME["border_accent"],
            segmented_button_unselected_color=BENTO_THEME["bg_root"],
            segmented_button_unselected_hover_color=BENTO_THEME["card_surface_alt"],
            text_color=BENTO_THEME["text_primary"],
            corner_radius=8
        )
        self.tabview.pack(fill="both", expand=True, padx=8, pady=8)

        self.tab_controls = self.tabview.add("Live Controls")
        self.tab_roi      = self.tabview.add("Spatial ROI")
        self.tab_hw       = self.tabview.add("Hardware & Comms")
        self.tab_log      = self.tabview.add("Event Log")

        # Populate Tabs
        self._build_tab1_controls()
        self._build_tab2_roi()
        self._build_tab3_hardware()
        self._build_tab4_event_log()

    # ========================================================================
    # Tab 1: Live Controls & Sliders
    # ========================================================================

    def _build_tab1_controls(self):
        parent = self.tab_controls

        # 1. Quick Presets Card
        card_presets = self._create_sub_card(parent, "QUICK ENVIRONMENT PRESETS")
        p_frame = ctk.CTkFrame(card_presets, fg_color="transparent")
        p_frame.pack(fill="x", padx=10, pady=(0, 10))
        p_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.btn_preset_classroom = ctk.CTkButton(
            p_frame, text="Classroom\n(0.50 / 2s / 5s)",
            font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
            fg_color=BENTO_THEME["card_surface_subtle"],
            hover_color=BENTO_THEME["border_accent"],
            border_width=1, border_color=BENTO_THEME["border_subtle"],
            command=lambda: self.apply_preset("Classroom")
        )
        self.btn_preset_classroom.grid(row=0, column=0, padx=2, sticky="ew")

        self.btn_preset_lab = ctk.CTkButton(
            p_frame, text="Study Lab\n(0.35 / 3s / 10s)",
            font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
            fg_color=BENTO_THEME["card_surface_subtle"],
            hover_color=BENTO_THEME["border_accent"],
            border_width=1, border_color=BENTO_THEME["border_subtle"],
            command=lambda: self.apply_preset("Study Lab")
        )
        self.btn_preset_lab.grid(row=0, column=1, padx=2, sticky="ew")

        self.btn_preset_eco = ctk.CTkButton(
            p_frame, text="Eco Max\n(0.60 / 1s / 15s)",
            font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
            fg_color=BENTO_THEME["card_surface_subtle"],
            hover_color=BENTO_THEME["border_accent"],
            border_width=1, border_color=BENTO_THEME["border_subtle"],
            command=lambda: self.apply_preset("Eco Maximum")
        )
        self.btn_preset_eco.grid(row=0, column=2, padx=2, sticky="ew")

        # 2. Confidence Threshold Slider Card
        card_conf = self._create_sub_card(parent, "DETECTION SENSITIVITY")
        header_conf = ctk.CTkFrame(card_conf, fg_color="transparent")
        header_conf.pack(fill="x", padx=10, pady=(0, 2))
        ctk.CTkLabel(
            header_conf, text="Confidence Threshold",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=BENTO_THEME["text_secondary"]
        ).pack(side="left")
        conf_val = self.engine.config.conf_threshold
        self.lbl_conf_val = ctk.CTkLabel(
            header_conf, text=f"{int(conf_val * 100)}% ({conf_val:.2f})",
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
            text_color=BENTO_THEME["accent_cyan"]
        )
        self.lbl_conf_val.pack(side="right")

        self.slider_conf = ctk.CTkSlider(
            card_conf, from_=0.01, to=1.00, number_of_steps=99,
            progress_color=BENTO_THEME["accent_cyan"],
            button_color=BENTO_THEME["accent_cyan"],
            button_hover_color="#38BDF8",
            command=self.on_conf_slider_change
        )
        self.slider_conf.set(conf_val)
        self.slider_conf.pack(fill="x", padx=10, pady=4)

        sub_conf = ctk.CTkFrame(card_conf, fg_color="transparent")
        sub_conf.pack(fill="x", padx=10, pady=(0, 8))
        ctk.CTkLabel(sub_conf, text="0.01 (High Sensitivity)", font=ctk.CTkFont(size=8), text_color=BENTO_THEME["text_muted"]).pack(side="left")
        ctk.CTkLabel(sub_conf, text="1.00 (Strict)", font=ctk.CTkFont(size=8), text_color=BENTO_THEME["text_muted"]).pack(side="right")

        # 3. Duty Cycle Eval Duration Slider Card
        card_eval = self._create_sub_card(parent, "EVALUATION DURATION (ACTIVE BURST)")
        header_eval = ctk.CTkFrame(card_eval, fg_color="transparent")
        header_eval.pack(fill="x", padx=10, pady=(0, 2))
        ctk.CTkLabel(
            header_eval, text="Inference Active Window",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=BENTO_THEME["text_secondary"]
        ).pack(side="left")
        eval_val = self.engine.config.eval_duration
        self.lbl_eval_val = ctk.CTkLabel(
            header_eval, text=f"{eval_val:.1f}s",
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
            text_color=BENTO_THEME["status_active_mint"]
        )
        self.lbl_eval_val.pack(side="right")

        self.slider_eval = ctk.CTkSlider(
            card_eval, from_=0.5, to=10.0, number_of_steps=19,
            progress_color=BENTO_THEME["status_active_mint"],
            button_color=BENTO_THEME["status_active_mint"],
            button_hover_color="#34D399",
            command=self.on_eval_slider_change
        )
        self.slider_eval.set(eval_val)
        self.slider_eval.pack(fill="x", padx=10, pady=4)

        sub_eval = ctk.CTkFrame(card_eval, fg_color="transparent")
        sub_eval.pack(fill="x", padx=10, pady=(0, 8))
        ctk.CTkLabel(sub_eval, text="0.5s (Quick Burst)", font=ctk.CTkFont(size=8), text_color=BENTO_THEME["text_muted"]).pack(side="left")
        ctk.CTkLabel(sub_eval, text="10.0s (Sustained)", font=ctk.CTkFont(size=8), text_color=BENTO_THEME["text_muted"]).pack(side="right")

        # 4. Duty Cycle Sleep Duration Slider Card
        card_sleep = self._create_sub_card(parent, "POWER-SAVING SLEEP DURATION")
        header_sleep = ctk.CTkFrame(card_sleep, fg_color="transparent")
        header_sleep.pack(fill="x", padx=10, pady=(0, 2))
        ctk.CTkLabel(
            header_sleep, text="GPU Standby Interval",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=BENTO_THEME["text_secondary"]
        ).pack(side="left")
        sleep_val = self.engine.config.sleep_duration
        self.lbl_sleep_val = ctk.CTkLabel(
            header_sleep, text=f"{sleep_val:.1f}s" if sleep_val > 0 else "0.0s (Continuous)",
            font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
            text_color=BENTO_THEME["accent_cyan"] if sleep_val <= 0 else BENTO_THEME["status_sleep_amber"]
        )
        self.lbl_sleep_val.pack(side="right")

        self.slider_sleep = ctk.CTkSlider(
            card_sleep, from_=0.0, to=30.0, number_of_steps=60,
            progress_color=BENTO_THEME["status_sleep_amber"],
            button_color=BENTO_THEME["status_sleep_amber"],
            button_hover_color="#FBBF24",
            command=self.on_sleep_slider_change
        )
        self.slider_sleep.set(sleep_val)
        self.slider_sleep.pack(fill="x", padx=10, pady=4)

        sub_sleep = ctk.CTkFrame(card_sleep, fg_color="transparent")
        sub_sleep.pack(fill="x", padx=10, pady=(0, 8))
        ctk.CTkLabel(sub_sleep, text="0.0s (Continuous Mode)", font=ctk.CTkFont(size=8), text_color=BENTO_THEME["text_muted"]).pack(side="left")
        ctk.CTkLabel(sub_sleep, text="30.0s (Max Savings)", font=ctk.CTkFont(size=8), text_color=BENTO_THEME["text_muted"]).pack(side="right")

    # ========================================================================
    # Tab 2: Spatial ROI & Detection Zones
    # ========================================================================

    def _build_tab2_roi(self):
        parent = self.tab_roi

        # 1. Master ROI Toggle Card
        card_toggle = self._create_sub_card(parent, "SPATIAL ROI FILTERING")
        self.switch_roi = ctk.CTkSwitch(
            card_toggle, text="Enable Spatial Detection Zone",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            progress_color=BENTO_THEME["status_active_mint"],
            command=self.on_roi_toggle
        )
        if self.engine.config.roi_enabled:
            self.switch_roi.select()
        self.switch_roi.pack(padx=10, pady=(4, 4), anchor="w")

        self.lbl_roi_status = ctk.CTkLabel(
            card_toggle,
            text="Status: " + ("ACTIVE (Filtering Inside Zone)" if self.engine.config.roi_enabled else "DISABLED (Full Frame Active)"),
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=BENTO_THEME["status_active_mint"] if self.engine.config.roi_enabled else BENTO_THEME["text_secondary"]
        )
        self.lbl_roi_status.pack(padx=10, pady=(0, 8), anchor="w")

        # 2. Spatial Anchor Mode
        card_anchor = self._create_sub_card(parent, "SPATIAL ANCHOR MODE")
        ctk.CTkLabel(
            card_anchor, text="Physical contact point for zone containment:",
            font=ctk.CTkFont(family="Segoe UI", size=10), text_color=BENTO_THEME["text_secondary"]
        ).pack(padx=10, pady=(0, 4), anchor="w")

        self.seg_anchor = ctk.CTkSegmentedButton(
            card_anchor, values=["Foot Point (Floor)", "Centroid (Body)"],
            selected_color="#2563EB", selected_hover_color="#1D4ED8",
            unselected_color=BENTO_THEME["card_surface_subtle"],
            command=self.on_anchor_mode_change
        )
        current_mode = "Foot Point (Floor)" if self.engine.config.roi_mode == "BOTTOM_CENTER" else "Centroid (Body)"
        self.seg_anchor.set(current_mode)
        self.seg_anchor.pack(fill="x", padx=10, pady=(0, 8))

        # 3. Zone Presets Card
        card_presets = self._create_sub_card(parent, "QUICK ZONE PRESETS")
        preset_frame = ctk.CTkFrame(card_presets, fg_color="transparent")
        preset_frame.pack(fill="x", padx=10, pady=(0, 8))
        preset_frame.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkButton(
            preset_frame, text="Full Frame\n(100% View)",
            font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
            fg_color=BENTO_THEME["card_surface_subtle"],
            hover_color=BENTO_THEME["border_accent"],
            border_width=1, border_color=BENTO_THEME["border_subtle"],
            command=lambda: self.apply_roi_preset("Full Frame")
        ).grid(row=0, column=0, padx=2, sticky="ew")

        ctk.CTkButton(
            preset_frame, text="Center Focus\n(Desks/Table)",
            font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
            fg_color=BENTO_THEME["card_surface_subtle"],
            hover_color=BENTO_THEME["border_accent"],
            border_width=1, border_color=BENTO_THEME["border_subtle"],
            command=lambda: self.apply_roi_preset("Center Focus")
        ).grid(row=0, column=1, padx=2, sticky="ew")

        ctk.CTkButton(
            preset_frame, text="Entrance\n(Doorway)",
            font=ctk.CTkFont(family="Segoe UI", size=9, weight="bold"),
            fg_color=BENTO_THEME["card_surface_subtle"],
            hover_color=BENTO_THEME["border_accent"],
            border_width=1, border_color=BENTO_THEME["border_subtle"],
            command=lambda: self.apply_roi_preset("Entrance / Doorway")
        ).grid(row=0, column=2, padx=2, sticky="ew")

        # 4. Custom Coordinates Card
        card_custom = self._create_sub_card(parent, "CUSTOM POLYGON COORDINATES")
        ctk.CTkLabel(
            card_custom, text="Normalized (u, v) points in [0.0, 1.0]:",
            font=ctk.CTkFont(family="Segoe UI", size=10), text_color=BENTO_THEME["text_secondary"]
        ).pack(padx=10, pady=(0, 4), anchor="w")

        self.entry_roi_coords = ctk.CTkEntry(
            card_custom, font=ctk.CTkFont(family="Consolas", size=10),
            placeholder_text="(0.20, 0.20), (0.80, 0.20), (0.80, 0.80), (0.20, 0.80)"
        )
        self.entry_roi_coords.pack(fill="x", padx=10, pady=4)
        if self.engine.config.roi_polygon:
            coords_str = ", ".join([f"({u:.2f}, {v:.2f})" for u, v in self.engine.config.roi_polygon])
            self.entry_roi_coords.insert(0, coords_str)
        else:
            self.entry_roi_coords.insert(0, "(0.20, 0.20), (0.80, 0.20), (0.80, 0.80), (0.20, 0.80)")

        self.btn_apply_roi = ctk.CTkButton(
            card_custom, text="Apply Custom Zone",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            fg_color="#2563EB", hover_color="#1D4ED8",
            command=self.apply_custom_roi
        )
        self.btn_apply_roi.pack(fill="x", padx=10, pady=6)

        self.lbl_roi_feedback = ctk.CTkLabel(
            card_custom, text="Ready to update polygon",
            font=ctk.CTkFont(family="Segoe UI", size=9),
            text_color=BENTO_THEME["text_secondary"]
        )
        self.lbl_roi_feedback.pack(padx=10, pady=(0, 8), anchor="w")

    # ========================================================================
    # Tab 3: Hardware & Comms
    # ========================================================================

    def _build_tab3_hardware(self):
        parent = self.tab_hw

        # 1. Serial Arduino Connection Card
        card_serial = self._create_sub_card(parent, "USB SERIAL HARDWARE LINK")

        # Port row
        row_port = ctk.CTkFrame(card_serial, fg_color="transparent")
        row_port.pack(fill="x", padx=10, pady=(0, 4))
        row_port.grid_columnconfigure(0, weight=3)
        row_port.grid_columnconfigure(1, weight=1)

        ports = HardwareController.list_available_ports()
        port_opts = list(ports) if ports else []
        port_opts.append("MOCK_COM1 (Simulator)")
        default_port = port_opts[0]

        self.combo_ports = ctk.CTkComboBox(row_port, values=port_opts)
        self.combo_ports.set(default_port)
        self.combo_ports.grid(row=0, column=0, padx=(0, 4), sticky="ew")

        self.btn_refresh_ports = ctk.CTkButton(
            row_port, text="⟳ Rescan", font=ctk.CTkFont(family="Segoe UI", size=10),
            fg_color=BENTO_THEME["card_surface_subtle"], hover_color=BENTO_THEME["border_accent"],
            border_width=1, border_color=BENTO_THEME["border_subtle"],
            command=self.refresh_ports
        )
        self.btn_refresh_ports.grid(row=0, column=1, sticky="ew")

        # Baud row
        row_baud = ctk.CTkFrame(card_serial, fg_color="transparent")
        row_baud.pack(fill="x", padx=10, pady=4)
        row_baud.grid_columnconfigure(0, weight=1)
        row_baud.grid_columnconfigure(1, weight=2)
        ctk.CTkLabel(row_baud, text="Baud Rate:", font=ctk.CTkFont(family="Segoe UI", size=10), text_color=BENTO_THEME["text_secondary"]).grid(row=0, column=0, sticky="w")
        self.combo_baud = ctk.CTkComboBox(row_baud, values=["9600", "19200", "38400", "57600", "115200"])
        self.combo_baud.set("9600")
        self.combo_baud.grid(row=0, column=1, sticky="ew")

        # Connect/Disconnect button
        self.btn_connect_port = ctk.CTkButton(
            card_serial, text="Connect Serial",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color=BENTO_THEME["status_active_mint"], hover_color="#059669",
            command=self.toggle_serial_connection
        )
        self.btn_connect_port.pack(fill="x", padx=10, pady=6)

        self.lbl_hw_status = ctk.CTkLabel(
            card_serial,
            text="● CONNECTED" if self.engine.hardware.is_connected else "○ DISCONNECTED",
            font=ctk.CTkFont(family="Consolas", size=10, weight="bold"),
            text_color=BENTO_THEME["status_active_mint"] if self.engine.hardware.is_connected else BENTO_THEME["text_muted"]
        )
        self.lbl_hw_status.pack(padx=10, pady=(0, 8), anchor="w")

        # 2. Video Stream Ingestion Card
        card_cam = self._create_sub_card(parent, "VIDEO STREAM INGESTION")
        self.combo_cam_mode = ctk.CTkComboBox(
            card_cam,
            values=["Webcam 0 (Laptop)", "DroidCam Dev (Virtual Cam 1)", "DroidCam IP (WiFi Stream)", "Mock Synthetic Video"],
            command=self.on_cam_mode_selected
        )
        self.combo_cam_mode.set("Webcam 0 (Laptop)")
        self.combo_cam_mode.pack(fill="x", padx=10, pady=(0, 4))

        self.entry_cam_src = ctk.CTkEntry(
            card_cam, font=ctk.CTkFont(family="Consolas", size=10),
            placeholder_text="0 (Webcam) or http://192.168.1.100:4747/video"
        )
        self.entry_cam_src.insert(0, "0")
        self.entry_cam_src.pack(fill="x", padx=10, pady=4)

        self.btn_apply_cam = ctk.CTkButton(
            card_cam, text="Switch Video Source",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            fg_color="#2563EB", hover_color="#1D4ED8",
            command=self.apply_camera_source
        )
        self.btn_apply_cam.pack(fill="x", padx=10, pady=6)

        self.lbl_cam_status = ctk.CTkLabel(
            card_cam, text="● STREAMING (Webcam 0)",
            font=ctk.CTkFont(family="Consolas", size=10, weight="bold"),
            text_color=BENTO_THEME["status_active_mint"]
        )
        self.lbl_cam_status.pack(padx=10, pady=(0, 8), anchor="w")

        # 3. Manual Overrides Card
        card_override = self._create_sub_card(parent, "LIGHTING RELAY MANUAL OVERRIDES")
        ctk.CTkLabel(
            card_override, text="Bypass AI inference to force hardware state:",
            font=ctk.CTkFont(family="Segoe UI", size=10), text_color=BENTO_THEME["text_secondary"]
        ).pack(padx=10, pady=(0, 4), anchor="w")

        self.seg_override = ctk.CTkSegmentedButton(
            card_override, values=["AUTO", "FORCE_ON", "FORCE_OFF"],
            selected_color="#2563EB", selected_hover_color="#1D4ED8",
            unselected_color=BENTO_THEME["card_surface_subtle"],
            command=self.on_override_change
        )
        self.seg_override.set(self.engine.config.override_mode)
        self.seg_override.pack(fill="x", padx=10, pady=6)

        self.lbl_override_status = ctk.CTkLabel(
            card_override,
            text=f"Current Mode: {self.engine.config.override_mode} (AI Controlled)",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color=BENTO_THEME["accent_cyan"]
        )
        self.lbl_override_status.pack(padx=10, pady=(0, 8), anchor="w")

    # ========================================================================
    # Tab 4: Chronological Event Log
    # ========================================================================

    def _build_tab4_event_log(self):
        parent = self.tab_log

        # Toolbar
        toolbar = ctk.CTkFrame(parent, fg_color="transparent", height=28)
        toolbar.pack(fill="x", padx=6, pady=(4, 4))

        self.lbl_log_count = ctk.CTkLabel(
            toolbar, text="0/100 Events",
            font=ctk.CTkFont(family="Consolas", size=10),
            text_color=BENTO_THEME["text_secondary"]
        )
        self.lbl_log_count.pack(side="left")

        btn_clear = ctk.CTkButton(
            toolbar, text="Clear", width=55, height=22,
            font=ctk.CTkFont(family="Segoe UI", size=9),
            fg_color=BENTO_THEME["card_surface_subtle"],
            hover_color=BENTO_THEME["border_accent"],
            border_width=1, border_color=BENTO_THEME["border_subtle"],
            command=self.clear_event_log
        )
        btn_clear.pack(side="right", padx=(4, 0))

        btn_export = ctk.CTkButton(
            toolbar, text="Export...", width=65, height=22,
            font=ctk.CTkFont(family="Segoe UI", size=9),
            fg_color=BENTO_THEME["card_surface_subtle"],
            hover_color=BENTO_THEME["border_accent"],
            border_width=1, border_color=BENTO_THEME["border_subtle"],
            command=self.export_event_log
        )
        btn_export.pack(side="right")

        # Monospace Textbox
        self.log_textbox = ctk.CTkTextbox(
            parent,
            font=ctk.CTkFont(family="Consolas", size=10),
            fg_color=BENTO_THEME["card_surface_subtle"],
            text_color=BENTO_THEME["text_primary"],
            border_color=BENTO_THEME["border_subtle"],
            border_width=1,
            wrap="none"
        )
        self.log_textbox.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        # Tag Configuration for Event Categories
        self.log_textbox.tag_config("TIME", foreground=BENTO_THEME["text_secondary"])
        self.log_textbox.tag_config("VISION", foreground=BENTO_THEME["accent_cyan"])
        self.log_textbox.tag_config("CYCLE", foreground=BENTO_THEME["status_sleep_amber"])
        self.log_textbox.tag_config("SERIAL", foreground=BENTO_THEME["status_active_mint"])
        self.log_textbox.tag_config("CONFIG", foreground=BENTO_THEME["accent_indigo"])
        self.log_textbox.tag_config("SYSTEM", foreground="#94A3B8")
        self.log_textbox.tag_config("MSG", foreground=BENTO_THEME["text_primary"])

        self.log_textbox.configure(state="disabled")

        # Register UI listener on EventLogger ring buffer
        self.event_logger.add_listener(self._append_log_entry_to_ui)

        # Dump initial buffered events
        for entry in self.event_logger.get_entries():
            self._append_log_entry_to_ui(entry)

    def _create_sub_card(self, parent, title: str) -> ctk.CTkFrame:
        """Helper to create elevated card frames inside the sidebar."""
        card = ctk.CTkFrame(
            parent, fg_color=BENTO_THEME["card_surface_alt"],
            corner_radius=8, border_width=1, border_color=BENTO_THEME["border_subtle"]
        )
        card.pack(fill="x", padx=4, pady=4)
        ctk.CTkLabel(
            card, text=title,
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color=BENTO_THEME["text_secondary"]
        ).pack(anchor="w", padx=10, pady=(6, 4))
        return card

    # ========================================================================
    # Event Logging Operations
    # ========================================================================

    def log_event(self, category: str, message: str, level: str = "INFO"):
        """Convenience method to dispatch a log entry."""
        self.event_logger.log(category, message, level)

    def _append_log_entry_to_ui(self, entry: Dict[str, Any]):
        """Appends entry to CTkTextbox with color tags and prunes to 100 lines."""
        if self._is_shutting_down:
            return
        try:
            self.log_textbox.configure(state="normal")
            t_str = f"[{entry['time']}] "
            c_str = f"[{entry['category']}] "
            m_str = f"{entry['message']}\n"

            self.log_textbox.insert("end", t_str, "TIME")
            self.log_textbox.insert("end", c_str, entry["category"])
            self.log_textbox.insert("end", m_str, "MSG")

            # Prune to strictly <= 100 lines
            line_count = int(self.log_textbox.index("end-1c").split(".")[0])
            if line_count > 100:
                self.log_textbox.delete("1.0", f"{line_count - 100 + 1}.0")

            self.log_textbox.see("end")
            self.log_textbox.configure(state="disabled")

            count = len(self.event_logger.buffer)
            self.lbl_log_count.configure(text=f"{count}/100 Events")
        except Exception:
            pass

    def clear_event_log(self):
        """Clears the ring buffer and the UI textbox."""
        self.event_logger.clear()
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")
        self.lbl_log_count.configure(text="0/100 Events")
        self.log_event("SYSTEM", "Event log cleared by user")

    def export_event_log(self):
        """Exports log history to a text file."""
        entries = self.event_logger.get_entries()
        if not entries:
            return
        default_name = f"telemetry_log_{int(time.time())}.txt"
        try:
            path = filedialog.asksaveasfilename(
                initialfile=default_name,
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
            )
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"# AIEcoLightswitcher Telemetry Log Export - {datetime.now().isoformat()}\n\n")
                    for e in entries:
                        f.write(f"[{e['time']}] [{e['category']}] {e['message']}\n")
                self.log_event("SYSTEM", f"Exported {len(entries)} events to {os.path.basename(path)}")
        except Exception as e:
            self.log_event("SYSTEM", f"Export failed: {e}", level="WARN")

    # ========================================================================
    # Tab 1 Callbacks: Sliders & Presets
    # ========================================================================

    def on_conf_slider_change(self, value: float):
        val = round(float(value), 2)
        self.lbl_conf_val.configure(text=f"{int(val * 100)}% ({val:.2f})")
        self.engine.update_config(conf_threshold=val)
        self.lbl_tile1_val.configure(text=f"{int(val * 100)}%")
        self.log_event("CONFIG", f"Confidence threshold set to {val:.2f}")

    def on_eval_slider_change(self, value: float):
        val = round(float(value), 1)
        self.lbl_eval_val.configure(text=f"{val:.1f}s")
        self.engine.update_config(eval_duration=val)
        self.log_event("CONFIG", f"Duty-cycle eval duration set to {val:.1f}s")

    def on_sleep_slider_change(self, value: float):
        val = round(float(value), 1)
        if val <= 0.0:
            self.lbl_sleep_val.configure(text="0.0s (Continuous)", text_color=BENTO_THEME["accent_cyan"])
            self.engine.update_config(sleep_duration=0.0, duty_cycle_enabled=False)
            self.log_event("CONFIG", "Duty-cycle disabled -> Continuous Evaluation Mode")
        else:
            self.lbl_sleep_val.configure(text=f"{val:.1f}s", text_color=BENTO_THEME["status_sleep_amber"])
            self.engine.update_config(sleep_duration=val, duty_cycle_enabled=True)
            self.log_event("CONFIG", f"Duty-cycle sleep duration set to {val:.1f}s")

    def apply_preset(self, preset_name: str):
        presets = {
            "Classroom":   {"conf": 0.50, "eval": 2.0, "sleep": 5.0},
            "Study Lab":   {"conf": 0.35, "eval": 3.0, "sleep": 10.0},
            "Eco Maximum": {"conf": 0.60, "eval": 1.0, "sleep": 15.0},
        }
        p = presets.get(preset_name)
        if not p:
            return

        # Update sliders & labels
        self.slider_conf.set(p["conf"])
        self.slider_eval.set(p["eval"])
        self.slider_sleep.set(p["sleep"])

        self.lbl_conf_val.configure(text=f"{int(p['conf'] * 100)}% ({p['conf']:.2f})")
        self.lbl_eval_val.configure(text=f"{p['eval']:.1f}s")
        self.lbl_sleep_val.configure(text=f"{p['sleep']:.1f}s", text_color=BENTO_THEME["status_sleep_amber"])
        self.lbl_tile1_val.configure(text=f"{int(p['conf'] * 100)}%")

        # Push to engine
        self.engine.update_config(
            conf_threshold=p["conf"],
            eval_duration=p["eval"],
            sleep_duration=p["sleep"],
            duty_cycle_enabled=True
        )
        self.log_event("CONFIG", f"Applied '{preset_name}' preset: conf={p['conf']}, eval={p['eval']}s, sleep={p['sleep']}s")

    # ========================================================================
    # Tab 2 Callbacks: Spatial ROI
    # ========================================================================

    def on_roi_toggle(self):
        enabled = bool(self.switch_roi.get())
        self.engine.update_config(roi_enabled=enabled)
        if enabled:
            self.lbl_roi_status.configure(
                text="Status: ACTIVE (Filtering Inside Zone)",
                text_color=BENTO_THEME["status_active_mint"]
            )
            self.log_event("VISION", "Spatial ROI zone filtering ENABLED")
        else:
            self.lbl_roi_status.configure(
                text="Status: DISABLED (Full Frame Active)",
                text_color=BENTO_THEME["text_secondary"]
            )
            self.log_event("VISION", "Spatial ROI zone filtering DISABLED (Full frame active)")

    def on_anchor_mode_change(self, mode_str: str):
        mode = "BOTTOM_CENTER" if "Foot" in mode_str else "CENTER"
        self.engine.update_config(roi_mode=mode)
        self.log_event("VISION", f"Spatial anchor mode set to {mode}")

    def apply_roi_preset(self, preset_name: str):
        presets = {
            "Full Frame": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            "Center Focus": [(0.20, 0.20), (0.80, 0.20), (0.80, 0.80), (0.20, 0.80)],
            "Entrance / Doorway": [(0.60, 0.10), (0.95, 0.10), (0.95, 0.95), (0.60, 0.95)],
        }
        poly = presets.get(preset_name)
        if not poly:
            return

        self.switch_roi.select()
        self.lbl_roi_status.configure(
            text=f"Status: ACTIVE ({preset_name})",
            text_color=BENTO_THEME["status_active_mint"]
        )

        coord_str = ", ".join([f"({u:.2f}, {v:.2f})" for u, v in poly])
        self.entry_roi_coords.delete(0, "end")
        self.entry_roi_coords.insert(0, coord_str)

        self.engine.update_config(roi_polygon=poly, roi_enabled=True)
        self.lbl_roi_feedback.configure(
            text=f"Preset '{preset_name}' applied ({len(poly)} pts)",
            text_color=BENTO_THEME["status_active_mint"]
        )
        self.log_event("VISION", f"Applied ROI preset '{preset_name}': {len(poly)} vertices")

    def apply_custom_roi(self):
        text = self.entry_roi_coords.get().strip()
        valid, result = parse_and_validate_roi(text)
        if not valid:
            self.lbl_roi_feedback.configure(text=f"Error: {result}", text_color=BENTO_THEME["status_tamper_crimson"])
            return

        poly = result
        self.switch_roi.select()
        self.lbl_roi_status.configure(
            text=f"Status: ACTIVE (Custom {len(poly)} pts)",
            text_color=BENTO_THEME["status_active_mint"]
        )
        self.lbl_roi_feedback.configure(
            text=f"Custom polygon applied ({len(poly)} vertices)",
            text_color=BENTO_THEME["status_active_mint"]
        )
        self.engine.update_config(roi_polygon=poly, roi_enabled=True)
        self.log_event("VISION", f"Applied custom ROI polygon ({len(poly)} vertices)")

    # ========================================================================
    # Tab 3 Callbacks: Hardware & Comms
    # ========================================================================

    def refresh_ports(self):
        ports = HardwareController.list_available_ports()
        opts = list(ports) if ports else []
        opts.append("MOCK_COM1 (Simulator)")
        cur = self.combo_ports.get()
        self.combo_ports.configure(values=opts)
        if cur in opts:
            self.combo_ports.set(cur)
        else:
            self.combo_ports.set(opts[0])
        self.log_event("SERIAL", f"Rescanned serial ports: {len(ports)} hardware devices found")

    def toggle_serial_connection(self):
        if self.engine.hardware.is_connected:
            self.engine.disconnect_serial()
            self.btn_connect_port.configure(
                text="Connect Serial",
                fg_color=BENTO_THEME["status_active_mint"], hover_color="#059669"
            )
            self.lbl_hw_status.configure(text="○ DISCONNECTED", text_color=BENTO_THEME["text_muted"])
            self.lbl_header_serial.configure(text="○ ARDUINO OFFLINE", text_color=BENTO_THEME["text_muted"])
            self.log_event("SERIAL", "Serial port disconnected")
        else:
            port_val = self.combo_ports.get().strip()
            baud_val = int(self.combo_baud.get().strip())
            port_id = "MOCK_COM1" if "mock" in port_val.lower() else port_val

            if hasattr(self.engine.hardware, "baudrate"):
                self.engine.hardware.baudrate = baud_val

            success = self.engine.connect_serial(port_id)
            if success:
                self.btn_connect_port.configure(
                    text="Disconnect Serial",
                    fg_color=BENTO_THEME["status_tamper_crimson"], hover_color="#DC2626"
                )
                self.lbl_hw_status.configure(
                    text=f"● CONNECTED ({port_id} @ {baud_val})",
                    text_color=BENTO_THEME["status_active_mint"]
                )
                self.lbl_header_serial.configure(
                    text=f"● ARDUINO ONLINE ({port_id})",
                    text_color=BENTO_THEME["status_active_mint"]
                )
                self.log_event("SERIAL", f"Connected to {port_id} @ {baud_val} baud")
            else:
                self.lbl_hw_status.configure(
                    text=f"✕ FAILED ({port_id})",
                    text_color=BENTO_THEME["status_tamper_crimson"]
                )
                self.log_event("SERIAL", f"Failed to connect to {port_id}", level="WARN")

    def on_cam_mode_selected(self, mode_str: str):
        if "Webcam 0" in mode_str:
            self.entry_cam_src.delete(0, "end")
            self.entry_cam_src.insert(0, "0")
        elif "DroidCam Dev" in mode_str:
            self.entry_cam_src.delete(0, "end")
            self.entry_cam_src.insert(0, "1")
        elif "DroidCam IP" in mode_str:
            self.entry_cam_src.delete(0, "end")
            self.entry_cam_src.insert(0, "http://192.168.1.100:4747/video")
        elif "Mock" in mode_str:
            self.entry_cam_src.delete(0, "end")
            self.entry_cam_src.insert(0, "mock")

    def apply_camera_source(self):
        raw_src = self.entry_cam_src.get().strip()
        if raw_src.isdigit():
            src = int(raw_src)
        elif raw_src.lower() in ("mock", "simulation", "-1"):
            src = "mock"
        else:
            src = raw_src

        self.log_event("SYSTEM", f"Switching camera ingestion source to: {src}")
        success = self.engine.switch_source(src)
        if success:
            self.lbl_cam_status.configure(
                text=f"● STREAMING ({src})",
                text_color=BENTO_THEME["status_active_mint"]
            )
            self.lbl_v_source.configure(text=f"SOURCE: {str(src).upper()}")
            self.log_event("SYSTEM", f"Successfully streaming from: {src}")
        else:
            self.lbl_cam_status.configure(
                text=f"⚠ FALLBACK ({src})",
                text_color=BENTO_THEME["status_sleep_amber"]
            )
            self.lbl_v_source.configure(text=f"SOURCE: {str(src).upper()} (FALLBACK)")
            self.log_event("SYSTEM", f"Stream fallback active for source: {src}", level="WARN")

    def on_override_change(self, mode: str):
        self.engine.set_override(mode)
        if mode == "AUTO":
            self.lbl_override_status.configure(
                text="Current Mode: AUTO (AI Controlled)",
                text_color=BENTO_THEME["accent_cyan"]
            )
            self.log_event("CONFIG", "Override switched to AUTO (AI vision in control)")
        elif mode == "FORCE_ON":
            self.lbl_override_status.configure(
                text="Current Mode: FORCE_ON (Relay Closed)",
                text_color=BENTO_THEME["status_active_mint"]
            )
            self.log_event("CONFIG", "Manual FORCE_ON: Sent '1' -> Lights forced ON")
        elif mode == "FORCE_OFF":
            self.lbl_override_status.configure(
                text="Current Mode: FORCE_OFF (Relay Open)",
                text_color=BENTO_THEME["status_tamper_crimson"]
            )
            self.log_event("CONFIG", "Manual FORCE_OFF: Sent '0' -> Lights forced OFF")

    # ========================================================================
    # Animated Pulsing Duty Cycle Indicator Loop
    # ========================================================================

    def _pulse_tick(self):
        """Animates the duty cycle indicator dot at 450ms intervals."""
        if self._is_shutting_down:
            return
        self._pulse_phase = (self._pulse_phase + 1) % 2
        self._update_duty_badge_visual()
        self._pulse_after_id = self.after(450, self._pulse_tick)

    def _update_duty_badge_visual(self):
        """Updates the animated pulse dot on the top header badge."""
        if not hasattr(self, "lbl_header_duty"):
            return

        cycle = self._last_telemetry.get("cycle_state", "EVALUATING")
        countdown = float(self._last_telemetry.get("cycle_countdown", 0.0))
        override = self.engine.config.override_mode

        if override == "FORCE_ON":
            self.lbl_header_duty.configure(
                text="■ OVERRIDE: FORCE_ON",
                text_color=BENTO_THEME["status_active_mint"]
            )
        elif override == "FORCE_OFF":
            self.lbl_header_duty.configure(
                text="■ OVERRIDE: FORCE_OFF",
                text_color=BENTO_THEME["status_tamper_crimson"]
            )
        elif cycle == "EVALUATING":
            dot = "●" if self._pulse_phase == 0 else "○"
            self.lbl_header_duty.configure(
                text=f"{dot} EVALUATING ({countdown:.1f}s)",
                text_color=BENTO_THEME["status_active_mint"]
            )
        else:
            dot = "◐" if self._pulse_phase == 0 else "◑"
            self.lbl_header_duty.configure(
                text=f"{dot} SLEEPING ({countdown:.1f}s)",
                text_color=BENTO_THEME["status_sleep_amber"]
            )

    # ========================================================================
    # Main GUI Update Loop (Non-blocking, 20-25ms)
    # ========================================================================

    def update_loop(self):
        """High-framerate 20ms UI loop consuming decoupled engine telemetry."""
        if self._is_shutting_down:
            return

        frame, metrics = self.engine.get_latest_frame_and_telemetry()
        self._last_telemetry = metrics or {}

        if frame is not None and not self._is_shutting_down:
            cycle_state = metrics.get("cycle_state", "EVALUATING")
            countdown = float(metrics.get("cycle_countdown", 0.0))
            roi_occupants = int(metrics.get("roi_occupants", 0))

            # 1. Render HUD Telemetry Chips onto the OpenCV frame
            HudRenderer.render_hud_telemetry_chips(frame, metrics)

            # 2. Render Tactical Corner Brackets on ROI Polygon Vertices
            if self.engine.config.roi_enabled and self.engine.config.roi_polygon:
                HudRenderer.draw_roi_vertex_brackets(
                    frame, self.engine.config.roi_polygon,
                    is_occupied=(roi_occupants > 0)
                )

            # 3. Render Sleek Duty-Cycle Sleep Overlay Banner
            if cycle_state == "SLEEPING":
                HudRenderer.draw_duty_sleep_overlay(frame, countdown)

            # 4. Convert OpenCV BGR -> RGB -> CTkImage & Display
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame_rgb.shape[:2]

                # Dynamic aspect ratio fitting
                vw = self.video_viewport.winfo_width()
                vh = self.video_viewport.winfo_height()
                if vw > 100 and vh > 100:
                    scale = min(vw / max(w, 1), vh / max(h, 1))
                    target_w = max(320, int(w * scale))
                    target_h = max(240, int(h * scale))
                else:
                    target_w, target_h = 720, int(h * (720 / max(w, 1)))

                img = Image.fromarray(frame_rgb)
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(target_w, target_h))
                self.video_label.configure(image=ctk_img)
                self.video_label.image = ctk_img
            except Exception:
                pass

        # 5. Update Header Telemetry Chips
        if metrics:
            fps = float(metrics.get("fps", 0.0))
            lat = float(metrics.get("latency_ms", 0.0))
            self.lbl_header_fps.configure(text=f"FPS: {fps:.1f}")
            self.lbl_header_latency.configure(text=f"{lat:.1f} ms")

            # 6. Update 4 Bento Stat Tiles
            self._update_stat_tiles(metrics)

            # 7. Check and Dispatch Edge-Triggered State Transitions to Tab 4
            self._check_and_log_state_transitions(metrics)

        # Reschedule next tick at 20ms (50 FPS)
        if not self._is_shutting_down:
            self._update_after_id = self.after(20, self.update_loop)

    def _update_stat_tiles(self, metrics: Dict[str, Any]):
        """Refreshes the 4 lower Bento stat tiles with latest telemetry."""
        # Tile 1: Target Confidence
        conf = float(self.engine.config.conf_threshold)
        self.lbl_tile1_val.configure(text=f"{int(conf * 100)}%")

        # Tile 2: Smoothed Occupancy
        smoothed_presence = bool(metrics.get("smoothed_presence", False))
        roi_occupants = int(metrics.get("roi_occupants", 0))
        if smoothed_presence:
            self.lbl_tile2_val.configure(
                text=f"OCCUPIED ({roi_occupants})",
                text_color=BENTO_THEME["status_active_mint"]
            )
        else:
            self.lbl_tile2_val.configure(
                text="VACANT (0)",
                text_color=BENTO_THEME["text_secondary"]
            )

        # Tile 3: Power Duty Cycle
        cycle_state = metrics.get("cycle_state", "EVALUATING")
        countdown = float(metrics.get("cycle_countdown", 0.0))
        if cycle_state == "EVALUATING":
            self.lbl_tile3_val.configure(
                text=f"EVAL ({countdown:.1f}s)",
                text_color=BENTO_THEME["status_active_mint"]
            )
        else:
            self.lbl_tile3_val.configure(
                text=f"SLEEP ({countdown:.1f}s)",
                text_color=BENTO_THEME["status_sleep_amber"]
            )

        # Tile 4: Arduino Relay State
        relay_state = bool(metrics.get("relay_state", False))
        if relay_state:
            self.lbl_tile4_val.configure(
                text="RELAY ON ('1')",
                text_color=BENTO_THEME["status_active_mint"]
            )
        else:
            self.lbl_tile4_val.configure(
                text="STANDBY ('0')",
                text_color=BENTO_THEME["status_tamper_crimson"]
            )

    def _check_and_log_state_transitions(self, metrics: Dict[str, Any]):
        """Detects edge transitions to log without flood."""
        roi_occ = int(metrics.get("roi_occupants", 0))
        presence = bool(metrics.get("smoothed_presence", False))
        cycle = str(metrics.get("cycle_state", "EVALUATING"))
        countdown = float(metrics.get("cycle_countdown", 0.0))
        relay = bool(metrics.get("relay_state", False))
        override = str(metrics.get("override_mode", "AUTO"))

        # 1. Occupant Count Transitions
        if self._last_log_roi_occupants != roi_occ:
            if roi_occ > 0:
                self.event_logger.log("VISION", f"Occupancy detected: {roi_occ} occupant(s) in active zone")
            elif self._last_log_roi_occupants > 0:
                self.event_logger.log("VISION", "Detection zone cleared: 0 occupants")
            self._last_log_roi_occupants = roi_occ

        # 2. Debounced Presence State Transitions
        if self._last_log_presence != presence:
            if self._last_log_presence is not None:
                state_str = "ACTIVE (Debounce ON confirmed)" if presence else "INACTIVE (Debounce OFF confirmed)"
                self.event_logger.log("VISION", f"Smoothed presence -> {state_str}")
            self._last_log_presence = presence

        # 3. Duty Cycle State Transitions
        if self._last_log_cycle != cycle:
            if self._last_log_cycle is not None:
                self.event_logger.log("CYCLE", f"Duty cycle phase: {self._last_log_cycle} -> {cycle} ({countdown:.1f}s)")
            self._last_log_cycle = cycle

        # 4. Serial Relay Transitions
        if self._last_log_relay != relay:
            if self._last_log_relay is not None:
                cmd = "RELAY CLOSED (Command '1' -> Lights ON)" if relay else "RELAY OPEN (Command '0' -> Lights STANDBY)"
                self.event_logger.log("SERIAL", f"Serial TX: {cmd}")
            self._last_log_relay = relay

        # 5. Override Mode Transitions
        if self._last_log_override != override:
            if self._last_log_override is not None:
                self.event_logger.log("CONFIG", f"Override mode transition -> {override}")
            self._last_log_override = override

    # ========================================================================
    # Clean Application Shutdown Protocol
    # ========================================================================

    def on_closing(self):
        """
        Graceful shutdown protocol:
        1. Cancel pending after() timer to avoid race condition or dangling callbacks.
        2. Signal engine to join worker thread, stop camera stream, and disconnect serial.
        3. Cleanly destroy the Tkinter window.
        """
        if self._is_shutting_down:
            return
        print("[App] Clean shutdown sequence initiated...")
        self._is_shutting_down = True

        # 1. Cancel pending UI and animation timers
        if self._update_after_id is not None:
            try:
                self.after_cancel(self._update_after_id)
            except Exception:
                pass
            self._update_after_id = None

        if self._pulse_after_id is not None:
            try:
                self.after_cancel(self._pulse_after_id)
            except Exception:
                pass
            self._pulse_after_id = None

        # 2. Stop engine background thread and hardware
        try:
            self.engine.stop()
        except Exception as e:
            print(f"[App] Engine stop warning: {e}")

        # 3. Destroy Tkinter window
        try:
            self.destroy()
        except Exception:
            pass
        print("[App] Shutdown complete. All threads joined cleanly.")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    app = AIEcoLightSwitcher()
    app.mainloop()
