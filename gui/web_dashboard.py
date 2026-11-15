"""
gui/web_dashboard.py - Responsive Multi-Feed Web Dashboard Server
AIEcoLightswitcher Showcase Edition (Milestone 4 Implementation)

High-performance, pure-Python standard library HTTP server delivering:
- Responsive CSS Grid multi-feed dashboard complying with Modern Web Guidance
- Non-blocking multi-client MJPEG video streaming for active feeds
- Informative offline placeholder rendering for disconnected / standby streams
- REST API for dynamic feed lifecycle (add, remove, toggle, manual override)
- Hardware COM auto-discovery and physical/virtual Arduino synchronization
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
import socket
import sys
import threading
import time
from typing import Any, Dict, Optional
import urllib.parse
import webbrowser

import cv2
import numpy as np

# Ensure root directory is on path
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from core.config import FeedConfig
from hardware.serial_controller import MultiPinHardwareController
from hardware.mock_controller import MockMultiPinHardwareController
from ingestion.base_stream import FeedStatus
from pipeline.multi_feed_manager import MultiFeedManager

try:
    import serial.tools.list_ports
except ImportError:
    serial = None

logger = logging.getLogger("gui.web_dashboard")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler routing REST endpoints, static assets, and MJPEG video streams.
    """
    manager: MultiFeedManager = None

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default verbose HTTP logging to prevent console clutter."""
        pass

    def do_GET(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path in ("/", "/index.html"):
            self._serve_static_file("index.html", "text/html; charset=utf-8")
        elif path.startswith("/static/"):
            filename = path[len("/static/"):]
            mime_type = "text/plain"
            if filename.endswith(".css"):
                mime_type = "text/css; charset=utf-8"
            elif filename.endswith(".js"):
                mime_type = "application/javascript; charset=utf-8"
            elif filename.endswith(".png"):
                mime_type = "image/png"
            elif filename.endswith(".svg"):
                mime_type = "image/svg+xml"
            self._serve_static_file(filename, mime_type)
        elif path == "/api/telemetry":
            self._serve_telemetry()
        elif path == "/api/hardware/ports":
            self._serve_serial_ports()
        elif path.startswith("/api/feed/") and path.endswith("/stream"):
            # Extract feed_id: /api/feed/<feed_id>/stream
            parts = path.split("/")
            if len(parts) >= 4:
                feed_id = parts[3]
                self._stream_feed_mjpeg(feed_id)
            else:
                self.send_error(400, "Invalid feed stream URL format")
        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        body = self._read_json_body()

        if path == "/api/feed/add":
            self._handle_add_feed(body)
        elif path == "/api/feed/remove":
            self._handle_remove_feed(body)
        elif path == "/api/feed/toggle":
            self._handle_toggle_feed(body)
        elif path == "/api/feed/override":
            self._handle_override_feed(body)
        elif path == "/api/hardware/connect":
            self._handle_connect_hardware(body)
        elif path == "/api/hardware/force_sync":
            self._handle_force_sync()
        else:
            self.send_error(404, "Unknown API action")

    def _read_json_body(self) -> Dict[str, Any]:
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 0:
                raw_data = self.rfile.read(content_length).decode("utf-8")
                return json.loads(raw_data)
        except Exception:
            pass
        return {}

    def _serve_json(self, data: Any, status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def _serve_static_file(self, filename: str, content_type: str) -> None:
        filepath = os.path.join(STATIC_DIR, filename)
        if not os.path.exists(filepath):
            self.send_error(404, f"File {filename} not found")
            return
        try:
            with open(filepath, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)
        except Exception as exc:
            self.send_error(500, f"Error reading file: {exc}")

    def _serve_telemetry(self) -> None:
        mgr = DashboardRequestHandler.manager
        if not mgr:
            self._serve_json({"status": "error", "message": "Manager not initialized"}, 500)
            return

        telems = mgr.get_all_feeds_telemetry()
        # Convert FeedStatus enum objects to string values
        clean_telems = {}
        for fid, t in telems.items():
            entry = dict(t)
            if isinstance(entry.get("status"), FeedStatus):
                entry["status"] = entry["status"].value
            clean_telems[fid] = entry

        fps_list = [t.get("fps", 0.0) for t in clean_telems.values() if t.get("fps")]
        global_fps = round(max(fps_list), 1) if fps_list else 0.0

        hw_info = {
            "is_connected": mgr.hardware.is_connected if mgr.hardware else False,
            "is_simulation": mgr.hardware.is_simulation if mgr.hardware else True,
            "port": getattr(mgr.hardware, "port", "SIMULATION")
        }

        dev_str = mgr.device[0] if isinstance(mgr.device, (list, tuple)) else str(mgr.device)

        self._serve_json({
            "feeds": clean_telems,
            "device": dev_str,
            "hardware": hw_info,
            "global_fps": global_fps
        })

    def _serve_serial_ports(self) -> None:
        ports = []
        if serial and hasattr(serial.tools, "list_ports"):
            try:
                for p in serial.tools.list_ports.comports():
                    ports.append({"device": p.device, "description": p.description})
            except Exception:
                pass
        self._serve_json({"ports": ports})

    def _stream_feed_mjpeg(self, feed_id: str) -> None:
        mgr = DashboardRequestHandler.manager
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        # Offline placeholder canvas (640x360 16:9 dark slate)
        placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
        placeholder[:] = (19, 15, 11)  # #0B0F13 dark bento background
        cv2.putText(
            placeholder, "FEED OFFLINE / CONNECTING",
            (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (130, 130, 130), 2
        )
        cv2.putText(
            placeholder, "Waiting for stream signal...",
            (160, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1
        )
        _, offline_bytes = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 60])
        offline_jpeg = offline_bytes.tobytes()

        try:
            while True:
                frame = None
                if mgr:
                    frame, _ = mgr.get_feed_annotated_frame(feed_id)

                if frame is not None:
                    # Compress to lightweight JPEG for web streaming
                    ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    jpg_bytes = jpeg.tobytes() if ret else offline_jpeg
                else:
                    jpg_bytes = offline_jpeg

                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg_bytes)}\r\n\r\n".encode("utf-8"))
                self.wfile.write(jpg_bytes)
                self.wfile.write(b"\r\n")

                time.sleep(0.04)  # ~25 FPS stream
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            logger.debug(f"[web_dashboard] MJPEG stream disconnected: {exc}")

    def _handle_add_feed(self, body: Dict[str, Any]) -> None:
        mgr = DashboardRequestHandler.manager
        if not mgr:
            self._serve_json({"status": "error", "message": "Manager not initialized"}, 500)
            return

        fid = body.get("feed_id", f"feed_{int(time.time())}")
        name = body.get("name", "Camera Feed")
        source = body.get("source", 0)
        pin = int(body.get("pin", 13))

        cfg = FeedConfig(
            feed_id=fid,
            name=name,
            source=source,
            pin=pin,
            enabled=True
        )

        ok = mgr.add_feed(cfg)
        self._serve_json({"status": "ok" if ok else "error", "feed_id": fid})

    def _handle_remove_feed(self, body: Dict[str, Any]) -> None:
        mgr = DashboardRequestHandler.manager
        fid = body.get("feed_id", "")
        ok = mgr.remove_feed(fid) if mgr else False
        self._serve_json({"status": "ok" if ok else "error"})

    def _handle_toggle_feed(self, body: Dict[str, Any]) -> None:
        mgr = DashboardRequestHandler.manager
        fid = body.get("feed_id", "")
        enabled = bool(body.get("enabled", True))
        if mgr:
            ok = mgr.enable_feed(fid) if enabled else mgr.disable_feed(fid)
        else:
            ok = False
        self._serve_json({"status": "ok" if ok else "error"})

    def _handle_override_feed(self, body: Dict[str, Any]) -> None:
        mgr = DashboardRequestHandler.manager
        fid = body.get("feed_id", "")
        mode = body.get("mode", "AUTO")
        ok = mgr.set_feed_override(fid, mode) if mgr else False
        self._serve_json({"status": "ok" if ok else "error"})

    def _handle_connect_hardware(self, body: Dict[str, Any]) -> None:
        mgr = DashboardRequestHandler.manager
        port = body.get("port", "SIMULATION")
        baud = int(body.get("baud", 9600))

        if port == "SIMULATION":
            mgr.hardware = MockMultiPinHardwareController()
            mgr.hardware.connect()
        else:
            mgr.hardware = MultiPinHardwareController(port=port, baudrate=baud)
            mgr.hardware.connect()

        self._serve_json({"status": "ok", "port": port})

    def _handle_force_sync(self) -> None:
        mgr = DashboardRequestHandler.manager
        if mgr and mgr.hardware:
            mgr.hardware.force_sync()
        self._serve_json({"status": "ok"})


def find_free_port(start_port: int = 8080, max_attempts: int = 20) -> int:
    """Finds an available local port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def start_dashboard_server(
    manager: MultiFeedManager,
    port: int = 8080,
    open_browser: bool = True
) -> ThreadingHTTPServer:
    """
    Initializes and starts the ThreadingHTTPServer in a background daemon thread.
    """
    DashboardRequestHandler.manager = manager
    actual_port = find_free_port(port)
    server = ThreadingHTTPServer(("0.0.0.0", actual_port), DashboardRequestHandler)

    thread = threading.Thread(
        target=server.serve_forever,
        name="DashboardServerThread",
        daemon=True
    )
    thread.start()

    url = f"http://localhost:{actual_port}"
    print(f"\n=================================================================")
    print(f"🚀 AIEcoLightswitcher Modern Web Dashboard is LIVE!")
    print(f"📡 Dashboard URL : {url}")
    print(f"⚡ Minimum 4-Card Responsive Grid active (CSS Grid auto-fit)")
    print(f"=================================================================\n")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    return server


if __name__ == "__main__":
    # Standalone Demo Launcher
    logging.basicConfig(level=logging.INFO)
    hw = MockMultiPinHardwareController()
    hw.connect()

    mgr = MultiFeedManager(hardware=hw)
    mgr.start()

    # Pre-populate default showcase feeds (demonstrating minimum 4-card grid + add/subtract)
    mgr.add_feed(FeedConfig(
        feed_id="feed_1",
        name="Front Entrance",
        source="MOCK:PERSON_IN_ROI",
        pin=13
    ))
    mgr.add_feed(FeedConfig(
        feed_id="feed_2",
        name="Study Commons",
        source="MOCK:EMPTY_FRAME",
        pin=12
    ))

    server = start_dashboard_server(mgr, port=8080, open_browser=True)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
        mgr.stop()
        server.shutdown()
