"""
tests/test_web_dashboard.py - Automated Verification Suite for Responsive Web Dashboard
Part of AIEcoLightswitcher Milestone 4 Implementation
"""

import json
import time
import unittest
import urllib.request
import urllib.error

from core.config import FeedConfig
from hardware.mock_controller import MockMultiPinHardwareController
from pipeline.multi_feed_manager import MultiFeedManager
from gui.web_dashboard import start_dashboard_server, find_free_port


class TestWebDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hw = MockMultiPinHardwareController()
        cls.mgr = MultiFeedManager(hardware=cls.hw)
        cls.mgr.start()

        # Add initial mock feeds
        cls.mgr.add_feed(FeedConfig(
            feed_id="cam_entry",
            name="Main Entrance",
            source="MOCK:PERSON_IN_ROI",
            pin=13
        ))
        cls.mgr.add_feed(FeedConfig(
            feed_id="cam_exit",
            name="Emergency Exit",
            source="MOCK:EMPTY_FRAME",
            pin=12
        ))

        cls.port = find_free_port(8890)
        cls.server = start_dashboard_server(cls.mgr, port=cls.port, open_browser=False)
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.mgr.stop()
        cls.server.shutdown()

    def test_01_index_html_serving(self):
        """Verify root endpoint / serves index.html with 200 OK and HTML content."""
        req = urllib.request.Request(f"{self.base_url}/")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            content = resp.read().decode("utf-8")
            self.assertIn("AIEcoLightswitcher", content)
            self.assertIn("feed-grid-container", content)

    def test_02_static_css_and_js_serving(self):
        """Verify /static/styles.css and /static/app.js serve with correct MIME types."""
        css_req = urllib.request.Request(f"{self.base_url}/static/styles.css")
        with urllib.request.urlopen(css_req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/css", resp.headers.get("Content-Type", ""))
            content = resp.read().decode("utf-8")
            self.assertIn("feed-grid", content)

        js_req = urllib.request.Request(f"{self.base_url}/static/app.js")
        with urllib.request.urlopen(js_req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("javascript", resp.headers.get("Content-Type", ""))
            content = resp.read().decode("utf-8")
            self.assertIn("pollTelemetry", content)

    def test_03_telemetry_endpoint(self):
        """Verify /api/telemetry returns full feed registry and hardware states."""
        req = urllib.request.Request(f"{self.base_url}/api/telemetry")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("feeds", data)
            self.assertIn("cam_entry", data["feeds"])
            self.assertIn("cam_exit", data["feeds"])
            self.assertEqual(data["feeds"]["cam_entry"]["pin"], 13)

    def test_04_dynamic_add_and_remove_feed_api(self):
        """Verify POST /api/feed/add and POST /api/feed/remove."""
        # 1. Add new feed via POST
        add_payload = json.dumps({
            "feed_id": "test_wifi_cam",
            "name": "Wi-Fi Corridor",
            "source": "MOCK:PERSON_IN_ROI",
            "pin": 11
        }).encode("utf-8")
        req_add = urllib.request.Request(
            f"{self.base_url}/api/feed/add",
            data=add_payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req_add, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            res = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(res.get("status"), "ok")

        # Verify added to telemetry
        telems = self.mgr.get_all_feeds_telemetry()
        self.assertIn("test_wifi_cam", telems)

        # 2. Remove feed via POST
        rem_payload = json.dumps({"feed_id": "test_wifi_cam"}).encode("utf-8")
        req_rem = urllib.request.Request(
            f"{self.base_url}/api/feed/remove",
            data=rem_payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req_rem, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)
            res = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(res.get("status"), "ok")

        telems_after = self.mgr.get_all_feeds_telemetry()
        self.assertNotIn("test_wifi_cam", telems_after)

    def test_05_manual_override_api(self):
        """Verify POST /api/feed/override actuates hardware pin."""
        ov_payload = json.dumps({
            "feed_id": "cam_exit",
            "mode": "FORCE_ON"
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/feed/override",
            data=ov_payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)

        # Pin 12 must be driven HIGH by FORCE_ON
        self.assertTrue(self.hw.get_pin(12))

        # Revert to FORCE_OFF
        ov_off = json.dumps({
            "feed_id": "cam_exit",
            "mode": "FORCE_OFF"
        }).encode("utf-8")
        req_off = urllib.request.Request(
            f"{self.base_url}/api/feed/override",
            data=ov_off,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req_off, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)

        self.assertFalse(self.hw.get_pin(12))

    def test_06_toggle_feed_api(self):
        """Verify POST /api/feed/toggle enables and disables feed."""
        dis_payload = json.dumps({
            "feed_id": "cam_entry",
            "enabled": False
        }).encode("utf-8")
        req_dis = urllib.request.Request(
            f"{self.base_url}/api/feed/toggle",
            data=dis_payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req_dis, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)

        self.assertFalse(self.mgr.get_feed_telemetry("cam_entry")["enabled"])

        # Re-enable
        en_payload = json.dumps({
            "feed_id": "cam_entry",
            "enabled": True
        }).encode("utf-8")
        req_en = urllib.request.Request(
            f"{self.base_url}/api/feed/toggle",
            data=en_payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req_en, timeout=3.0) as resp:
            self.assertEqual(resp.status, 200)

        self.assertTrue(self.mgr.get_feed_telemetry("cam_entry")["enabled"])


if __name__ == "__main__":
    unittest.main()
