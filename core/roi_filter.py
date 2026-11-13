"""
core/roi_filter.py - Resolution-Independent Spatial ROI Filter
Part of AIEcoLightswitcher Modular Architecture (Milestone 1)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


class SpatialROIFilter:
    """
    Manages normalized polygon ROI definition and evaluates spatial containment
    using OpenCV pointPolygonTest with resolution-independent coordinates.
    """
    def __init__(
        self,
        normalized_vertices: Optional[List[Tuple[float, float]]] = None,
        mode: str = "BOTTOM_CENTER",
        enabled: bool = False
    ):
        self.mode = mode.upper().strip()
        self.enabled = bool(enabled)
        self.vertices = self._sanitize_vertices(normalized_vertices)
        self._cached_poly_px = None
        self._last_shape = None

    def _sanitize_vertices(
        self, vertices: Optional[List[Tuple[float, float]]]
    ) -> List[Tuple[float, float]]:
        """Ensures vertices are valid normalized coordinates in [0.0, 1.0]."""
        if not vertices or len(vertices) < 3:
            return [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        return [(max(0.0, min(1.0, float(u))), max(0.0, min(1.0, float(v)))) for u, v in vertices]

    def update_polygon(
        self,
        normalized_vertices: Optional[List[Tuple[float, float]]] = None,
        vertices: Optional[List[Tuple[float, float]]] = None,
        mode: Optional[str] = None,
        enabled: Optional[bool] = None
    ) -> None:
        """Updates the ROI polygon vertices, mode, and enabled state."""
        verts = normalized_vertices if normalized_vertices is not None else vertices
        if verts is not None:
            self.vertices = self._sanitize_vertices(verts)
            self._cached_poly_px = None
        if mode is not None:
            self.set_mode(mode)
        if enabled is not None:
            self.set_enabled(enabled)

    def set_mode(self, mode: str) -> None:
        """Sets the anchor mode: 'BOTTOM_CENTER' (feet) or 'CENTER' (centroid)."""
        self.mode = mode.upper().strip()

    def set_enabled(self, enabled: bool) -> None:
        """Enables or disables ROI filtering."""
        self.enabled = bool(enabled)

    def get_anchor_point(
        self, xyxy: Union[np.ndarray, List[int], Tuple[int, ...]]
    ) -> Tuple[int, int]:
        """
        Computes the spatial anchor point for a bounding box [x1, y1, x2, y2].
        BOTTOM_CENTER: ((x1 + x2) // 2, y2) -> physically represents floor/foot contact.
        CENTER: ((x1 + x2) // 2, (y1 + y2) // 2) -> body center.
        """
        x1, y1, x2, y2 = xyxy
        mid_x = int((x1 + x2) / 2)
        if self.mode in ("CENTER", "CENTROID"):
            mid_y = int((y1 + y2) / 2)
            return mid_x, mid_y
        return mid_x, int(y2)

    def _get_pixel_polygon(self, width: int, height: int) -> np.ndarray:
        """Converts normalized vertices to pixel coordinates for the specified frame dimensions."""
        cached = self._cached_poly_px
        if cached is not None and self._last_shape == (width, height):
            return cached
        pts = np.array(
            [[int(u * width), int(v * height)] for u, v in self.vertices],
            dtype=np.int32
        ).reshape((-1, 1, 2))
        self._cached_poly_px = pts
        self._last_shape = (width, height)
        return pts

    def is_point_in_roi(self, point: Tuple[int, int], width: int, height: int) -> bool:
        """Evaluates whether point (x, y) is inside or on the boundary of the polygon."""
        if width <= 0 or height <= 0:
            return False
        px = min(max(0, int(point[0])), width - 1)
        py = min(max(0, int(point[1])), height - 1)
        poly = self._get_pixel_polygon(width, height)
        res = cv2.pointPolygonTest(poly, (float(px), float(py)), measureDist=False)
        return res >= 0

    def filter_detections(
        self,
        detections: List[Dict[str, Any]],
        width: Union[int, Tuple[int, ...], List[int]],
        height: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filters detections by testing spatial anchor containment in the polygon.
        Supports both (detections, width, height) and (detections, frame.shape).

        Returns:
            Tuple[List[Dict], List[Dict]]: (inside_detections, outside_detections)
        """
        if isinstance(width, (tuple, list, np.ndarray)):
            h, w = width[0], width[1]
        else:
            w = int(width)
            h = int(height) if height is not None else w

        if not self.enabled:
            for det in detections:
                if isinstance(det, dict):
                    det["in_roi"] = True
                    fx, fy = self.get_anchor_point(det["xyxy"])
                    det["anchor"] = (min(max(0, int(fx)), w - 1), min(max(0, int(fy)), h - 1))
            return list(detections), []

        inside = []
        outside = []
        for det in detections:
            xyxy = det["xyxy"] if isinstance(det, dict) else det[0]
            fx, fy = self.get_anchor_point(xyxy)
            anchor = (min(max(0, int(fx)), w - 1), min(max(0, int(fy)), h - 1))
            in_zone = self.is_point_in_roi(anchor, w, h)
            if isinstance(det, dict):
                det["anchor"] = anchor
                det["in_roi"] = in_zone
            if in_zone:
                inside.append(det)
            else:
                outside.append(det)
        return inside, outside

    def draw_roi_overlay(
        self,
        frame: np.ndarray,
        is_occupied: bool = False,
        label: Optional[str] = None
    ) -> np.ndarray:
        """
        Draws a translucent polygon overlay on the frame.
        - Active/Occupied: Emerald green tint (alpha 0.22) with vibrant outline.
        - Idle/Empty: Cyan/Blue tint (alpha 0.12) with solid outline.
        """
        if not self.enabled:
            return frame

        h, w = frame.shape[:2]
        pts = self._get_pixel_polygon(w, h)
        overlay = frame.copy()

        color = (50, 220, 50) if is_occupied else (255, 180, 0)
        alpha = 0.22 if is_occupied else 0.12

        cv2.fillPoly(overlay, [pts], color)
        annotated = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)
        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

        pts_xy = pts.reshape(-1, 2)
        min_x = int(np.min(pts_xy[:, 0]))
        min_y = int(np.min(pts_xy[:, 1]))
        badge_text = label or ("ROI: OCCUPIED" if is_occupied else "ROI: ACTIVE (EMPTY)")
        cv2.putText(
            annotated, badge_text,
            (max(10, min_x + 5), max(25, min_y + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
        return annotated
