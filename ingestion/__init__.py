"""
ingestion - Universal Video Ingestion & Parallel Stream Engine
Part of AIEcoLightswitcher Modular Architecture (Milestone 1)
"""

from ingestion.base_stream import (
    BaseVideoStream,
    FeedStatus,
    StreamTelemetry,
)
from ingestion.mock_stream import (
    MockVideoSource,
    create_mock_feed_grid,
)
from ingestion.stream_worker import (
    BufferedFrame,
    SingleFrameBuffer,
    StreamWorker,
)
from ingestion.video_stream import (
    VideoStream,
    open_capture,
)

__all__ = [
    "BaseVideoStream",
    "BufferedFrame",
    "FeedStatus",
    "MockVideoSource",
    "SingleFrameBuffer",
    "StreamTelemetry",
    "StreamWorker",
    "VideoStream",
    "create_mock_feed_grid",
    "open_capture",
]
