"""
pipeline package - Multi-Camera Stream Ingestion, Inference & Hardware Routing Pipeline
AIEcoLightswitcher Showcase Edition (Milestone 3 Implementation)
"""

from pipeline.feed_pipeline import FeedPipeline
from pipeline.multi_feed_manager import MultiFeedManager

__all__ = [
    "FeedPipeline",
    "MultiFeedManager",
]
