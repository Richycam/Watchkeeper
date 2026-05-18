import threading
from dataclasses import dataclass, field
from enum import Enum


class FeedType(Enum):
    RECORDED = "recorded"
    LIVE = "live"


class FeedStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class FeedConfig:
    feed_id: str
    feed_type: FeedType
    source: str
    detector_type: str
    label: str
    detector_params: dict = field(default_factory=dict)
    status: FeedStatus = FeedStatus.IDLE
    progress: float = 0.0
    error_message: str = ""


class FeedRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._feeds: dict = {}

    def register(self, config: FeedConfig):
        with self._lock:
            self._feeds[config.feed_id] = config

    def get(self, feed_id: str):
        with self._lock:
            return self._feeds.get(feed_id)

    def all_feeds(self):
        with self._lock:
            return list(self._feeds.values())

    def update_status(self, feed_id: str, status: FeedStatus,
                      progress: float = None, error: str = ""):
        with self._lock:
            feed = self._feeds.get(feed_id)
            if feed:
                feed.status = status
                if progress is not None:
                    feed.progress = progress
                if error:
                    feed.error_message = error
