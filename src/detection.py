from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Detection:
    feed_id: str
    detector_type: str
    timestamp_seconds: float
    label: str
    confidence: float
    thumbnail_path: Path
    metadata: dict = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self):
        return {
            "feed_id": self.feed_id,
            "detector_type": self.detector_type,
            "timestamp_seconds": self.timestamp_seconds,
            "label": self.label,
            "confidence": self.confidence,
            "thumbnail_path": str(self.thumbnail_path),
            "metadata": self.metadata,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class MotionEvent:
    feed_id: str
    start_time: datetime
    clip_path: Path
    duration_seconds: float = 0.0
