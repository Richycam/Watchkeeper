import json
import threading
from pathlib import Path
from typing import Callable

from src.detection import Detection, MotionEvent


class DetectionStore:
    def __init__(self, persist_path: Path):
        self._lock = threading.Lock()
        self._detections: list = []
        self._motion_events: list = []
        self._listeners: list = []
        self._persist_path = persist_path
        self._persist_path.mkdir(parents=True, exist_ok=True)
        self._load_persisted()

    def _load_persisted(self):
        index = self._persist_path / "detections.json"
        if not index.exists():
            return
        try:
            with open(index) as f:
                raw = json.load(f)
            for item in raw:
                item["thumbnail_path"] = Path(item.get("thumbnail_path", ""))
                item.pop("detected_at", None)
                self._detections.append(Detection(**item))
        except Exception:
            pass

    def _persist(self):
        index = self._persist_path / "detections.json"
        with open(index, "w") as f:
            json.dump([d.to_dict() for d in self._detections], f, indent=2)

    def add_detection(self, detection: Detection):
        with self._lock:
            self._detections.append(detection)
            self._persist()
        for listener in self._listeners:
            try:
                listener(detection)
            except Exception:
                pass

    def add_motion_event(self, event: MotionEvent):
        with self._lock:
            self._motion_events.append(event)

    def subscribe(self, listener: Callable):
        self._listeners.append(listener)

    def query(self, feed_id=None, detector_type=None, label=None):
        with self._lock:
            results = list(self._detections)
        if feed_id:
            results = [d for d in results if d.feed_id == feed_id]
        if detector_type:
            results = [d for d in results if d.detector_type == detector_type]
        if label:
            results = [d for d in results if label.lower() in d.label.lower()]
        return results

    def all_detections(self):
        with self._lock:
            return list(self._detections)

    def all_motion_events(self):
        with self._lock:
            return list(self._motion_events)

    def stats(self):
        with self._lock:
            total = len(self._detections)
            by_feed: dict = {}
            by_type: dict = {}
            for d in self._detections:
                by_feed[d.feed_id] = by_feed.get(d.feed_id, 0) + 1
                by_type[d.detector_type] = by_type.get(d.detector_type, 0) + 1
        return {"total": total, "by_feed": by_feed, "by_type": by_type}
