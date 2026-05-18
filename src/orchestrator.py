import threading
from pathlib import Path

from src.detection_store import DetectionStore
from src.face_store import FaceStore
from src.feed_registry import FeedConfig, FeedRegistry, FeedStatus, FeedType
from src.detector_anpr import ANPRDetector
from src.detector_face import FaceRecognitionDetector
from src.detector_motion import MotionDetector
from src.detector_red_car import RedCarDetector


class FeedOrchestrator:
    def __init__(self, store: DetectionStore, registry: FeedRegistry,
                 face_store: FaceStore, output_root: Path):
        self.store = store
        self.registry = registry
        self.face_store = face_store
        self.output_root = output_root
        self._active: dict = {}
        self._threads: dict = {}

    def register_feed(self, feed_id, source, feed_type: FeedType,
                      detector_type, label, detector_params=None):
        self.registry.register(FeedConfig(
            feed_id=feed_id, feed_type=feed_type, source=source,
            detector_type=detector_type, label=label,
            detector_params=detector_params or {},
        ))

    def start_feed(self, feed_id):
        config = self.registry.get(feed_id)
        if not config:
            return
        # Allow restart from STOPPED/IDLE/ERROR but not if already running
        det = self._active.get(feed_id)
        if det and isinstance(det, MotionDetector) and det.is_running():
            return
        if config.status == FeedStatus.PROCESSING:
            return
        output_dir = self.output_root / feed_id
        if config.detector_type == "red_car":
            det = RedCarDetector(feed_id, self.store, self.registry, output_dir)
            t = threading.Thread(target=det.process_video, args=(config.source,), daemon=True)
        elif config.detector_type == "face_recognition":
            det = FaceRecognitionDetector(
                feed_id, self.store, self.registry, output_dir, self.face_store
            )
            t = threading.Thread(target=det.process_video, args=(config.source,), daemon=True)
        elif config.detector_type == "anpr":
            plate = config.detector_params.get("target_plate", "")
            det = ANPRDetector(feed_id, self.store, self.registry, output_dir, plate)
            t = threading.Thread(target=det.process_video, args=(config.source,), daemon=True)
        elif config.detector_type == "motion":
            det = MotionDetector(feed_id, self.store, self.registry, output_dir, config.source)
            t = threading.Thread(target=det.start, daemon=True)
        else:
            return
        self._active[feed_id] = det
        self._threads[feed_id] = t
        t.start()

    def stop_feed(self, feed_id):
        det = self._active.get(feed_id)
        if det and hasattr(det, "stop"):
            det.stop()

    def toggle_feed(self, feed_id) -> str:
        config = self.registry.get(feed_id)
        if not config:
            return "unknown"
        det = self._active.get(feed_id)
        # Check if running
        if det:
            running = False
            if isinstance(det, MotionDetector):
                running = det.is_running()
            else:
                t = self._threads.get(feed_id)
                running = t is not None and t.is_alive()
            if running:
                self.stop_feed(feed_id)
                return "stopped"
        self.start_feed(feed_id)
        return "started"

    def is_feed_running(self, feed_id) -> bool:
        det = self._active.get(feed_id)
        if det is None:
            return False
        if isinstance(det, MotionDetector):
            return det.is_running()
        t = self._threads.get(feed_id)
        return t is not None and t.is_alive()

    def start_all(self):
        for feed in self.registry.all_feeds():
            self.start_feed(feed.feed_id)

    def get_face_detector_for_feed(self, feed_id):
        det = self._active.get(feed_id)
        if isinstance(det, FaceRecognitionDetector):
            return det
        return None
