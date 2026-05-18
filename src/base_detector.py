import os
import queue
import threading
from abc import ABC, abstractmethod
from pathlib import Path

import cv2

from src.feed_registry import FeedRegistry, FeedStatus
from src.detection_store import DetectionStore

WORKER_THREADS = 4


class BaseDetector(ABC):
    def __init__(self, feed_id: str, store: DetectionStore,
                 registry: FeedRegistry, output_dir: Path):
        self.feed_id = feed_id
        self.store = store
        self.registry = registry
        self.output_dir = output_dir
        self._stop_event = threading.Event()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def detector_type(self) -> str:
        pass

    @abstractmethod
    def detect_frame(self, frame, timestamp_seconds: float) -> list:
        pass

    def stop(self):
        self._stop_event.set()

    def is_running(self) -> bool:
        return False

    def save_thumbnail(self, frame, timestamp_seconds: float, suffix: str = "") -> Path:
        filename = self.feed_id + "_" + str(round(timestamp_seconds, 2)) + suffix + ".jpg"
        path = self.output_dir / filename
        cv2.imwrite(str(path), frame)
        return path

    def _suppress_stderr(self):
        devnull = os.open(os.devnull, os.O_WRONLY)
        old = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)
        return old

    def _restore_stderr(self, old: int):
        os.dup2(old, 2)
        os.close(old)

    def _resize_for_inference(self, frame, max_width: int = 960):
        h, w = frame.shape[:2]
        if w <= max_width:
            return frame
        scale = max_width / w
        return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

    def process_video(self, source: str, sample_rate: float = 0.5):
        old = self._suppress_stderr()
        cap = cv2.VideoCapture(source)
        self._restore_stderr(old)
        if not cap.isOpened():
            self.registry.update_status(
                self.feed_id, FeedStatus.ERROR, error="Cannot open: " + source
            )
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_interval = max(1, int(fps * sample_rate))
        frame_queue: queue.Queue = queue.Queue(maxsize=WORKER_THREADS * 2)
        frame_number = 0

        def worker():
            while True:
                item = frame_queue.get()
                if item is None:
                    frame_queue.task_done()
                    break
                frm, ts = item
                small = self._resize_for_inference(frm)
                for d in self.detect_frame(small, ts):
                    self.store.add_detection(d)
                frame_queue.task_done()

        workers = [threading.Thread(target=worker, daemon=True)
                   for _ in range(WORKER_THREADS)]
        for w in workers:
            w.start()
        self.registry.update_status(self.feed_id, FeedStatus.PROCESSING, progress=0.0)

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_number % frame_interval == 0:
                frame_queue.put((frame.copy(), frame_number / fps))
            frame_number += 1
            if total_frames > 0:
                self.registry.update_status(
                    self.feed_id, FeedStatus.PROCESSING,
                    progress=(frame_number / total_frames) * 100,
                )

        for _ in workers:
            frame_queue.put(None)
        frame_queue.join()
        cap.release()

        if self._stop_event.is_set():
            self.registry.update_status(self.feed_id, FeedStatus.STOPPED)
        else:
            self.registry.update_status(self.feed_id, FeedStatus.COMPLETE, progress=100.0)
