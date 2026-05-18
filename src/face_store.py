import threading


class FaceStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._references: dict = {}

    def add_reference(self, name: str, embedding):
        with self._lock:
            self._references[name] = embedding

    def remove_reference(self, name: str):
        with self._lock:
            self._references.pop(name, None)

    def all_references(self):
        with self._lock:
            return dict(self._references)

    def names(self):
        with self._lock:
            return list(self._references.keys())
