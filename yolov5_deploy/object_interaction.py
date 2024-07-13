from datetime import datetime, timedelta

class ObjectInteraction:
    def __init__(self, object_id, confidence):
        self.object_id = object_id
        self.detected_time = datetime.now()
        self.notified = False
        self.consecutive_detections = 1
        self.confidence = confidence

    def update(self, confidence):
        self.detected_time = datetime.now()
        self.consecutive_detections += 1
        self.confidence = max(self.confidence, confidence)

    def is_active(self, timeout=30):
        return datetime.now() - self.detected_time < timedelta(seconds=timeout)