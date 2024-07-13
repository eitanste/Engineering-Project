# from consts import PERSON
# from notification_manager import NotificationManager
#
# class ObjectDetector:
#     def __init__(self, notification_manager: NotificationManager):
#         self.notification_manager = notification_manager
#         self.object_interactions = {}
#         self.detection_threshold = 0.7  # Confidence threshold
#         self.min_consecutive_detections = 3  # Minimum frames for confirmation
#
#     @staticmethod
#     def detectx(frame, model):
#         frame = [frame]
#         results = model(frame)
#         labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
#
#         return labels, cordinates
#
#     @staticmethod
#     def check_interactions_with_dangerous_objects(hazards):
#         from consts import dangerous_labels
#         for label in dangerous_labels:
#             if check_proximity_in_2D(hazards, label):
#                 return True
#         return False
#
#     def update_interactions(self, detected_objects, confidences):
#         from object_interaction import ObjectInteraction
#         from datetime import datetime
#
#         current_time = datetime.now()
#         detected_ids = {obj_id for obj_id in detected_objects}
#         for obj_id, confidence in zip(detected_objects, confidences):
#             if obj_id not in self.object_interactions:
#                 self.object_interactions[obj_id] = ObjectInteraction(obj_id, confidence)
#             else:
#                 self.object_interactions[obj_id].update(confidence)
#
#         # Remove inactive interactions
#         self.object_interactions = {
#             obj_id: interaction for obj_id, interaction in self.object_interactions.items()
#             if interaction.is_active() and obj_id in detected_ids
#         }
#
#     def verify_and_notify(self, frame, hazards):
#         dangerous_labels = ["knife", "scissors", "bottle"]
#         for obj_id, interaction in self.object_interactions.items():
#             if (obj_id in dangerous_labels and
#                     not interaction.notified and
#                     (interaction.consecutive_detections >= self.min_consecutive_detections or
#                      interaction.confidence >= self.detection_threshold)):
#
#                 if self.check_interactions_with_dangerous_objects(hazards):
#                     self.notification_manager.send_message("ALERT!!!! Danger was detected!")
#                     self.notification_manager.send_photo(frame)
#                     interaction.notified = True
#
#     def detect_and_notify(self, frame, model):
#         labels, coordinates = self.detectx(frame, model)
#         hazards = self.process_detections(labels, coordinates)
#         self.update_interactions(labels, [1.0] * len(labels))  # Assuming a confidence of 1.0 for simplicity
#         self.verify_and_notify(frame, hazards)
#
#     def process_detections(self, labels, coordinates):
#         hazards = {}
#         for label, coordinate in zip(labels, coordinates):
#             hazards[label] = coordinate
#         return hazards
#
#
# def check_proximity_in_2D(hazards, label):
#     # if is_person_and_hazard_in_one_frame(hazards, label):
#     #     print("Distance btw obj is " + str((hazards[PERSON][0] / 2) - hazards[label][0] / 2))
#     return is_person_and_hazard_in_one_frame(hazards, label) and measure_distance_in_2D(hazards, label)
#
#
# def is_intersecting(hazards, label):
#     return is_Y_intersect(hazards, label) and is_X_intersect(hazards, label)
#
#
# def is_X_intersect(hazards, label):
#     return ((hazards[label][0] >= hazards[PERSON][0] and hazards[label][0] <= hazards[PERSON][2])
#             or (hazards[label][2] >= hazards[PERSON][0] and hazards[label][2] <= hazards[PERSON][2]))
#
#
# def is_Y_intersect(hazards, label):
#     return ((hazards[label][1] >= hazards[PERSON][1] and hazards[label][1] <= hazards[PERSON][3])
#             or (hazards[label][3] >= hazards[PERSON][1] and hazards[label][3] <= hazards[PERSON][3]))
#
#
# def measure_distance_in_2D(hazards, label):
#     intersection = is_intersecting(hazards, label)
#     return intersection
#
#
# def is_person_and_hazard_in_one_frame(hazards, label):
#     return (PERSON in hazards.keys() and label in hazards.keys())
#


from consts import PERSON, dangerous_labels
from notification_manager import NotificationManager
from object_interaction import ObjectInteraction
from datetime import datetime, timedelta


class ObjectDetector:
    def __init__(self, notification_manager: NotificationManager, rate_limit_seconds: int = 60):
        self.notification_manager = notification_manager
        self.object_interactions = {}
        self.detection_threshold = 0.7  # Confidence threshold
        self.min_consecutive_detections = 3  # Minimum frames for confirmation
        self.rate_limit_seconds = rate_limit_seconds
        self.dangerous_labels = dangerous_labels
        self.last_notification_times = {label: datetime.min for label in dangerous_labels}

    @staticmethod
    def detectx(frame, model):
        frame = [frame]
        results = model(frame)
        labels, coordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, coordinates

    def check_interactions_with_dangerous_objects(self, hazards):
        for label in self.dangerous_labels:
            if check_proximity_in_2D(hazards, label):
                return label
        return None

    def update_interactions(self, detected_objects, confidences):
        current_time = datetime.now()
        detected_ids = {obj_id for obj_id in detected_objects}
        for obj_id, confidence in zip(detected_objects, confidences):
            if obj_id not in self.object_interactions:
                self.object_interactions[obj_id] = ObjectInteraction(obj_id, confidence)
            else:
                self.object_interactions[obj_id].update(confidence)

        # Remove inactive interactions
        self.object_interactions = {
            obj_id: interaction for obj_id, interaction in self.object_interactions.items()
            if interaction.is_active() and obj_id in detected_ids
        }

    def verify_and_notify(self, frame, hazards):
        dangerous_label = self.check_interactions_with_dangerous_objects(hazards)
        if dangerous_label:
            current_time = datetime.now()
            last_notification_time = self.last_notification_times[dangerous_label]
            if current_time - last_notification_time >= timedelta(seconds=self.rate_limit_seconds):
                self.notification_manager.send_message(f"ALERT!!!! Danger was detected: {dangerous_label}!")
                self.notification_manager.send_photo(frame)
                self.last_notification_times[dangerous_label] = current_time

    def detect_and_notify(self, frame, model):
        labels, coordinates = self.detectx(frame, model)
        confidences = [1.0] * len(labels)  # Assuming a confidence of 1.0 for simplicity

        hazards = self.process_detections(labels, coordinates)
        self.update_interactions(labels, confidences)  # Update interactions with labels and confidences
        self.verify_and_notify(frame, hazards)

    def process_detections(self, labels, coordinates):
        hazards = {}
        for label, coordinate in zip(labels, coordinates):
            hazards[label] = coordinate
        return hazards


def check_proximity_in_2D(hazards, label):
    return is_person_and_hazard_in_one_frame(hazards, label) and measure_distance_in_2D(hazards, label)


def is_intersecting(hazards, label):
    return is_Y_intersect(hazards, label) and is_X_intersect(hazards, label)


def is_X_intersect(hazards, label):
    return ((hazards[label][0] >= hazards[PERSON][0] and hazards[label][0] <= hazards[PERSON][2])
            or (hazards[label][2] >= hazards[PERSON][0] and hazards[label][2] <= hazards[PERSON][2]))


def is_Y_intersect(hazards, label):
    return ((hazards[label][1] >= hazards[PERSON][1] and hazards[label][1] <= hazards[PERSON][3])
            or (hazards[label][3] >= hazards[PERSON][1] and hazards[label][3] <= hazards[PERSON][3]))


def measure_distance_in_2D(hazards, label):
    return is_intersecting(hazards, label)


def is_person_and_hazard_in_one_frame(hazards, label):
    return (PERSON in hazards.keys() and label in hazards.keys())
