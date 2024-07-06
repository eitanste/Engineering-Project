from consts import PERSON
from notification_manager import NotificationManager


class ObjectDetector:
    def __init__(self, notification_manager: NotificationManager):
        self.notification_manager = notification_manager

    def detect_and_notify(self, frame):
        # Assuming we have a detection function that returns a list of detected objects
        detected_objects = self.detect_objects(frame)
        if self.is_dangerous(detected_objects):
            self.notification_manager.send_message("ALERT!!!! Danger was detected!")
            self.notification_manager.send_photo(frame)

    def detect_objects(self, frame):
        # Placeholder for actual detection logic
        return ["detected_object_1", "detected_object_2"]

    def is_dangerous(self, detected_objects):
        # Placeholder for logic to determine if any detected objects are dangerous
        dangerous_labels = ["knife", "scissors", "bottle"]
        return any(obj in dangerous_labels for obj in detected_objects)

    @staticmethod
    def detectx(frame, model):
        frame = [frame]
        results = model(frame)
        labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        return labels, cordinates


    @staticmethod
    def check_dangerous_labels(hazards):
        from consts import dangerous_labels
        for label in dangerous_labels:
            if check_proximity_in_2D(hazards, label):
                return True
        return False


def check_proximity_in_2D(hazards, label):
    # if is_person_and_hazard_in_one_frame(hazards, label):
    #     print("Distance btw obj is " + str((hazards[PERSON][0] / 2) - hazards[label][0] / 2))
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
    intersection = is_intersecting(hazards, label)
    return intersection


def is_person_and_hazard_in_one_frame(hazards, label):
    return (PERSON in hazards.keys() and label in hazards.keys())

