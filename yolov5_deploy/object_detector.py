import math
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from consts import PERSON, dangerous_labels
from notification_manager import NotificationManager
from object_interaction import ObjectInteraction
from datetime import datetime, timedelta


COCO_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head connections
    (5, 6),  # Shoulders
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 11), (6, 12),  # Torso
    (11, 12),  # Hips
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16)  # Right leg
]

class ObjectDetector:


    def __init__(self, notification_manager: NotificationManager, rate_limit_seconds: int = 60):
        self.depth_threshold = 400
        self.notification_manager = notification_manager
        self.object_interactions = {}
        self.detection_threshold = 0.7  # Confidence threshold
        self.min_consecutive_detections = 3  # Minimum frames for confirmation
        self.rate_limit_seconds = rate_limit_seconds
        self.dangerous_labels = dangerous_labels
        self.last_notification_times = {label: datetime.min for label in self.dangerous_labels}
        depth_model_type = "DPT_Hybrid"
        self.depth_estimation_model = torch.hub.load("intel-isl/MiDaS", depth_model_type)
        self.pose_estimation_model = YOLO('yolov8x-pose.pt')
        self.load_model()
        self.keypoints = None


    def detectx(self, frame, model):
        frame = [frame]
        results = model(frame)
        labels, coordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        self.keypoints = self.extract_human_keypoints(frame[0])
        return labels, coordinates

    def extract_human_keypoints(self, frame):
        person_keypoints = []

        # Perform pose estimation
        results = self.pose_estimation_model(frame)

        # Extract keypoints and plot them
        for r in results:
            if r.keypoints is not None:
                keypoints = r.keypoints.xy.cpu().numpy()  # Get keypoints' xy coordinates
                for person in keypoints:
                    if person.any():
                        for kp in person:
                            x, y = kp  # Extract x and y coordinates
                            person_keypoints.append((x, y))
                            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)  # Draw green circles for keypoints

                        # Draw lines between keypoints
                        for partA, partB in COCO_PAIRS:
                            if (person[partA][0] > 0 and person[partA][1] > 0) and (
                                    person[partB][0] > 0 and person[partB][1] > 0):
                                x1, y1 = person[partA]
                                x2, y2 = person[partB]
                                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        return person_keypoints

    def check_interactions_with_dangerous_objects(self, hazards, frame):
        relevant_keypoints_labels = self.check_proximity_in_2D(hazards)
        if relevant_keypoints_labels:
            dangerous_label = self.measure_distance_in_3D(hazards, relevant_keypoints_labels, frame)
            return dangerous_label
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

    def process_detections(self, labels, coordinates):
        hazards = {}
        for label, coordinate in zip(labels, coordinates):
            hazards[label] = coordinate
        return hazards

    def check_proximity_in_2D(self, hazards):
        relevant_keypoints_labels = []
        for label in self.dangerous_labels:
            if is_person_and_hazard_in_one_frame(hazards, label):
                for keypoint in self.keypoints:
                    if (keypoint[0] != 0 and keypoint[1] != 0) and abs(get_center_of_bbox(hazards[label])[0] - keypoint[0]) < 70 and abs(get_center_of_bbox(hazards[label])[1] - keypoint[1]) < 200:
                        relevant_keypoints_labels.append((keypoint, label))
        return relevant_keypoints_labels


    def calculate_weights(self, hazard_bbox_center):
        distances = np.array([np.sqrt((hazard_bbox_center[0] - kp[0])**2 + (hazard_bbox_center[1] - kp[1])**2) for kp in self.keypoints])
        weights = 1 / (distances + 1e-6)  # Avoid division by zero
        normalized_weights = weights / np.sum(weights)
        return normalized_weights


    def measure_distance_in_3D(self, hazards, relevant_keypoints_labels, frame):
        # Compute the depth map once for the entire frame
        # device, transform =
        # self.load_model()
        depth_map = self.get_depth_estimation(self.device, frame, self.transform)

        for keypoint, label in relevant_keypoints_labels:

            hazard_bbox_center = get_center_of_bbox(hazards[label])
            hazard_depth = depth_map[hazard_bbox_center[1], hazard_bbox_center[0]]

            # Calculate weights for each keypoint
            weights = self.calculate_weights(hazard_bbox_center)

            # Compute weighted average depth
            weighted_sum = 0
            try:
                for kp, weight in zip(self.keypoints, weights):
                    if kp[1] ==480 or kp[0] == 640:
                        continue
                    weighted_sum += depth_map[int(kp[1]), int(kp[0])] * weight
            except:
                print("Error calculating weighted sum")
                return None
            weighted_avg_depth = weighted_sum

            depth_diff = abs(hazard_depth - weighted_avg_depth)

            self.depth_debug_prints(depth_diff, hazard_depth, label, weighted_avg_depth)

            if depth_diff < self.depth_threshold:  # Adjust threshold as needed
                return label
        return None

    def depth_debug_prints(self, depth_diff, hazard_depth, label, weighted_avg_depth):
        print(f"label detected: {label}")
        print(f"hazard_depth: {hazard_depth}")
        print(f"weighted_avg_depth: {weighted_avg_depth}")
        print(f"depth_diff: {depth_diff}")

    def get_depth_estimation(self, device, frame, transform):

        # Preprocess the image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(img).to(device)
        # Perform depth estimation
        with torch.no_grad():
            prediction = self.depth_estimation_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        # Convert to numpy array
        depth_map = prediction.cpu().numpy()
        return depth_map

    def load_model(self):
        model_type = "DPT_Hybrid"  # MiDaS v3 - Hybrid (medium accuracy, medium inference speed)
        # Move model to GPU if available
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.depth_estimation_model.to(device)
        self.depth_estimation_model.eval()
        # Load the transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform
        self.device = device
        self.transform = transform
        # return device, transform


def get_center_x_of_bbox(bbox):
    return math.ceil(abs((bbox[0] + bbox[2]) / 2))


def get_center_y_of_bbox(bbox):
    return math.ceil(abs((bbox[1] + bbox[3]) / 2))


def get_center_of_bbox(bbox):
    return [get_center_x_of_bbox(bbox), get_center_y_of_bbox(bbox)]




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
