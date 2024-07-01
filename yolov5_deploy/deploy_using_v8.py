import cv2

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

def extract_human_keypoints(model, frame):
    person_keypoints = []

    # Perform pose estimation
    results = model(frame)

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
                        if (person[partA][0] > 0 and person[partA][1] > 0) and (person[partB][0] > 0 and person[partB][1] > 0):
                            x1, y1 = person[partA]
                            x2, y2 = person[partB]
                            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)


    return person_keypoints
