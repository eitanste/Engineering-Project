import cv2
from ultralytics import YOLO
# from ultralytics import YOLO
# import numpy as np
#
#
#
#
# # Load the YOLOv8 pose estimation model
# model = YOLO('yolov8l-pose.pt')
#
# # Perform inference on a video stream (e.g., webcam)
# results = model(source=0, show=True, conf=0.1, save=True)
# print('never here')
#
# for result in results:
#     keypoints = result.keypoints.xy
#     keypoints = np.clip(keypoints, 0, 639)  # Ensure keypoints are within the image bounds
#     print(keypoints)
    # Further processing or visualization
# # Print the extracted wrist positions
# for i, wrists in enumerate(wrist_positions):
#     print(f"Person {i+1}:")
#     if 'left_wrist' in wrists:
#         print(f"  Left wrist: {wrists['left_wrist']}")
#     if 'right_wrist' in wrists:
#         print(f"  Right wrist: {wrists['right_wrist']}")
#
#





def plot_pose_estimation_webcam(model):
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform pose estimation
        results = model(frame)
        # print(results)

        # print('if nothing not good')
        # Extract keypoints and plot them
        # for r in results:
        #     if r.keypoints is not None:
        #         keypoints = r.keypoints.xy.cpu().numpy()  # Get keypoints' xy coordinates
        #         print(keypoints)
        #         for person in keypoints:
        #             for kp in person:
        #                 x, y = kp  # Extract x and y coordinates
        #                 cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)  # Draw green circles for keypoints
        #
        # # Display the frame with pose estimations
        # cv2.imshow("Pose Estimation", frame)

        left_wrist_index = 7
        right_wrist_index = 4

        wrist_positions = []

        # Extract keypoints and plot them
        for r in results:
            print(r)

            if r.keypoints is not None:
                keypoints = r.keypoints.xy.cpu().numpy()  # Get keypoints' xy coordinates
                print(keypoints)

                for person in keypoints:
                    if person.any():
                        print(person)
                        wrist_data = {}
            #             # Extract left wrist
                        print("left_wrist")
                        print(person[9])
        #             if not (left_wrist == [0, 0]).all():  # Check if the keypoint is valid
                        x, y = person[9][0], person[9][1]
                        print(x, y)
                        wrist_data['left_wrist'] = (x, y)
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)  # Draw green circles for keypoints
                        print("rigth_wrist")
                        print(person[10])
                        #             if not (left_wrist == [0, 0]).all():  # Check if the keypoint is valid
                        x, y = person[10][0], person[10][1]
                        print(x, y)
                        wrist_data['rigth_wrist'] = (x, y)
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)  # Draw green circles for keypoints

        cv2.imshow("Pose Estimation", frame)

        # print(wrist_positions)
        # return wrist_positions

        # # Exit on pressing 'q'
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()


# if __name__ == "__main__":
    # main(vid_path=0, vid_out="default_out.mp4")
# Load the YOLOv8 pose estimation model directly from ultralytics
model = YOLO('yolov8n-pose.pt')  # Ensure this is the correct path to your model file

    # Start pose estimation from the webcam
print('print')
wrist_positions = plot_pose_estimation_webcam(model)