import cv2
import torch
import urllib.request

import matplotlib.pyplot as plt

# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# urllib.request.urlretrieve(url, filename)
#
#
# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
#
# midas = torch.hub.load("intel-isl/MiDaS", model_type)
#
#
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# midas.to(device)
# midas.eval()
#
#
#
# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
#
# if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
#     transform = midas_transforms.dpt_transform
# else:
#     transform = midas_transforms.small_transform
#
#
# img = cv2.imread(filename)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# input_batch = transform(img).to(device)
#
#
# with torch.no_grad():
#     prediction = midas(input_batch)
#
#     prediction = torch.nn.functional.interpolate(
#         prediction.unsqueeze(1),
#         size=img.shape[:2],
#         mode="bicubic",
#         align_corners=False,
#     ).squeeze()
#
# output = prediction.cpu().numpy()
# x, y = 100, 150  # Example coordinates
# depth_value = output[y, x]
# depth_value_near = output[y+2, x+2]
#
# print(f"Depth value at ({x}, {y}): {depth_value}")
# print(f"Depth value at ({x+2}, {y+2}): {depth_value_near}")


def get_depth_at_coordinates(x, y, image, midas):
    # print(f"inside_midas")
    # model_type = "DPT_Large"  # MiDaS v3 - Large (highest accuracy, slowest inference speed)
    model_type = "DPT_Hybrid"  # MiDaS v3 - Hybrid (medium accuracy, medium inference speed)
    # model_type = "MiDaS_small" # MiDaS v2.1 - Small (lowest accuracy, highest inference speed)

    # Load the MiDaS model


    # Move model to GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    # Load the transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    # Preprocess the image
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    # Perform depth estimation
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Convert to numpy array and get the depth value at (x, y)
    depth_map = prediction.cpu().numpy()
    depth_value = depth_map[y, x]

    # print(f"Depth value at ({x}, {y}): {depth_value}")

    return depth_value