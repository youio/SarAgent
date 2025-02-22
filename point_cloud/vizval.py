import torch
import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

import os

# Load trained YOLOv8 segmentation model
# apollobot = YOLO("best.pt")  # Change "best.pt" to your trained model's filename
datapath = "./valdata/"



def get_visual_prediction(model, image):
    # # Load and preprocess the test image
    # image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB (YOLO expects RGB)

    # # Run inference
    results = model(image)

    # Process and visualize results
    for result in results:
        masks = result.masks  # Get segmentation masks
        combined = np.zeros(image.shape[:2], np.uint8)
        if masks is not None:
            for mask in masks.data:
                mask = mask.cpu().numpy()  # Convert to NumPy
                mask = (mask * 255).astype(np.uint8)  # Convert to 0-255 range
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))  # Resize to original image size
                combined += mask
    # cv2.imshow("Segmented Output", combined)
    # cv2.waitKey(0)
    return combined
    #         colored_mask = cv2.applyColorMap(combined, cv2.COLORMAP_JET)  # Apply a color map
    #         image = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)  # Blend mask with image

    # # Convert back to BGR for OpenCV display
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # # Show the image
    # cv2.imshow("Segmented Output", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # Run inference
    results = model(image)

    # Visualize results directly using YOLO's built-in plotting
    for result in results:
        image_with_masks = result.plot()  # YOLO handles mask overlay correctly

    # Show the result
    cv2.imshow("Segmented Output", image_with_masks)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# for filename in os.listdir(datapath):
#     get_visual_prediction(apollobot, datapath+filename)