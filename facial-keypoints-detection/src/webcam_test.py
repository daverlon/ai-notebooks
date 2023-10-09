import torch 
import torch.nn as nn

import numpy as np

import cv2 as cv

from model import MiniGoogLeNet

if __name__ == "__main__":

    model = MiniGoogLeNet()
    model.load_state_dict(torch.load("../nets/testa124.pt", map_location='cpu'))

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to find video capture device.")
        exit(1)

    while True:
        go, frame = cap.read()
        if not go:
            print("Failed to retrieve frame")
            break
            
        frame_height, frame_width, _ = frame.shape


        top_left_x = (frame_width - 96) // 2
        top_left_y = (frame_height - 96) // 2
        bottom_right_x = top_left_x + 96
        bottom_right_y = top_left_y + 96

        center_region = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]


        # frame_tensor = torch.from_numpy(np.transpose(center_region (2, 0, 1))).float()
        ten = torch.from_numpy(np.transpose(center_region)).type(torch.float)
        # print(ten.shape)

        pred = model(ten.unsqueeze(dim=0)).squeeze().detach().numpy()


        # Extract the predictions for "left" and "right"
        left = [int(pred[0]), int(pred[1])]  # Scale to the 96x96 region

        right = [int(pred[2]), int(pred[3])]  # Scale to the 96x96 region

        # Draw circles representing "left" and "right" points on the center region
        cv.circle(center_region, (left[0], left[1]), 5, (0, 0, 255), -1)  # Red circle for "left"
        cv.circle(center_region, (right[0], right[1]), 5, (0, 255, 0), -1)  # Green circle for "right"

        print(pred)  

        cv.imshow('Center 96x96 Region', center_region)


        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
