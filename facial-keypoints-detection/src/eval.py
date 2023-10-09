import torch
import torch.nn as nn

import numpy as np

import cv2 as cv

if __name__ == "__main__":

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to find video capture device.")
        exit(1)

    while True:
        go, frame = cap.read()
        if not go:
            print("Failed to retrieve frame")
            break
            
        gray = cv.cvtColor(frame, cv.COLOR_BAYER_BGGR2GRAY) 
        cv.imshow('frame', gray)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
