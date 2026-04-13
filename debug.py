"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import cv2

from pipeline.detector.deimv2 import DEIMv2Detector


def main():
    config = "modules/detector/DEIMv2/configs/deimv2/deimv2_dinov3_x_coco.yml"
    resume = "weights/deimv2/deimv2_dinov3_x_coco.pth"
    device = "cpu"
    threshold = 0.5
    detector = DEIMv2Detector(config, resume, device, threshold)
    image = cv2.imread("modules/detector/DEIMv2/example.jpg")
    result = detector.infer(image)
    image = detector.display_result(image, result)
    cv2.imwrite("result.jpg", image)
    # print(result[0].shape)
    # print(result[1].shape)
    # print(result[2].shape)


if __name__ == "__main__":
    main()
