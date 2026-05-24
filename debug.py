import time

import cv2
from ultralytics import YOLO


def main(size: str):
    model = YOLO(f"weights/yolo26/yolo26{size}.pt")
    path2image = "hd_demo.jpg"
    img = cv2.imread(path2image)

    device = "cpu"

    n = 50

    start_time = time.perf_counter()
    for _ in range(n):
        result = model(
            path2image,
            imgsz=1920,
            conf=0.2,
            device=device,
            classes=[0],
            rect=True,
            verbose=False,
        )[0]
    end_time = time.perf_counter()
    print(f"size: {size}, time: {(end_time - start_time) / n * 1000:.2f} ms")
    # result.save(f"size_{size}.jpg", labels=False, conf=False)


if __name__ == "__main__":
    main("n")
    main("s")
    main("m")
    main("l")
    main("x")
