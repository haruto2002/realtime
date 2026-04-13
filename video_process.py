import time

import cv2
import numpy as np

from components import Detector, Tracker


def main():
    path2video = "palace.mp4"
    detector = Detector("pipeline/config/deimv2.yaml")
    tracker = Tracker("pipeline/config/bytetrack.yaml")
    video_capture = cv2.VideoCapture(path2video)

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(
        "output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    while True:
        print("frame", video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        t0 = time.perf_counter()
        ret, frame = video_capture.read()
        t1 = time.perf_counter()
        print("read time", t1 - t0)
        if not ret:
            break
        t2 = time.perf_counter()
        dets = detector.infer(frame)
        t3 = time.perf_counter()
        print("infer time", t3 - t2)
        t4 = time.perf_counter()
        tracks = tracker.update(convert_to_tracker_inputs(dets))
        t5 = time.perf_counter()
        print("update time", t5 - t4)
        frame = tracker.draw(frame, tracks)
        t6 = time.perf_counter()
        print("draw time", t6 - t5)
        video_writer.write(frame)
    video_capture.release()
    video_writer.release()


def convert_to_tracker_inputs(dets):
    dets = [dets[:, :4], dets[:, 5:6]]
    dets = np.concatenate(dets, axis=1)
    return dets


if __name__ == "__main__":
    main()
