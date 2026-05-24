import os
import time
from pathlib import Path

import cv2

from processor.components import Detector, Tracker


def run_video(
    path2detector_cfg: str, path2tracker_cfg: str, path2video: str, output_path: str
) -> None:

    detector = Detector(Path(path2detector_cfg), devide_size=None)
    print("Detector Setup Done")
    tracker = Tracker(Path(path2tracker_cfg))
    print("Tracker Setup Done")

    pipeline(path2video, detector, tracker, output_path)


def pipeline(path2video: str, detector: Detector, tracker: Tracker, output_path: str):
    video_capture = cv2.VideoCapture(path2video)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    while True:
        start_ts = time.perf_counter()
        ret, frame = video_capture.read()
        if not ret:
            print("Video finished")
            break
        frame_set_ts = time.perf_counter()
        dets = detector.infer(frame)
        detected_ts = time.perf_counter()
        tracks = tracker.update(tracker.convert_to_tracker_inputs(dets))
        tracked_ts = time.perf_counter()
        frame = tracker.draw(frame, tracks)
        # frame = detector.draw(frame, dets)
        drawn_ts = time.perf_counter()
        video_writer.write(frame)
        end_ts = time.perf_counter()
        print(
            f"total: {ms(end_ts - start_ts)}, frame_set: {ms(frame_set_ts - start_ts)}, detect: {ms(detected_ts - frame_set_ts)}, track: {ms(tracked_ts - detected_ts)}, drawn: {ms(drawn_ts - tracked_ts)}, write: {ms(end_ts - drawn_ts)}"
        )
    video_capture.release()
    video_writer.release()


def ms(time: float) -> str:
    return f"{time * 1000:.2f} ms"


def main(model: str, name: str):
    if model == "yolo26":
        path2detector_cfg = "video_conf/detector/yolo26/yolo26.yaml"
        path2tracker_cfg = "video_conf/tracker/bytetrack/bytetrack.yaml"
    elif model == "deimv2":
        path2detector_cfg = "video_conf/detector/deimv2/deimv2.yaml"
        path2tracker_cfg = "video_conf/tracker/bytetrack/bytetrack.yaml"
    elif model == "p2pnet":
        path2detector_cfg = "video_conf/detector/p2pnet/p2pnet.yaml"
        path2tracker_cfg = "video_conf/tracker/point_bytetrack/point_bytetrack.yaml"
    else:
        raise ValueError(f"Invalid model: {model}")
    path2video = f"samples/vid/videos/{name}.mp4"
    output_path = f"vis/videos/{name}_{model}.mp4"
    assert os.path.exists(path2video), f"Video file not found: {path2video}"
    run_video(path2detector_cfg, path2tracker_cfg, path2video, output_path)


def main_all():
    os.makedirs("vis/videos", exist_ok=True)
    models = ["yolo26", "deimv2", "p2pnet"]
    names = ["newyork", "palace", "hanabi", "sibuya_a", "sibuya_b"]
    for model in models:
        for name in names:
            print(f"Running {model} {name}")
            main(model, name)


if __name__ == "__main__":
    main_all()
