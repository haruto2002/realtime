import argparse
import time
from pathlib import Path

import cv2
from hydra.utils import instantiate
from omegaconf import OmegaConf

from processor.components import Detector, Tracker


def main(
    path2detector_cfg: str, path2tracker_cfg: str, path2video: str, output_path: str
) -> None:

    detector: Detector = instantiate(OmegaConf.load(Path(path2detector_cfg)))
    print("Detector Setup Done")
    tracker: Tracker = instantiate(OmegaConf.load(Path(path2tracker_cfg)))
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path2detector_cfg", type=str, default="video_conf/detector.yaml"
    )
    parser.add_argument(
        "--path2tracker_cfg", type=str, default="video_conf/tracker.yaml"
    )
    parser.add_argument(
        "--path2video",
        type=str,
        default="hanabi.mp4",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="hanabi_output_deim.mp4",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # args = get_args()
    # path2detector_cfg = args.path2detector_cfg
    # path2tracker_cfg = args.path2tracker_cfg
    # path2video = args.path2video
    # output_path = args.output_path

    path2detector_cfg = "video_conf/detector.yaml"
    path2tracker_cfg = "video_conf/tracker.yaml"
    name = "newyork"
    path2video = f"videos/raw/{name}.mov"
    output_path = f"videos/output/{name}.mp4"
    main(path2detector_cfg, path2tracker_cfg, path2video, output_path)
