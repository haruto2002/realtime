import os
from pathlib import Path

import cv2
import numpy as np

from processor.components import Detector, Tracker
from processor.worker import MotWorker


def run_image(
    image: np.ndarray, detectors: list[Detector], trackers: list[Tracker]
) -> np.ndarray:
    # start_ts = time.perf_counter()
    frame, detected_ts, tracked_ts, drawn_ts = MotWorker.process_frame(
        image, detectors, trackers
    )
    # end_ts = time.perf_counter()
    # print(
    #     f"total: {ms(end_ts - start_ts)}, detect: {ms(detected_ts - start_ts)}, track: {ms(tracked_ts - detected_ts)}, drawn: {ms(drawn_ts - tracked_ts)}, write: {ms(end_ts - drawn_ts)}"
    # )
    return frame


def run_video(
    path2video: str,
    detectors: list[Detector],
    trackers: list[Tracker],
    output_path: str,
    fps: int,
    scale: float,
):
    video_capture = cv2.VideoCapture(path2video)
    raw_fps = video_capture.get(cv2.CAP_PROP_FPS)

    if raw_fps < fps:
        step = 1
        output_fps = raw_fps
    else:
        step = int(round(raw_fps / fps))
        output_fps = raw_fps / step

    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)

    video_writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height)
    )

    frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % step == 0:
            frame = cv2.resize(frame, (width, height))
            frame = run_image(frame, detectors, trackers)
            video_writer.write(frame)

        frame_count += 1

    video_capture.release()
    video_writer.release()

    print("-" * 100)
    print(
        f"Video saved to {output_path} with {output_fps:.2f} fps and {width}x{height} resolution"
    )
    print("-" * 100)


def ms(time: float) -> str:
    return f"{time * 1000:.2f} ms"


def main(fps: int, scale: float, sample_dir: str, names: list[str], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    path2detector_cfg = [
        "video_conf/detector/p2pnet/p2pnet.yaml",
        # "video_conf/detector/yolo26/yolo26.yaml",
    ]
    path2tracker_cfg = [
        f"video_conf/tracker/point_bytetrack/point_bytetrack_{fps:02d}.yaml",
        # f"video_conf/tracker/bytetrack/bytetrack_{fps:02d}.yaml",
    ]

    detectors = [
        Detector(Path(path2detector_cfg)) for path2detector_cfg in path2detector_cfg
    ]
    trackers = [
        Tracker(Path(path2tracker_cfg)) for path2tracker_cfg in path2tracker_cfg
    ]

    for name in names:
        path2video = f"{sample_dir}/{name}.mp4"
        output_path = f"{save_dir}/{name}_{fps:02d}fps_{int(scale * 1000):03d}scale.mp4"
        assert os.path.exists(path2video), f"Video file not found: {path2video}"
        run_video(path2video, detectors, trackers, output_path, fps, scale)


if __name__ == "__main__":
    fps_list = [5, 10, 15, 30]
    scale_list = [1.0]
    sample_dir = "samples/vid/videos"
    names = ["pasifico_a", "pasifico_b"]
    save_dir = "vis/videos/exp0512_person"

    # fps_list = [5, 10, 15, 30]
    # scale_list = [0.125, 0.25, 0.5]
    # sample_dir = "samples/vid/videos"
    # names = ["pasifico_a", "pasifico_b"]
    # save_dir = "vis/videos/exp0512_car"

    for fps in fps_list:
        for scale in scale_list:
            main(fps, scale, sample_dir, names, save_dir)

    # main(
    #     fps=5,
    #     scale=0.5,
    #     sample_dir="samples/vid/videos",
    #     names=["sibuya_b"],
    #     save_dir="vis/videos/exp0512_v2",
    # )
