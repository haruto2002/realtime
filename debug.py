import glob
import json
import os
from pathlib import Path

import cv2

from processor.components import Detector, Tracker
from processor.worker import MotWorker


def run_video(
    path2video: str, save_dir: str, detectors: list[Detector], trackers: list[Tracker]
):
    video_capture = cv2.VideoCapture(path2video)
    frame_id = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_id += 1
        frame, detected_ts, tracked_ts, drawn_ts, dets, tracks = (
            MotWorker.process_frame(frame, detectors, trackers)
        )
        objects = MotWorker.convert_tracks_to_objects(tracks)
        payload = {
            "camera_id": None,
            "pc_id": None,
            "timestamp": None,
            "frame_id": frame_id,
            "objects": objects,
        }
        save_path = os.path.join(save_dir, f"{frame_id:06d}.json")
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=4)
        # cv2.imwrite(os.path.join(save_dir, f"{frame_id:06d}.jpg"), frame)


def main():
    video_dir = "samples/vid/yokohama_2020508"
    path2video_list = sorted(glob.glob(os.path.join(video_dir, "*.MOV")))

    root_save_dir = "tmp_results/yokohama_2020508"
    os.makedirs(root_save_dir, exist_ok=True)

    path2detector_cfg = [
        "video_conf/detector/p2pnet/p2pnet.yaml",
        "video_conf/detector/yolo26/yolo26.yaml",
    ]
    path2tracker_cfg = [
        "video_conf/tracker/point_bytetrack/point_bytetrack_30.yaml",
        "video_conf/tracker/bytetrack/bytetrack_30.yaml",
    ]

    detectors = [
        Detector(Path(path2detector_cfg)) for path2detector_cfg in path2detector_cfg
    ]
    trackers = [
        Tracker(Path(path2tracker_cfg)) for path2tracker_cfg in path2tracker_cfg
    ]

    path2video_list = ["samples/vid/yokohama_2020508/kokusaibashi.MOV"]
    for path2video in path2video_list:
        place_name = path2video.split("/")[-1].split(".")[0]
        save_dir = os.path.join(root_save_dir, f"{place_name}")
        os.makedirs(save_dir, exist_ok=True)
        run_video(path2video, save_dir, detectors, trackers)


if __name__ == "__main__":
    main()
