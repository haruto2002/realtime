import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from processor.components import Detector, Tracker


@dataclass
class AreaConfig:
    path2homography_matrix: Path
    points_dir: Path
    original_area_size: list[float]
    all_map_left_top_coor: list[float]
    scale: float
    homography_matrix: np.ndarray = field(init=False)

    def __post_init__(self):
        self.homography_matrix = np.loadtxt(self.path2homography_matrix)


class AllMapDisplayer:
    def __init__(
        self,
        path2all_map: Path,
        config_list: list[AreaConfig],
    ):
        self.path2all_map = path2all_map
        self.all_map_img = cv2.imread(path2all_map.as_posix())
        self.config_list = config_list

    def run(self, time: datetime):
        output_img = self.all_map_img.copy()
        for config in self.config_list:
            points, area = self.set_data(time, config)
            if points is None:
                continue
            output_img = self.display_on_all_map(points, area, output_img)
        output_img = self.add_time_text(output_img, time)
        return time, output_img

    def set_data(self, time, config):
        time_str = time.strftime("%Y-%m-%d_%H:%M:%S")
        path2points = list(config.points_dir.glob(f"*{time_str}.txt"))
        if len(path2points) == 0:
            return None, None
        points = np.loadtxt(path2points[0])
        if points.shape[1] == 3:
            points = points[points[:, 2] > 0.5][:, :2]

        points = self.project_points(points, config.homography_matrix)

        points_on_all_map = points * config.scale + np.array(
            config.all_map_left_top_coor
        )
        area = [
            config.all_map_left_top_coor[0],
            config.all_map_left_top_coor[1],
            config.all_map_left_top_coor[0]
            + config.original_area_size[0] * config.scale,
            config.all_map_left_top_coor[1]
            + config.original_area_size[1] * config.scale,
        ]
        return points_on_all_map, area

    def project_points(self, points, homography_matrix):
        points = cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography_matrix)
        return points.reshape(-1, 2)

    def display_on_all_map(self, points, area, all_map_img):
        for point in points:
            x, y = int(point[0]), int(point[1])
            cv2.circle(
                all_map_img,
                (x, y),
                radius=3,
                color=(0, 0, 255),
                thickness=-1,
            )
        top_left = (int(area[0]), int(area[1]))
        bottom_right = (int(area[2]), int(area[3]))
        cv2.rectangle(
            all_map_img,
            top_left,
            bottom_right,
            (0, 255, 0),
            5,
        )
        return all_map_img

    def add_time_text(self, all_map_img, time):
        time_str = time.strftime("%H:%M:%S")
        # 画面右上に白い四角、その上に黒の時刻表⽰（解像度対応）
        h, w = all_map_img.shape[:2]
        rect_width = int(w * 0.15)
        rect_height = int(h * 0.05)
        margin = int(h * 0.01)
        top_left = (w - rect_width - margin, margin)
        bottom_right = (w - margin, margin + rect_height)
        cv2.rectangle(
            all_map_img, top_left, bottom_right, (255, 255, 255), thickness=-1
        )
        font_scale = rect_height / 50
        thickness = max(1, int(rect_height / 30))
        text_size, baseline = cv2.getTextSize(
            time_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        text_x = top_left[0] + (rect_width - text_size[0]) // 2
        text_y = top_left[1] + (rect_height + text_size[1]) // 2
        cv2.putText(
            all_map_img,
            time_str,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )
        return all_map_img


def process_frame(
    frame: np.ndarray, detectors: list[Detector], trackers: list[Tracker]
):
    dets = [detector.infer(frame) for detector in detectors]
    detected_ts = time.perf_counter()

    tracks = [
        tracker.update(tracker.convert_to_tracker_inputs(det))
        for tracker, det in zip(trackers, dets)
    ]
    tracked_ts = time.perf_counter()

    [tracker.draw(frame, track) for tracker, track in zip(trackers, tracks)]
    drawn_ts = time.perf_counter()

    return frame, detected_ts, tracked_ts, drawn_ts


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
