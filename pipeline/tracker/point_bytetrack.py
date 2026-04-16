from dataclasses import dataclass

import cv2
import numpy as np

from modules.tracker.point_bytetracker.byte_tracker import PointBYTETracker


@dataclass
class PointByteTrackArgs:
    track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 10.0
    distance_metric: str = "euclidean"
    frame_rate: int = 30


class PointByteTrackTracker:
    def __init__(
        self,
        track_thresh: float,
        track_buffer: int,
        match_thresh: float,
        distance_metric: str,
        frame_rate: int,
    ):
        args = PointByteTrackArgs(
            track_thresh, track_buffer, match_thresh, distance_metric
        )
        self.tracker = PointBYTETracker(args, frame_rate=frame_rate)

    def update(self, dets: np.ndarray):
        """
        dets: [[x, y, score], ...]
        """
        return self.tracker.update(dets)

    def draw(self, image, tracks):
        point_size = 5
        for i, track in enumerate(tracks):
            x, y = track.point
            obj_id = track.track_id
            color = self.get_color(abs(obj_id))
            cv2.circle(image, (int(x), int(y)), point_size, color, -1)

        return image

    def get_color(self, idx):
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
        return color
