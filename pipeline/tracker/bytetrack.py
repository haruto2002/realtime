from dataclasses import dataclass

import numpy as np

from modules.tracker.bytetrack.byte_tracker import BYTETracker


@dataclass
class ByteTrackArgs:
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    mot20: bool = False


class ByteTrackTracker:
    def __init__(
        self,
        track_thresh: float,
        track_buffer: int,
        match_thresh: float,
        mot20: bool,
        frame_rate: int,
    ):
        args = ByteTrackArgs(track_thresh, track_buffer, match_thresh, mot20)
        self.tracker = BYTETracker(args, frame_rate=frame_rate)

    def update(self, dets: np.ndarray):
        """
        dets: [[left, top, right, bottom, score], ...]
        """
        return self.tracker.update(dets)
