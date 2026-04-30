import threading
import time

import numpy as np

from processor.components import Detector, Tracker
from processor.displayer import Displayer
from processor.reader import FFmpegRTSPReader
from processor.timer import TimeCounter


class MotWorker:
    def __init__(
        self,
        time_counter: TimeCounter,
        reader: FFmpegRTSPReader,
        displayer: Displayer,
        detectors: list[Detector],
        trackers: list[Tracker],
    ):
        self.time_counter = time_counter
        self.reader = reader
        self.displayer = displayer
        self.detectors = detectors
        self.trackers = trackers
        assert len(self.detectors) == len(self.trackers), (
            "The number of detectors and trackers must be the same"
        )

        self._pre_frame_set_ts: float = 0.0
        self._pre_submit_ts: float = 0.0

    def start(self):
        self.worker_thread = threading.Thread(
            target=self.pipeline_worker, name="MotPipeline", daemon=False
        )
        self.worker_thread.start()

    def stop(self):
        self.worker_thread.join(timeout=30.0)
        self.reader.stop()

    def pipeline_worker(self):
        while not self.displayer.stopped:
            start_ts = time.perf_counter()
            frame, seq, _ts = self.reader.get_next()
            frame_set_ts = time.perf_counter()
            if self._pre_frame_set_ts != 0.0:
                frame_set_interval = frame_set_ts - self._pre_frame_set_ts
                frame_set_fps = (
                    (1.0 / frame_set_interval) if frame_set_interval > 0 else None
                )
            else:
                frame_set_fps = None
            self._pre_frame_set_ts = frame_set_ts

            if self.displayer.stopped:
                break
            if frame is None:
                continue

            frame, detected_ts, tracked_ts, drawn_ts = self.process_frame(
                frame, self.detectors, self.trackers
            )

            self.displayer.submit(frame, seq)
            submitted_ts = time.perf_counter()
            if self._pre_submit_ts != 0.0:
                submit_interval = submitted_ts - self._pre_submit_ts
                submit_fps = (1.0 / submit_interval) if submit_interval > 0 else None
            else:
                submit_fps = None
            self._pre_submit_ts = submitted_ts

            end_ts = time.perf_counter()

            ts_logger = self.time_counter.get(seq)
            ts_logger.worker.start = start_ts
            ts_logger.worker.frame_set = frame_set_ts
            ts_logger.worker.detected = detected_ts
            ts_logger.worker.tracked = tracked_ts
            ts_logger.worker.drawn = drawn_ts
            ts_logger.worker.submitted = submitted_ts
            ts_logger.worker.end = end_ts
            ts_logger.frame_set_fps = frame_set_fps
            ts_logger.submit_fps = submit_fps

    @staticmethod
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


class PoseWorker:
    def __init__(
        self,
        time_counter: TimeCounter,
        reader: FFmpegRTSPReader,
        displayer: Displayer,
    ):
        self.time_counter = time_counter
        self.reader = reader
        self.displayer = displayer
        self.model = self.build_model()
        self._pre_frame_set_ts: float = 0.0
        self._pre_submit_ts: float = 0.0

    def build_model(self):
        from ultralytics import YOLO

        model = YOLO("weights/yolo26/yolo26x-pose.pt")
        model.to("cuda:0")
        model.eval()
        return model

    def start(self):
        self.worker_thread = threading.Thread(
            target=self.pipeline_worker, name="PosePipeline", daemon=False
        )
        self.worker_thread.start()

    def stop(self):
        self.worker_thread.join(timeout=30.0)
        self.reader.stop()

    def pipeline_worker(self):
        while not self.displayer.stopped:
            start_ts = time.perf_counter()
            frame, seq, _ts = self.reader.get_next()
            if frame is None:
                continue
            frame_set_ts = time.perf_counter()
            if self._pre_frame_set_ts != 0.0:
                frame_set_interval = frame_set_ts - self._pre_frame_set_ts
                frame_set_fps = (
                    (1.0 / frame_set_interval) if frame_set_interval > 0 else None
                )
            else:
                frame_set_fps = None
            self._pre_frame_set_ts = frame_set_ts

            assert seq in self.time_counter.log, (
                f"seq {seq} not found in time_counter.log"
            )

            result = self.model(frame, imgsz=1920, verbose=False)[0]
            detected_ts = time.perf_counter()

            frame = result.plot()
            drawn_ts = time.perf_counter()

            self.displayer.submit(frame, seq)
            submitted_ts = time.perf_counter()
            if self._pre_submit_ts != 0.0:
                submit_interval = submitted_ts - self._pre_submit_ts
                submit_fps = (1.0 / submit_interval) if submit_interval > 0 else None
            else:
                submit_fps = None
            self._pre_submit_ts = submitted_ts

            end_ts = time.perf_counter()

            ts_logger = self.time_counter.get(seq)
            ts_logger.worker.start = start_ts
            ts_logger.worker.frame_set = frame_set_ts
            ts_logger.worker.detected = detected_ts
            ts_logger.worker.drawn = drawn_ts
            ts_logger.worker.submitted = submitted_ts
            ts_logger.worker.end = end_ts
            ts_logger.frame_set_fps = frame_set_fps
            ts_logger.submit_fps = submit_fps
