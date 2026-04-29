import threading
import time

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
        detector: Detector,
        tracker: Tracker,
    ):
        self.time_counter = time_counter
        self.reader = reader
        self.displayer = displayer
        self.detector = detector
        self.tracker = tracker

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
            frame, seq, _ts = self.reader.get_next(timeout=0.2)
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

            assert seq in self.time_counter.log, (
                f"seq {seq} not found in time_counter.log"
            )

            dets = self.detector.infer(frame)
            detected_ts = time.perf_counter()

            tracks = self.tracker.update(self.tracker.convert_to_tracker_inputs(dets))
            tracked_ts = time.perf_counter()

            # frame = self.tracker.draw(frame, tracks)
            frame = self.detector.draw(frame, dets)
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
            self.time_counter.log[seq].main_processor.start = start_ts
            self.time_counter.log[seq].main_processor.frame_set = frame_set_ts
            self.time_counter.log[seq].main_processor.detected = detected_ts
            self.time_counter.log[seq].main_processor.tracked = tracked_ts
            self.time_counter.log[seq].main_processor.drawn = drawn_ts
            self.time_counter.log[seq].main_processor.submitted = submitted_ts
            self.time_counter.log[seq].main_processor.end = end_ts

            self.time_counter.log[seq].frame_set_fps = frame_set_fps
            self.time_counter.log[seq].submit_fps = submit_fps


class TwoModelMotWorker:
    def __init__(
        self,
        time_counter: TimeCounter,
        reader: FFmpegRTSPReader,
        displayer: Displayer,
        point_detector: Detector,
        box_detector: Detector,
        point_tracker: Tracker,
        box_tracker: Tracker,
    ):
        self.time_counter = time_counter
        self.reader = reader
        self.displayer = displayer
        self.point_detector = point_detector
        self.box_detector = box_detector
        self.point_tracker = point_tracker
        self.box_tracker = box_tracker

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
            frame, seq, _ts = self.reader.get_next(timeout=0.2)
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

            assert seq in self.time_counter.log, (
                f"seq {seq} not found in time_counter.log"
            )

            point_dets = self.point_detector.infer(frame)
            box_dets = self.box_detector.infer(frame)
            detected_ts = time.perf_counter()

            point_tracks = self.point_tracker.update(
                self.point_tracker.convert_to_tracker_inputs(point_dets)
            )
            box_tracks = self.box_tracker.update(
                self.box_tracker.convert_to_tracker_inputs(box_dets)
            )
            tracked_ts = time.perf_counter()

            frame = self.point_tracker.draw(frame, point_tracks)
            frame = self.box_tracker.draw(frame, box_tracks)
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
            self.time_counter.log[seq].main_processor.start = start_ts
            self.time_counter.log[seq].main_processor.frame_set = frame_set_ts
            self.time_counter.log[seq].main_processor.detected = detected_ts
            self.time_counter.log[seq].main_processor.tracked = tracked_ts
            self.time_counter.log[seq].main_processor.drawn = drawn_ts
            self.time_counter.log[seq].main_processor.submitted = submitted_ts
            self.time_counter.log[seq].main_processor.end = end_ts

            self.time_counter.log[seq].frame_set_fps = frame_set_fps
            self.time_counter.log[seq].submit_fps = submit_fps


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

        model = YOLO("yolo26x-pose.pt")
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
            frame, seq, _ts = self.reader.get_next(timeout=0.2)
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

            assert seq in self.time_counter.log, (
                f"seq {seq} not found in time_counter.log"
            )

            result = self.model(frame, imgsz=(1920, 1080))[0]
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
            self.time_counter.log[seq].main_processor.start = start_ts
            self.time_counter.log[seq].main_processor.frame_set = frame_set_ts
            self.time_counter.log[seq].main_processor.detected = detected_ts
            self.time_counter.log[seq].main_processor.drawn = drawn_ts
            self.time_counter.log[seq].main_processor.submitted = submitted_ts
            self.time_counter.log[seq].main_processor.end = end_ts

            self.time_counter.log[seq].frame_set_fps = frame_set_fps
            self.time_counter.log[seq].submit_fps = submit_fps
