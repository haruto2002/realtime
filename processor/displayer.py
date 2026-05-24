import queue
import threading
import time
from typing import Tuple

import cv2
import numpy as np

from processor.timer import TimeCounter


class Displayer:
    def __init__(
        self,
        time_counter: TimeCounter,
        window_name: str = "frame",
        maxsize: int = 1,
        freq_to_report_fps: int = 100,
        flush_on_queue_full: bool = True,
        report_single: bool = True,
        report_avg: bool = True,
    ):
        self.time_counter = time_counter
        self.window_name = window_name
        self.queue: queue.Queue[Tuple[np.ndarray, int]] = queue.Queue(maxsize=maxsize)
        self.freq_to_report_fps = freq_to_report_fps
        self.flush_on_queue_full = flush_on_queue_full
        self.report_single = report_single
        self.report_avg = report_avg

        self._stop = threading.Event()

        self.queue_full_flushes = 0
        self.frames_dropped_on_queue_flush = 0
        self._prev_display_ts = 0.0

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def request_stop(self) -> None:
        self._stop.set()

    def _flush_display_queue(self) -> int:
        n = 0
        while True:
            try:
                self.queue.get_nowait()
                n += 1
            except queue.Empty:
                break
        return n

    def submit(self, frame, seq):
        if self._stop.is_set():
            return

        arrived_ts = time.perf_counter()
        self.time_counter.log[seq].displayer.arrived = arrived_ts
        item = (frame, seq)

        # キューが空いてれば入れる
        try:
            self.queue.put_nowait(item)
            return
        except queue.Full:
            pass

        # キューが満杯のとき、
        # flush_on_queue_full が False ならブロックして待つ
        if not self.flush_on_queue_full:
            while not self._stop.is_set():
                try:
                    self.queue.put(item, timeout=0.2)
                    break
                except queue.Full:
                    continue
            return

        # flush_on_queue_full が True ならキューを空にしてから入れる
        print(f"Flush display queue ({self.queue_full_flushes + 1})")
        dropped = self._flush_display_queue()
        self.queue_full_flushes += 1
        self.frames_dropped_on_queue_flush += dropped
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            raise RuntimeError(
                "Failed to put item into display queue even after flushing"
            )

    def run_loop(self):
        try:
            while not self._stop.is_set():
                start_ts = time.perf_counter()
                try:
                    frame, seq = self.queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                if self.time_counter.log.get(seq) is None:
                    raise RuntimeError(f"seq {seq} not found in time_counter.log")

                frame_set_ts = time.perf_counter()

                cv2.imshow(self.window_name, frame)
                key = cv2.waitKey(1) & 0xFF

                displayed_ts = time.perf_counter()

                if self._prev_display_ts != 0.0:
                    display_interval = displayed_ts - self._prev_display_ts
                    fps = (1.0 / display_interval) if display_interval > 0 else None
                else:
                    fps = None
                self._prev_display_ts = displayed_ts

                end_ts = time.perf_counter()

                ts_logger = self.time_counter.get(seq)
                ts_logger.displayer.start = start_ts
                ts_logger.displayer.frame_set = frame_set_ts
                ts_logger.displayer.displayed = displayed_ts
                ts_logger.displayer.end = end_ts
                ts_logger.display_fps = fps
                if self.report_single:
                    self.time_counter.report_single(seq)

                if self.report_avg and seq % self.freq_to_report_fps == 0:
                    self.time_counter.report_avg(self.freq_to_report_fps)

                if key == ord("q"):
                    self._stop.set()
                    break
        finally:
            cv2.destroyAllWindows()
