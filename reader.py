import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from detector import Detector


@dataclass
class ReaderStats:
    read_latest_frame_id: int = 0
    read_frames: int = 0
    read_fail: int = 0
    restarts: int = 0

    det_latest_frame_id: int = 0
    det_frames: int = 0
    det_time: float = 0.0

    skipped_frames: int = 0
    process_time: float = 0.0
    display_time: float = 0.0


class FFmpegRTSPReader:
    def __init__(
        self,
        rtsp_url: str,
        size: Tuple[int, int],
        transport: str = "udp",
        ffmpeg_path: str = "ffmpeg",
        log_every_sec: float = 1.0,
        reconnect_backoff_sec: float = 0.5,
    ):
        self.rtsp_url = rtsp_url
        self.w, self.h = size
        self.transport = transport
        self.ffmpeg_path = ffmpeg_path
        self.log_every_sec = log_every_sec
        self.reconnect_backoff_sec = reconnect_backoff_sec

        self.frame_bytes = self.w * self.h * 3  # bgr24

        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None

        # latest（最新フレーム1枚）
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._latest_seq: int = 0

        # detection（最新結果1つ）
        self._det_lock = threading.Lock()
        self._frame: np.ndarray = np.zeros((self.w, self.h, 3), np.uint8)
        self._input_frame_read_ts: float = 0.0
        self._det_result: Optional[np.ndarray] = None
        self._det_seq: int = 0
        self._det_time: float = 0.0
        self._det_ts: float = 0.0
        self._det_thread: Optional[threading.Thread] = None

        self.stats = ReaderStats()

    # -------- public --------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._terminate_ffmpeg()

    def get_latest(self) -> Tuple[Optional[np.ndarray], int, float]:
        with self._lock:
            if self._latest_frame is None:
                return None, self._latest_seq, self._latest_ts
            return self._latest_frame.copy(), self._latest_seq, self._latest_ts

    # ★ Detector を組み込むためのAPI
    def start_detector(
        self,
        detector: Detector,
        infer_every_n: int = 1,
    ) -> None:
        if self._det_thread and self._det_thread.is_alive():
            return

        def _loop():
            last_seq = 0
            while not self._stop.is_set():
                frame, seq, frame_read_ts = self.get_latest()
                if frame is None or seq == last_seq:
                    time.sleep(0.002)
                    continue

                # 推論頻度（任意）
                if infer_every_n > 1 and (seq % infer_every_n != 0):
                    last_seq = seq
                    continue

                last_seq = seq

                t0 = time.perf_counter()
                res = detector(frame)
                t1 = time.perf_counter()

                frame = detector.display_result(frame, res)

                with self._det_lock:
                    self._frame = frame
                    self._input_frame_read_ts = frame_read_ts
                    self._det_result = res
                    self._det_seq = seq
                    self._det_time = t1 - t0
                    self._det_ts = time.perf_counter()

                    self.stats.det_latest_frame_id = self._det_seq
                    self.stats.det_frames += 1
                    self.stats.det_time = self._det_time

        self._det_thread = threading.Thread(target=_loop, daemon=True)
        self._det_thread.start()

    def get_latest_detection(
        self,
    ) -> Tuple[Optional[np.ndarray], float, Optional[np.ndarray], int, float]:
        with self._det_lock:
            return (
                self._frame,
                self._input_frame_read_ts,
                self._det_result,
                self._det_seq,
                self._det_time,
            )

    # -------- internal --------

    def _build_ffmpeg_cmd(self) -> list:
        return [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-rtsp_transport",
            self.transport,
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-probesize",
            "32",
            "-analyzeduration",
            "0",
            "-i",
            self.rtsp_url,
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-vf",
            f"scale={self.w}:{self.h}",
            "pipe:1",
        ]

    def _spawn_ffmpeg(self) -> None:
        self._terminate_ffmpeg()
        cmd = self._build_ffmpeg_cmd()
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8,
        )
        self.stats.restarts += 1

    def _terminate_ffmpeg(self) -> None:
        proc = self._proc
        self._proc = None
        if not proc:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _reader_loop(self) -> None:
        while not self._stop.is_set():
            if self._proc is None:
                self._spawn_ffmpeg()
                time.sleep(0.05)

            proc = self._proc
            if proc is None:
                continue

            if proc.poll() is not None:
                self._print_ffmpeg_error(proc)
                self._spawn_ffmpeg()
                time.sleep(self.reconnect_backoff_sec)
                continue

            try:
                raw = proc.stdout.read(self.frame_bytes) if proc.stdout else b""
            except Exception:
                raw = b""

            if not raw or len(raw) < self.frame_bytes:
                self.stats.read_fail += 1
                self._spawn_ffmpeg()
                time.sleep(self.reconnect_backoff_sec)
                continue

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.h, self.w, 3))
            now = time.perf_counter()

            with self._lock:
                self._latest_frame = frame
                self._latest_ts = now
                self._latest_seq += 1
                self.stats.read_latest_frame_id = self._latest_seq
                self.stats.read_frames += 1

        self._terminate_ffmpeg()

    def _print_ffmpeg_error(self, proc: subprocess.Popen) -> None:
        try:
            if proc.stderr:
                err = proc.stderr.read().decode("utf-8", errors="ignore").strip()
                if err:
                    print("[FFMPEG-ERR]", err)
        except Exception:
            pass

    def log_status(self) -> None:
        s = self.stats
        print(
            f"[STAT] shown={s.det_latest_frame_id} (read={s.read_latest_frame_id}) "
            f"process_num={s.det_frames} (read={s.read_frames}) "
            f"display={s.display_time * 1000:.2f}ms full_process={s.process_time * 1000:.2f}ms det={s.det_time * 1000:.2f}ms "
            f"skipped={s.skipped_frames} "
            f"read_fail={s.read_fail} restarts={s.restarts} "
        )
