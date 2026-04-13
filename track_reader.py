import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ReaderStats:
    read_latest_frame_id: int = 0
    read_frames: int = 0
    read_fail: int = 0
    restarts: int = 0
    latency: float = 0.0


class FFmpegRTSPReader:
    def __init__(
        self,
        rtsp_url: str,
        size: Tuple[int, int],
        transport: str = "udp",
        ffmpeg_path: str = "ffmpeg",
        reconnect_backoff_sec: float = 0.5,
        output_fps: Optional[float] = None,
        frame_queue_maxsize: int = 0,
    ):
        self.rtsp_url = rtsp_url
        self.w, self.h = size
        self.transport = transport
        self.ffmpeg_path = ffmpeg_path
        self.reconnect_backoff_sec = reconnect_backoff_sec
        self.output_fps = output_fps
        self.frame_queue_maxsize = frame_queue_maxsize

        self.frame_bytes = self.w * self.h * 3  # bgr24

        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None

        # latest（最新フレーム1枚）
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0
        self._latest_seq: int = 0

        # 順次取得（フレームを間引かずに渡す。キューが満杯のときは読み取りスレッドがブロック）
        self._frame_queue: Optional[queue.Queue[Tuple[np.ndarray, int, float]]] = None
        if frame_queue_maxsize > 0:
            self._frame_queue = queue.Queue(maxsize=frame_queue_maxsize)

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

    def get_next(
        self, block: bool = True, timeout: Optional[float] = None
    ) -> Tuple[Optional[np.ndarray], int, float]:
        """順番どおりに1フレーム取得。frame_queue_maxsize>0 が必要。

        block=True でキューが空のとき待つ。消費が遅いと読み取り側がブロックし、フレームを落とさない。
        """
        if self._frame_queue is None:
            raise RuntimeError(
                "get_next を使うには frame_queue_maxsize を 1 以上にしてください"
            )
        try:
            frame, seq, ts = self._frame_queue.get(block=block, timeout=timeout)
            return frame, seq, ts
        except queue.Empty:
            return None, self._latest_seq, self._latest_ts

    # -------- internal --------

    def _build_ffmpeg_cmd(self) -> list:
        vf = f"scale={self.w}:{self.h}"
        if self.output_fps is not None and self.output_fps > 0:
            vf = f"{vf},fps={self.output_fps}"
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
            vf,
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
                seq = self._latest_seq

            if self._frame_queue is not None:
                item = (frame.copy(), seq, now)
                while not self._stop.is_set():
                    try:
                        self._frame_queue.put(item, timeout=0.2)
                        break
                    except queue.Full:
                        continue

        self._terminate_ffmpeg()

    def _print_ffmpeg_error(self, proc: subprocess.Popen) -> None:
        try:
            if proc.stderr:
                err = proc.stderr.read().decode("utf-8", errors="ignore").strip()
                if err:
                    print("[FFMPEG-ERR]", err)
        except Exception:
            pass
