import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from processor.timer import TimeCounter


@dataclass
class ReaderStats:
    read_latest_frame_id: int = 0
    read_frames: int = 0
    read_fail: int = 0
    restarts: int = 0
    latency: float = 0.0
    queue_full_flushes: int = 0
    frames_dropped_on_queue_flush: int = 0


class FFmpegRTSPReader:
    def __init__(
        self,
        time_counter: TimeCounter,
        rtsp_url: str,
        size: Tuple[int, int],
        transport: str = "udp",
        ffmpeg_path: str = "ffmpeg",
        reconnect_backoff_sec: float = 0.5,
        output_fps: Optional[float] = None,
        frame_queue_maxsize: int = 0,
        flush_on_queue_full: bool = True,
    ):
        """flush_on_queue_full: キュー満杯時に中身を捨ててから現在フレームを1枚だけ入れる。
        False のときは従来どおり put が空くまでブロック（バックプレッシャー）。
        """
        self.time_counter = time_counter
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

        # 順次取得。満杯のときはキューを空にしてから最新1枚だけ入れ直す（バックプレッシャーで詰まらない）
        self._frame_queue: Optional[queue.Queue[Tuple[np.ndarray, int, float]]] = None
        if frame_queue_maxsize > 0:
            self._frame_queue = queue.Queue(maxsize=frame_queue_maxsize)
        self.flush_on_queue_full = flush_on_queue_full

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

    def get_next(
        self, block: bool = True, timeout: Optional[float] = None
    ) -> Tuple[Optional[np.ndarray], int, float]:
        """順番どおりに1フレーム取得。frame_queue_maxsize>0 が必要。

        block=True でキューが空のとき待つ。消費が極端に遅いと読み取り側がキューをフラッシュするため、
        seq に飛び番が出ることがある。
        """
        if self._frame_queue is None:
            raise RuntimeError(
                "get_next を使うには frame_queue_maxsize を 1 以上にしてください"
            )
        try:
            frame, seq, ts = self._frame_queue.get(block=block, timeout=timeout)
            return frame, seq, ts
        except queue.Empty:
            with self._lock:
                seq, ts = self._latest_seq, self._latest_ts
            return None, seq, ts

    def drain_frame_queue(self) -> int:
        """キューに溜まったフレームをすべて捨てる。get_latest から get_next へ切り替える前や、本処理開始直前に呼ぶ。

        get_latest はキューを消費しないため、初期化中に古いフレームが最大 maxsize まで溜まる。
        本処理の最初の get_next が「古い先頭」にならないよう、ここで空にする。
        """
        if self._frame_queue is None:
            return 0
        return self._flush_frame_queue()

    def _flush_frame_queue(self) -> int:
        """キューを空にし、取り除いた要素数を返す（スレッドセーフな Queue のみを触る）。"""
        if self._frame_queue is None:
            return 0
        n = 0
        while True:
            try:
                self._frame_queue.get_nowait()
                n += 1
            except queue.Empty:
                break
        return n

    # -------- internal --------

    @staticmethod
    def _read_exact(stream, n: int) -> bytes:
        """パイプからちょうど n バイト読む（複数回の read にまたがる場合も連結）。

        1回の read(n) は環境によって n 未満で返ることがある。欠けたまま reshape すると
        フレーム境界がずれ、以降すべて矩形状の色破損に見えることがある。
        """
        parts: list[bytes] = []
        remaining = n
        while remaining > 0:
            chunk = stream.read(remaining)
            if not chunk:
                break
            parts.append(chunk)
            remaining -= len(chunk)
        return b"".join(parts)

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
            start_ts = time.perf_counter()
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
                raw = (
                    self._read_exact(proc.stdout, self.frame_bytes)
                    if proc.stdout
                    else b""
                )
            except Exception:
                raw = b""

            if len(raw) < self.frame_bytes:
                self.stats.read_fail += 1
                self._spawn_ffmpeg()
                time.sleep(self.reconnect_backoff_sec)
                continue

            arrived_ts = time.perf_counter()
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.h, self.w, 3))
            loaded_ts = time.perf_counter()

            with self._lock:
                self._latest_frame = frame
                self._latest_ts = loaded_ts
                self._latest_seq += 1
                self.stats.read_latest_frame_id = self._latest_seq
                self.stats.read_frames += 1
                seq = self._latest_seq

            if self._frame_queue is not None:
                item = (frame.copy(), seq, loaded_ts)
                try:
                    self._frame_queue.put_nowait(item)
                except queue.Full:
                    if self.flush_on_queue_full:
                        print(
                            f"Flush frame queue ({self.stats.queue_full_flushes + 1})"
                        )
                        dropped = self._flush_frame_queue()
                        self.stats.queue_full_flushes += 1
                        self.stats.frames_dropped_on_queue_flush += dropped
                        try:
                            self._frame_queue.put_nowait(item)
                        except queue.Full:
                            self._frame_queue.put(item, block=True, timeout=1.0)
                    else:
                        while not self._stop.is_set():
                            try:
                                self._frame_queue.put(item, timeout=0.2)
                                break
                            except queue.Full:
                                continue
            end_ts = time.perf_counter()

            ts_logger = self.time_counter.add(seq)
            ts_logger.frame_reader.start = start_ts
            ts_logger.frame_reader.arrived = arrived_ts
            ts_logger.frame_reader.loaded = loaded_ts
            ts_logger.frame_reader.end = end_ts

        self._terminate_ffmpeg()

    def _put_frame_queue(self, item: tuple[np.ndarray, int, float]) -> None:
        # キューを使用しない場合
        if self._frame_queue is None:
            return

        # キューが空いてれば入れる
        try:
            self._frame_queue.put_nowait(item)
            return
        except queue.Full:
            pass

        # キューが満杯のとき、
        # flush_on_queue_full が False ならブロックして待つ
        if not self.flush_on_queue_full:
            while not self._stop.is_set():
                try:
                    self._frame_queue.put(item, timeout=0.2)
                    return
                except queue.Full:
                    continue
            return

        # flush_on_queue_full が True ならキューを空にしてから入れる
        print(f"Flush frame queue ({self.stats.queue_full_flushes + 1})")
        dropped = self._flush_frame_queue()
        self.stats.queue_full_flushes += 1
        self.stats.frames_dropped_on_queue_flush += dropped

        try:
            self._frame_queue.put_nowait(item)
        except queue.Full:
            self.stats.frames_dropped_on_queue_flush += 1

    def _print_ffmpeg_error(self, proc: subprocess.Popen) -> None:
        try:
            if proc.stderr:
                err = proc.stderr.read().decode("utf-8", errors="ignore").strip()
                if err:
                    print("[FFMPEG-ERR]", err)
        except Exception:
            pass
