import time
from pathlib import Path

import cv2
import numpy as np
from detector import Detector

from reader import FFmpegRTSPReader

HOST = "member"
IP = "192.168.0.10"
PORT = 554
PW = "AIST-rwdc"
RTSP_URL = f"rtsp://{HOST}:{PW}@{IP}:{PORT}/ONVIF/MediaInput?profile=def_profile1"
W, H = 1920, 1080


def main():
    reader = FFmpegRTSPReader(
        rtsp_url=RTSP_URL,
        size=(W, H),
        transport="udp",
        log_every_sec=1.0,
        reconnect_backoff_sec=0.5,
    )
    reader.start()
    print("Reader Started")

    # cfg_path = Path("pipeline/config/p2pnet.yaml")
    cfg_path = Path("pipeline/config/deimv2.yaml")
    detector = Detector(cfg_path)
    reader.start_detector(detector, infer_every_n=1)
    print("Detector Started")

    # GUIウォームアップ
    cv2.imshow("frame", np.zeros((H, W, 3), np.uint8))
    cv2.waitKey(1)

    last_shown_seq = 0

    n = 0
    start_iter = 500

    try:
        while True:
            n += 1
            if n == start_iter:
                fps_time = time.perf_counter()
                fps_counter = 0

            start_ts = time.perf_counter()
            # 推論結果を取得
            frame, read_ts, det, det_seq, det_time = reader.get_latest_detection()

            if det_seq == last_shown_seq:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                time.sleep(0.01)
                continue

            # スキップ検出
            if last_shown_seq != 0:
                skipped = det_seq - last_shown_seq - 1
                if skipped > 0:
                    reader.stats.skipped_frames += skipped
            last_shown_seq = det_seq

            if n >= start_iter:
                fps_counter += 1
                now = time.perf_counter()
                elapsed = now - fps_time
                fps = fps_counter / elapsed
                reader.stats.fps = fps
                cv2.putText(
                    frame,
                    f"{fps}FPS",
                    (1200, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            end_ts = time.perf_counter()
            latency = end_ts - read_ts
            display_time = end_ts - start_ts
            reader.stats.latency = latency
            reader.stats.display_time = display_time

            reader.log_status()
    finally:
        reader.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    # print(RTSP_URL)
