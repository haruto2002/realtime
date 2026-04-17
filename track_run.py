import time

import cv2

from components import Detector, Tracker
from track_reader import FFmpegRTSPReader

HOST = "member"
IP = "192.168.0.10"
PORT = 554
PW = "AIST-rwdc"
RTSP_URL = f"rtsp://{HOST}:{PW}@{IP}:{PORT}/ONVIF/MediaInput?profile=def_profile1"
W, H = 1920, 1080

# この枚数処理するごとに、かかった合計時間から平均 FPS を表示
FPS_REPORT_EVERY_FRAMES = 10


def main():
    reader = FFmpegRTSPReader(
        rtsp_url=RTSP_URL,
        size=(W, H),
        transport="udp",
        output_fps=15,
        frame_queue_maxsize=100,
        reconnect_backoff_sec=0.5,
        flush_on_queue_full=True,
    )
    reader.start()
    print("Reader Started")

    path2det_cfg = "pipeline/config/deimv2.yaml"
    path2track_cfg = "pipeline/config/bytetrack.yaml"

    # path2det_cfg = "pipeline/config/p2pnet.yaml"
    # path2track_cfg = "pipeline/config/point_bytetrack.yaml"

    detector = Detector(path2det_cfg)
    print("Detector Setup Done")
    tracker = Tracker(path2track_cfg)
    print("Tracker Setup Done")

    fps_window_start = time.perf_counter()
    fps_window_frames = 0
    while True:
        t0 = time.perf_counter()
        frame, seq, ts = reader.get_next(timeout=0.2)
        t1 = time.perf_counter()
        read_latency = t1 - t0
        read_wait_latency = t1 - ts
        if frame is None:
            continue

        # dets = detector.infer(frame)
        dets = detector.infer_split(frame, 640)
        t2 = time.perf_counter()
        det_latency = t2 - t1

        tracks = tracker.update(tracker.convert_to_tracker_inputs(dets))
        t3 = time.perf_counter()
        track_latency = t3 - t2

        frame = tracker.draw(frame, tracks)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        t4 = time.perf_counter()
        display_latency = t4 - t3

        total_latency = t4 - ts

        p_time = t4 - t0

        print(
            f"latency: {int(total_latency * 1000)}ms -- frame_wait: {int(read_wait_latency * 1000)}ms, process: {int(p_time * 1000)}ms "
            f"(read: {int(read_latency * 1000)}ms, det: {int(det_latency * 1000)}ms, track: {int(track_latency * 1000)}ms, display: {int(display_latency * 1000)}ms)"
        )
        # print(reader.stats)

        fps_window_frames += 1
        if fps_window_frames >= FPS_REPORT_EVERY_FRAMES:
            elapsed = time.perf_counter() - fps_window_start
            fps = fps_window_frames / elapsed
            print(f"FPS: {fps:.2f}")
            fps_window_start = time.perf_counter()
            fps_window_frames = 0
    cv2.destroyAllWindows()
    reader.stop()


if __name__ == "__main__":
    main()
