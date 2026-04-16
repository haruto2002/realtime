import time

import cv2

from components import Detector, Tracker

HOST = "member"
IP = "192.168.0.10"
PORT = 554
PW = "AIST-rwdc"
RTSP_URL = f"rtsp://{HOST}:{PW}@{IP}:{PORT}/ONVIF/MediaInput?profile=def_profile1"
W, H = 1920, 1080

# この枚数処理するごとに、かかった合計時間から平均 FPS を表示
FPS_REPORT_EVERY_FRAMES = 10


def main():
    video_capture = cv2.VideoCapture(RTSP_URL)
    print("Video Capture Started")

    # detector = Detector("pipeline/config/p2pnet.yaml")
    # print("Detector Started")
    # tracker = Tracker("pipeline/config/point_bytetrack.yaml")
    # print("Tracker Started")

    detector = Detector("pipeline/config/deimv2.yaml")
    print("Detector Started")
    tracker = Tracker("pipeline/config/bytetrack.yaml")
    print("Tracker Started")

    for i in range(100):
        ret, frame = video_capture.read()
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    fps_window_start = time.perf_counter()
    fps_window_frames = 0

    while True:
        ret, frame = video_capture.read()
        t1 = time.perf_counter()

        dets = detector.infer(frame)
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
        draw_latency = t4 - t3

        print(
            f"det_latency: {det_latency:.3f}s, track_latency: {track_latency:.3f}s, draw_latency: {draw_latency:.3f}s"
        )

        fps_window_frames += 1
        if fps_window_frames >= FPS_REPORT_EVERY_FRAMES:
            elapsed = time.perf_counter() - fps_window_start
            fps = fps_window_frames / elapsed
            print(
                f"fps (avg over {fps_window_frames} frames, {elapsed:.3f}s): {fps:.2f}"
            )
            fps_window_start = time.perf_counter()
            fps_window_frames = 0
    cv2.destroyAllWindows()
    video_capture.release()


if __name__ == "__main__":
    main()
