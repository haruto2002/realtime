import cv2
import numpy as np

from components import Detector, Tracker
from track_reader import FFmpegRTSPReader

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
        output_fps=10,
        frame_queue_maxsize=10,
        reconnect_backoff_sec=0.5,
    )
    reader.start()
    print("Reader Started")

    detector = Detector("pipeline/config/deimv2.yaml")
    print("Detector Started")
    tracker = Tracker("pipeline/config/bytetrack.yaml")
    print("Tracker Started")

    while True:
        frame, seq, ts = reader.get_next()
        if frame is None:
            continue

        dets = detector.infer(frame)
        frame = detector.draw(frame, dets)
        # tracks = tracker.update(convert_to_tracker_inputs(dets))
        # frame = tracker.draw(frame, tracks)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    reader.stop()


def convert_to_tracker_inputs(dets):
    dets = [dets[:, :4], dets[:, 5:6]]
    dets = np.concatenate(dets, axis=1)
    return dets


if __name__ == "__main__":
    main()
