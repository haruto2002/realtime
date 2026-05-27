import os
import time

import cv2

from processor.reader import FFmpegVideoReader
from processor.timer import TimeCounter


def main(place: str):
    time_counter = TimeCounter()
    reader = FFmpegVideoReader(
        time_counter,
        f"/Users/haruto/Desktop/yokohama_202508/video/{place}.MOV",
        # (7680, 4320),
        (1920 // 3, 1080 // 3),
        output_fps=30,
        frame_queue_maxsize=100,
        flush_on_queue_full=False,
    )
    reader.start()

    save_dir = "/Users/haruto/Desktop/yokohama_202508/timestamp"
    os.makedirs(save_dir, exist_ok=True)

    timestamp_file = os.path.join(save_dir, f"{place}.txt")
    with open(timestamp_file, "w") as f:
        f.write("")

    frame_id = 0
    while True:
        frame_id += 1
        print(f"Frame {frame_id}")
        frame, seq, frame_ts = reader.get_next()
        if frame is None:
            frame_id -= 1
            continue
        with open(timestamp_file, "a") as f:
            f.write(f"{frame_id} {frame_ts}\n")

        if frame_id > 29 * 5 * 60:
            break
    reader.stop()


def main2(place: str):
    video_path = f"/Users/haruto/Desktop/yokohama_202508/video/{place}.MOV"
    save_dir = "/Users/haruto/Desktop/yokohama_202508/timestamp"
    os.makedirs(save_dir, exist_ok=True)
    timestamp_file = os.path.join(save_dir, f"{place}.txt")
    with open(timestamp_file, "w") as f:
        f.write("")

    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    frame_id = 0
    start_ts = None
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_id += 1
        print(f"Frame {frame_id}")
        frame_ts = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # 秒単位
        if frame_id == 101:
            start_ts = time.time()
        if frame_id > 100:
            with open(timestamp_file, "a") as f:
                f.write(f"{frame_id} {frame_ts}\n")
        if frame_id == 201:
            end_ts = time.time()
            fps_calc = 100 / (end_ts - start_ts)
            print(f"FPS: {fps_calc:.2f}")
            break
    video_capture.release()


if __name__ == "__main__":
    # main("akarenga")
    # main("chosha")
    main("kokusaibashi")
    # main("worldporter")
