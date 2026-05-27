import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from processor.publisher import Publisher
from processor.subscriber import Subscriber
from vis_tool.yokohama_hanabi_202508.map_displayer import (
    AllMapDisplayer,
    DrawStyle,
    MapConfig,
)

TIME_OFFSET = {
    "worldporter": 876,
    "akarenga": 2,
    "chosha": 294,
    "kokusaibashi": 41243,
}

PLACES = ["worldporter", "akarenga", "chosha", "kokusaibashi"]
BUFFER_FRAME_NUM = 100
TAIL_FRAME_NUM = 30
START_TIMESTAMP = 902100.0
END_TIMESTAMP = 902110.0
SYNC_FREQ = 1.0
PUBLISH_INTERVAL_SEC = 0.01


# Publish (pub threads)
def load_data(
    data_file: Path,
    timestamp_file: Path,
    time_offset: int = 0,
) -> dict:
    timestamp_data = np.loadtxt(timestamp_file, delimiter=" ")
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        data["timestamp"] = timestamp_data[data["frame_id"] - 1, 1] - time_offset
    return data


def publish_place(place: str, stop_event: threading.Event) -> None:
    publisher = Publisher(camera_id=place)
    publisher.start()
    data_dir = Path(f"/Users/haruto/Desktop/yokohama_202508/output_data/{place}")
    timestamp_file = Path(
        f"/Users/haruto/Desktop/yokohama_202508/timestamp/{place}.txt"
    )
    data_files = sorted(data_dir.glob("*.json"))
    num_data_files = len(data_files)
    print(f"[{place}] data_files: {num_data_files}")

    try:
        while len(data_files) > 0 and not stop_event.is_set():
            data_file = data_files.pop(0)
            data = load_data(data_file, timestamp_file, time_offset=TIME_OFFSET[place])
            publisher.publish_result(
                data["frame_id"], data["timestamp"], data["objects"]
            )
            time.sleep(PUBLISH_INTERVAL_SEC)
    finally:
        publisher.stop()


# Map display
def load_config(yaml_path: Path) -> MapConfig:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return MapConfig(
        path2homography_matrix=Path(cfg["path2homography_matrix"]),
        original_map_size=[float(x) for x in cfg["original_map_size"]],
        all_map_left_top_coor=[float(x) for x in cfg["all_map_left_top_coor"]],
        scale=float(cfg["scale"]),
        draw_style=DrawStyle(
            head_radius=5,
            trail_radius=1,
            rect_width=40,
            rect_height=20,
            line_thickness=2,
        ),
        draw_map_border=False,
    )


def set_visualizer(places: list[str]) -> AllMapDisplayer:
    path2all_map = Path(
        "/Users/haruto/Desktop/yokohama_202508/venue_data/all/all_map.jpg"
    )
    config_dir = Path("/Users/haruto/Desktop/yokohama_202508/all_map_config")
    config_dict = {place: load_config(config_dir / f"{place}.yaml") for place in places}
    all_map_displayer = AllMapDisplayer(path2all_map, config_dict, scale=2.0)
    return all_map_displayer


class TimeSyncDisplayer:
    def __init__(self, all_map_displayer: AllMapDisplayer):
        self.all_map_displayer = all_map_displayer
        self.sync_data: list[dict[str, list[dict]]] = []

    def submit(self, sync_data_dict: dict[str, list[dict]]) -> None:
        self.sync_data.append(sync_data_dict)

    def set_track_data(
        self,
        target_data: list[dict],
    ) -> dict[str, dict[int, np.ndarray]]:
        point_existing_ids, box_existing_ids = self.get_existing_ids(target_data)
        track_data: dict[str, dict[int, list]] = {
            "point": {id: [] for id in point_existing_ids},
            "bbox": {id: [] for id in box_existing_ids},
        }
        for data in target_data:
            point_data = data["objects"]["point"]
            box_data = data["objects"]["bbox"]
            for point in point_data:
                if point["id"] in point_existing_ids:
                    x, y = point["x"], point["y"]
                    coord = [x, y]
                    track_data["point"][point["id"]].append(coord)
            for box in box_data:
                if box["id"] in box_existing_ids:
                    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                    coord = [(x1 + x2) / 2, (y1 + y2) / 2]
                    track_data["bbox"][box["id"]].append(coord)

        track_data_with_array: dict[str, dict[int, np.ndarray]] = {
            "point": {
                id: np.array(points) for id, points in track_data["point"].items()
            },
            "bbox": {id: np.array(boxes) for id, boxes in track_data["bbox"].items()},
        }
        return track_data_with_array

    def get_existing_ids(self, target_data: list[dict]) -> tuple[set[int], set[int]]:
        latest_data = target_data[-1]
        point_existing_ids = set()
        bbox_existing_ids = set()
        for obj in latest_data["objects"]["point"]:
            point_existing_ids.add(obj["id"])
        for obj in latest_data["objects"]["bbox"]:
            bbox_existing_ids.add(obj["id"])
        return point_existing_ids, bbox_existing_ids

    def create_video(self, output_path: Path) -> None:
        h, w = (
            int(
                self.all_map_displayer.all_map_img.shape[0]
                * self.all_map_displayer.scale
            ),
            int(
                self.all_map_displayer.all_map_img.shape[1]
                * self.all_map_displayer.scale
            ),
        )
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            output_path.as_posix(), fourcc, 1 / SYNC_FREQ, (w, h)
        )
        for sync_data in self.sync_data:
            track_data_dict = {
                place: self.set_track_data(data) for place, data in sync_data.items()
            }
            output_img = self.all_map_displayer.run(track_data_dict)
            video_writer.write(output_img)
        video_writer.release()


# Time sync (sub thread)
def wait_for_sync(
    subscriber: Subscriber,
    places: list[str],
    target_timestamp: float,
    timeout_sec: float = 60.0,
) -> dict[str, list[dict]]:
    deadline = time.time() + timeout_sec
    target_data_dict = {}
    pending_places = list(places)
    while not subscriber.stop_event.is_set() and time.time() < deadline:
        for place in list(pending_places):
            target_data = subscriber.get_target_timestamp_tail(place, target_timestamp)
            if target_data is not None:
                target_data_dict[place] = target_data
                pending_places.remove(place)

        if not pending_places:
            return target_data_dict

    raise TimeoutError(
        f"timed out waiting for target_timestamp={target_timestamp}, "
        f"target_data_dict={list(target_data_dict.keys())}"
    )


def sync_time(
    subscriber: Subscriber,
    displayer: TimeSyncDisplayer,
    places: list[str],
    start_ts: float,
    end_ts: float,
    sync_freq: float,
) -> None:
    target_ts = start_ts
    while target_ts < end_ts:
        sync_data_dict = wait_for_sync(subscriber, places, target_ts)
        displayer.submit(sync_data_dict)
        print(
            f"[SYNC] target={target_ts} latency={[(p, v[-1]['timestamp'] - target_ts) for p, v in sync_data_dict.items()]}"
        )
        target_ts += sync_freq


def main(places: list[str] | None = None) -> None:
    places = places or PLACES

    all_map_displayer = set_visualizer(places)
    displayer = TimeSyncDisplayer(all_map_displayer)

    subscriber = Subscriber(
        sub_topic="camera/+", buffer_num=BUFFER_FRAME_NUM, tail_frame_num=TAIL_FRAME_NUM
    )
    subscriber.start()

    stop_event = threading.Event()
    pub_threads: list[threading.Thread] = []
    for place in places:
        t = threading.Thread(
            target=publish_place,
            args=(place, stop_event),
            daemon=True,
        )
        t.start()
        pub_threads.append(t)

    try:
        sync_time(
            subscriber, displayer, places, START_TIMESTAMP, END_TIMESTAMP, SYNC_FREQ
        )
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        subscriber.stop()
        stop_event.set()
        for t in pub_threads:
            t.join()

    time.sleep(1.0)
    displayer.create_video(Path("output_all_map.mp4"))


def check_timestamp() -> None:
    for place in ["worldporter", "akarenga", "chosha", "kokusaibashi"]:
        timestamp_file = Path(
            f"/Users/haruto/Desktop/yokohama_202508/timestamp/{place}.txt"
        )
        timestamp_data = np.loadtxt(timestamp_file, delimiter=" ")
        print(
            f"[{place}] initial ts: {timestamp_data[0, 1] - TIME_OFFSET[place]} last ts: {timestamp_data[-1, 1] - TIME_OFFSET[place]}"
        )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2 and sys.argv[1] == "-c":
        check_timestamp()
    else:
        print("Usage: python time_sync_vis.py [-c]")
        sys.exit(1)
