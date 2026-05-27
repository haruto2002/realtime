import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from map_displayer import MapConfig, MapDisplayer


@dataclass
class PointData:
    id: int
    x: float
    y: float


@dataclass
class BoxData:
    id: int
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class ObjectData:
    point: list[PointData]
    box: list[BoxData]


@dataclass
class FrameData:
    camera_id: str
    pc_id: str
    timestamp: float
    frame_id: int
    objects: ObjectData


def load_data(
    data_dir: Path,
    timestamp_file: Path,
    target_frame: int,
    tail_frame_num: int = 30,
):
    data_files = sorted(list(data_dir.glob("*.json")))
    data_files = data_files[target_frame - tail_frame_num : target_frame]

    timestamp_data = np.loadtxt(timestamp_file, delimiter=" ")

    target_data = []
    for data_file in data_files:
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["timestamp"] = timestamp_data[data["frame_id"] - 1, 1]
            target_data.append(data)

    return target_data


def set_track_data(target_data: dict):
    point_existing_ids, box_existing_ids = get_existing_ids(target_data)
    track_data = {
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

    track_data = {
        "point": {id: np.array(points) for id, points in track_data["point"].items()},
        "bbox": {id: np.array(boxes) for id, boxes in track_data["bbox"].items()},
    }
    return track_data


def get_existing_ids(target_data: dict):
    latest_data = target_data[-1]
    point_existing_ids = set()
    bbox_existing_ids = set()
    for obj in latest_data["objects"]["point"]:
        point_existing_ids.add(obj["id"])
    for obj in latest_data["objects"]["bbox"]:
        bbox_existing_ids.add(obj["id"])
    return point_existing_ids, bbox_existing_ids


def main(place="worldporter"):
    data_dir = Path(f"/Users/haruto/Desktop/yokohama_202508/output_data/{place}")
    timestamp_file = Path(
        f"/Users/haruto/Desktop/yokohama_202508/timestamp/{place}.txt"
    )
    target_frame = 120
    tail_frame_num = 100
    target_data = load_data(
        data_dir,
        timestamp_file,
        target_frame,
        tail_frame_num,
    )
    print(target_data[-1]["timestamp"])
    track_data = set_track_data(target_data)

    path2map_image = Path(
        f"/Users/haruto/Desktop/yokohama_202508/venue_data/{place}/map.jpg"
    )
    path2homography_matrix = Path(
        f"/Users/haruto/Desktop/yokohama_202508/venue_data/{place}/homography.txt"
    )
    map_config = MapConfig(path2homography_matrix=path2homography_matrix)
    map_displayer = MapDisplayer(path2map_image, map_config, scale=2.0)
    output_img = map_displayer.run(track_data)
    cv2.imwrite(f"output_{place}.png", output_img)


if __name__ == "__main__":
    main(place="worldporter")
    main(place="akarenga")
    main(place="chosha")
    main(place="kokusaibashi")
