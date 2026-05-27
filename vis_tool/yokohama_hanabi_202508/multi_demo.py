from pathlib import Path

import cv2
import yaml
from map_displayer import AllMapDisplayer, DrawStyle, MapConfig
from single_demo import load_data, set_track_data


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


def main(places: list[str] = ["worldporter", "akarenga", "chosha", "kokusaibashi"]):
    path2all_map = Path(
        "/Users/haruto/Desktop/yokohama_202508/venue_data/all/all_map.jpg"
    )
    config_dir = Path("/Users/haruto/Desktop/yokohama_202508/all_map_config")
    config_dict = {place: load_config(config_dir / f"{place}.yaml") for place in places}
    all_map_displayer = AllMapDisplayer(path2all_map, config_dict, scale=2.0)

    track_data_dict = {place: get_track_data(place) for place in places}
    output_img = all_map_displayer.run(track_data_dict)
    cv2.imwrite("output_all_map.png", output_img)


def get_track_data(place="worldporter"):
    data_dir = Path(f"/Users/haruto/Desktop/yokohama_202508/output_data/{place}")
    target_frame = 120
    tail_frame_num = 100
    target_data = load_data(data_dir, target_frame, tail_frame_num)
    track_data = set_track_data(target_data)
    return track_data


if __name__ == "__main__":
    main()
