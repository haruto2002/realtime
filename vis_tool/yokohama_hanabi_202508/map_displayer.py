from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

TrackData = dict[str, dict[int, np.ndarray]]


@dataclass
class DrawStyle:
    head_radius: int = 5
    trail_radius: int = 1
    rect_width: int = 40
    rect_height: int = 20
    line_thickness: int = 2


def project_points(points: np.ndarray, homography_matrix: np.ndarray) -> np.ndarray:
    points = cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography_matrix)
    return points.reshape(-1, 2)


def get_color(label_id: int) -> tuple[int, int, int]:
    return (
        int((37 * label_id) % 255),
        int((17 * label_id) % 255),
        int((29 * label_id) % 255),
    )


def create_rectangle(
    center_point: np.ndarray,
    angle: float,
    rect_width: int,
    rect_height: int,
) -> np.ndarray:
    rect_corners = np.array(
        [
            [-rect_width / 2, -rect_height / 2],
            [rect_width / 2, -rect_height / 2],
            [rect_width / 2, rect_height / 2],
            [-rect_width / 2, rect_height / 2],
        ]
    )
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rot_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    rotated_corners = rect_corners @ rot_matrix.T
    final_corners = rotated_corners + np.array([center_point[0], center_point[1]])
    return np.round(final_corners).astype(np.int32).reshape((-1, 1, 2))


def draw_tracks(
    output_img: np.ndarray,
    data: TrackData,
    to_map_coords: Callable[[np.ndarray], np.ndarray],
    style: DrawStyle = DrawStyle(),
    scale: float = 1.0,
) -> np.ndarray:
    for label_id, points in data["point"].items():
        color = get_color(label_id)
        map_points = to_map_coords(points)
        for map_point in map_points[:-1]:
            x, y = int(map_point[0] * scale), int(map_point[1] * scale)
            cv2.circle(
                output_img,
                (x, y),
                radius=int(style.trail_radius),
                color=color,
                thickness=-1,
            )
        cv2.circle(
            output_img,
            (int(map_points[-1][0] * scale), int(map_points[-1][1] * scale)),
            radius=int(style.head_radius),
            color=color,
            thickness=-1,
        )

    for label_id, boxes in data["bbox"].items():
        color = get_color(label_id)
        map_center_points = to_map_coords(boxes)
        for map_center_point in map_center_points[:-1]:
            x, y = int(map_center_point[0] * scale), int(map_center_point[1] * scale)
            cv2.circle(
                output_img,
                (x, y),
                radius=int(style.trail_radius),
                color=color,
                thickness=-1,
            )
        cv2.circle(
            output_img,
            (
                int(map_center_points[-1][0] * scale),
                int(map_center_points[-1][1] * scale),
            ),
            radius=int(style.head_radius),
            color=color,
            thickness=-1,
        )
        angle = np.arctan2(
            map_center_points[-1][1] - map_center_points[0][1],
            map_center_points[-1][0] - map_center_points[0][0],
        )
        rect_corners = create_rectangle(
            map_center_points[-1] * scale,
            angle,
            rect_width=int(style.rect_width),
            rect_height=int(style.rect_height),
        )
        cv2.polylines(
            output_img,
            [rect_corners],
            isClosed=True,
            color=color,
            thickness=int(style.line_thickness),
        )

    return output_img


def draw_map_border(
    output_img: np.ndarray, config: "MapConfig", scale: float = 1.0
) -> np.ndarray:
    if config.original_map_size is None:
        return output_img

    top_left = (
        int(config.all_map_left_top_coor[0] * scale),
        int(config.all_map_left_top_coor[1] * scale),
    )
    bottom_right = (
        int(
            config.all_map_left_top_coor[0] + config.original_map_size[0] * config.scale
        ),
        int(
            config.all_map_left_top_coor[1] + config.original_map_size[1] * config.scale
        ),
    )
    cv2.rectangle(output_img, top_left, bottom_right, (0, 255, 0), 5)
    return output_img


@dataclass
class MapConfig:
    path2homography_matrix: Path
    original_map_size: list[float] | None = None
    all_map_left_top_coor: list[float] = field(default_factory=lambda: [0.0, 0.0])
    scale: float = 1.0
    draw_style: DrawStyle = field(default_factory=DrawStyle)
    draw_map_border: bool = False
    homography_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.homography_matrix = np.loadtxt(self.path2homography_matrix)

    def to_map_coords(self, points: np.ndarray) -> np.ndarray:
        bev_points = project_points(points, self.homography_matrix)
        return bev_points * self.scale + np.array(self.all_map_left_top_coor)


class MapDisplayer:
    map_img: np.ndarray
    map_config: MapConfig
    scale: float

    def __init__(
        self,
        path2map_image: Path,
        map_config: MapConfig,
        scale: float = 1.0,
    ):
        assert path2map_image.exists(), (
            f"Map image file does not exist: {path2map_image}"
        )
        map_img = cv2.imread(path2map_image.as_posix())
        if map_img is None:
            raise FileNotFoundError(f"Failed to read map image: {path2map_image}")
        self.map_img = map_img
        self.map_config = map_config
        self.scale = scale

    def run(self, data: TrackData) -> np.ndarray:
        output_img = self.map_img.copy()
        output_img = cv2.resize(
            output_img,
            (
                int(output_img.shape[1] * self.scale),
                int(output_img.shape[0] * self.scale),
            ),
        )
        return draw_tracks(
            output_img,
            data,
            self.map_config.to_map_coords,
            style=self.map_config.draw_style,
            scale=self.scale,
        )


class AllMapDisplayer:
    all_map_img: np.ndarray
    config_dict: dict[str, MapConfig]
    scale: float

    def __init__(
        self,
        path2all_map: Path,
        config_dict: dict[str, MapConfig],
        scale: float = 1.0,
    ):
        all_map_img = cv2.imread(path2all_map.as_posix())
        if all_map_img is None:
            raise FileNotFoundError(
                f"All map image file does not exist: {path2all_map}"
            )
        self.all_map_img = all_map_img
        self.config_dict = config_dict
        self.scale = scale

    def run(self, data_dict: dict[str, TrackData]) -> np.ndarray:
        output_img = self.all_map_img.copy()
        output_img = cv2.resize(
            output_img,
            (
                int(output_img.shape[1] * self.scale),
                int(output_img.shape[0] * self.scale),
            ),
        )
        for place, data in data_dict.items():
            config = self.config_dict[place]
            output_img = draw_tracks(
                output_img,
                data,
                config.to_map_coords,
                style=config.draw_style,
                scale=self.scale,
            )
            if config.draw_map_border:
                output_img = draw_map_border(output_img, config, scale=self.scale)
        return output_img
