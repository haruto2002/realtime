import time
from pathlib import Path

import cv2
from hydra.utils import instantiate
from omegaconf import OmegaConf

from pipeline.detector.p2p_yolo import P2PNetYolo26Detector


def main(
    path2detector_cfg: str,
    path2image: str,
    output_path: str,
) -> None:

    detector: P2PNetYolo26Detector = instantiate(
        OmegaConf.load(Path(path2detector_cfg))
    )
    print("Detectors Setup Done")

    max_frames = 100
    frame_count = 0
    while frame_count < max_frames:
        frame_count += 1
        run(path2image, detector, output_path)


def run(path2image: str, detector: P2PNetYolo26Detector, output_path: str):
    img = cv2.imread(path2image)
    start_ts = time.perf_counter()
    detector.infer_parallel(img)
    # detector.infer_sequential(img)
    # p2pnet_result, yolo26_result = detector.infer_sequential(img)
    detected_ts = time.perf_counter()
    # frame = detector.display_result(img, p2pnet_result, yolo26_result)
    # drawn_ts = time.perf_counter()
    # cv2.imwrite(output_path, frame)
    # end_ts = time.perf_counter()
    # print(
    #     f"total: {ms(end_ts - start_ts)}, detect: {ms(detected_ts - start_ts)}, drawn: {ms(drawn_ts - detected_ts)}, write: {ms(end_ts - drawn_ts)}"
    # )
    print(ms(detected_ts - start_ts))


def ms(time: float) -> str:
    return f"{time * 1000:.2f} ms"


if __name__ == "__main__":
    path2detector_cfg = "img_conf/detector/p2p_yolo/p2p_p2p.yaml"
    # path2detector_cfg = "img_conf/detector/p2p_yolo/p2p_yolo.yaml"
    path2image = "hd_demo.jpg"
    output_path = "hd_demo_output.jpg"
    main(path2detector_cfg, path2image, output_path)
