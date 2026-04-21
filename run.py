import argparse
import threading
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

from processor.components import Detector, Tracker
from processor.displayer import Displayer
from processor.reader import FFmpegRTSPReader
from processor.timer import TimeCounter
from processor.worker import MotWorker

# HOST = "member"
# IP = "192.168.0.10"
# PORT = 554
# PW = "AIST-rwdc"
# RTSP_URL = f"rtsp://{HOST}:{PW}@{IP}:{PORT}/ONVIF/MediaInput?profile=def_profile1"
# W, H = 1920, 1080


def main(cfg_dir: Path, max_wall_seconds: float | None = None) -> None:
    time_counter: TimeCounter = instantiate(OmegaConf.load(cfg_dir / "timer.yaml"))
    print("TimeCounter Setup Done")

    reader: FFmpegRTSPReader = instantiate(
        OmegaConf.load(cfg_dir / "reader.yaml"),
        time_counter=time_counter,
    )
    reader.start()
    print("Reader Started")

    displayer: Displayer = instantiate(
        OmegaConf.load(cfg_dir / "displayer.yaml"),
        time_counter=time_counter,
    )
    print("Displayer Setup Done")

    detector: Detector = instantiate(OmegaConf.load(cfg_dir / "detector.yaml"))
    print("Detector Setup Done")
    tracker: Tracker = instantiate(OmegaConf.load(cfg_dir / "tracker.yaml"))
    print("Tracker Setup Done")

    worker = MotWorker(
        time_counter=time_counter,
        reader=reader,
        displayer=displayer,
        detector=detector,
        tracker=tracker,
    )
    worker.start()
    print("Worker Started")

    if max_wall_seconds is not None and max_wall_seconds > 0:
        threading.Timer(max_wall_seconds, displayer.request_stop).start()
        print(f"Will stop after {max_wall_seconds} s (wall clock)")

    displayer.run_loop()

    worker.stop()
    reader.stop()

    time_counter.save()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_dir", type=str, default="processor/conf")
    parser.add_argument(
        "--max_wall_seconds",
        type=float,
        default=None,
        help="この秒数経過後に表示ループを止め、通常どおり save して終了する。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(Path(args.cfg_dir), max_wall_seconds=args.max_wall_seconds)
