import json
import threading
import time
from pathlib import Path

import numpy as np

from processor.publisher import Publisher
from processor.subscriber import Subscriber

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


def wait_for_sync(
    subscriber: Subscriber,
    places: list[str],
    target_timestamp: float,
    timeout_sec: float = 10.0,
    poll_interval_sec: float = 0.1,
) -> dict[str, list[dict]]:
    deadline = time.time() + timeout_sec
    target_data_dict = {}
    pending_places = list(places)
    while time.time() < deadline:
        for place in list(pending_places):
            target_data = subscriber.get_target_timestamp_tail(place, target_timestamp)
            if target_data is not None:
                target_data_dict[place] = target_data
                pending_places.remove(place)

        if not pending_places:
            return target_data_dict

        time.sleep(poll_interval_sec)

    raise TimeoutError(
        f"timed out waiting for target_timestamp={target_timestamp}, "
        f"target_data_dict={list(target_data_dict.keys())}"
    )


def publish_place(place: str, stop_event: threading.Event) -> None:
    publisher = Publisher(camera_id=place)
    publisher.start()
    data_dir = Path(f"/Users/haruto/Desktop/yokohama_202508/output_data/{place}")
    timestamp_file = Path(
        f"/Users/haruto/Desktop/yokohama_202508/timestamp/{place}.txt"
    )
    data_files = sorted(data_dir.glob("*.json"))
    print(f"[{place}] data_files: {len(data_files)}")

    try:
        for data_file in data_files:
            if stop_event.is_set():
                break
            data = load_data(data_file, timestamp_file, time_offset=TIME_OFFSET[place])
            publisher.publish_result(
                data["frame_id"], data["timestamp"], data["objects"]
            )
            time.sleep(PUBLISH_INTERVAL_SEC)
    finally:
        publisher.stop()


def run_sync(subscriber: Subscriber, places: list[str]) -> None:
    for target_ts in np.arange(START_TIMESTAMP, END_TIMESTAMP, SYNC_FREQ):
        target_ts = float(target_ts)
        snap = wait_for_sync(subscriber, places, target_ts)
        print(
            f"[SYNC] target={target_ts} latency={[(p, v[-1]['timestamp'] - target_ts) for p, v in snap.items()]}"
        )


def main(places: list[str] | None = None) -> None:
    places = places or PLACES

    subscriber = Subscriber(
        sub_topic="camera/+", buffer_num=BUFFER_FRAME_NUM, tail_frame_num=TAIL_FRAME_NUM
    )
    subscriber.start_subscriber()
    time.sleep(0.5)

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
        run_sync(subscriber, places)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        for t in pub_threads:
            t.join(timeout=2.0)
        subscriber.stop()


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
    main()
    # check_timestamp()
