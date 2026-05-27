import json
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from processor.publisher import Publisher
from processor.subscriber import Subscriber

TIME_OFFSET = {
    "worldporter": 874,
    "akarenga": 0,
    "chosha": 292,
    "kokusaibashi": 41239,
}


def load_data(
    data_file: Path,
    timestamp_file: Path,
    time_offset: int = 0,
):
    timestamp_data = np.loadtxt(timestamp_file, delimiter=" ")
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        data["timestamp"] = timestamp_data[data["frame_id"] - 1, 1] - time_offset
    return data


def get_sync_data(data, target_timestamp=902100.0):

    tail_frame_num = 100
    target_data = deque(maxlen=tail_frame_num)
    target_data.append(data)

    if data["timestamp"] > target_timestamp:
        return True, target_data
    return False, None


def run_sync(place):
    start_timestamp = 902100.0
    end_timestamp = 902380.0
    freq = 1.0
    for timestamp in np.arange(start_timestamp, end_timestamp, freq):
        print(f"Timestamp: {timestamp}")
        sync_data_dict = {place: get_sync_data(place, timestamp) for place in places}


def main():
    place = "worldporter"
    publisher = Publisher(camera_id=place)
    subscriber = Subscriber(sub_topic="camera/+")
    # Subscriberを別スレッドで起動
    subscriber_thread = threading.Thread(
        target=subscriber.start_subscriber, daemon=True
    )
    subscriber_thread.start()

    # Publisherを接続
    publisher.start_publisher()

    data_dir = Path(f"/Users/haruto/Desktop/yokohama_202508/output_data/{place}")
    timestamp_file = Path(
        f"/Users/haruto/Desktop/yokohama_202508/timestamp/{place}.txt"
    )
    data_files = sorted(list(data_dir.glob("*.json")))

    try:
        for data_file in data_files:
            data = load_data(data_file, timestamp_file, time_offset=TIME_OFFSET[place])
            publisher.publish_result(
                data["frame_id"], data["timestamp"], data["objects"]
            )
            time.sleep(0.04)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        publisher.pub_client.loop_stop()
        publisher.pub_client.disconnect()


def check_timestamp():
    places = ["worldporter", "akarenga", "chosha", "kokusaibashi"]
    for place in places:
        timestamp_file = Path(
            f"/Users/haruto/Desktop/yokohama_202508/timestamp/{place}.txt"
        )
        timestamp_data = np.loadtxt(timestamp_file, delimiter=" ")
        print(f"[{place}] Timestamp: {timestamp_data[-1, 1] - TIME_OFFSET[place]}")


if __name__ == "__main__":
    # check_timestamp()
    main()
