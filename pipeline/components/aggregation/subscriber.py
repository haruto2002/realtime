import json
import threading
from collections import deque
from typing import Optional

import paho.mqtt.client as mqtt

BROKER_HOST = "localhost"
BROKER_PORT = 1883

SUB_TOPIC = "camera/+"


class Subscriber:
    """
    MQTT Subscriber.

    - `camera/<camera_id>` の `<camera_id>` ごとに固定長バッファを保持します
    - `get_tail(camera_id)` で直近データ列を取得できます
    """

    def __init__(
        self,
        broker_host: str = BROKER_HOST,
        broker_port: int = BROKER_PORT,
        sub_topic: str = SUB_TOPIC,
        buffer_num: int = 100,
        tail_frame_num: int = 30,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.sub_topic = sub_topic
        self.buffer_num = buffer_num
        self.tail_frame_num = tail_frame_num

        self._lock = threading.Lock()
        self._buffers: dict[str, deque[dict]] = {}

        self._client: Optional[mqtt.Client] = None

        self.stop_event = threading.Event()

    def _topic_to_camera_id(self, topic: str) -> str:
        parts = topic.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid topic: {topic}")
        return parts[1]

    def on_connect(self, client, userdata, flags, rc):
        print("[SUB] connected:", rc)

        if rc == 0:
            client.subscribe(self.sub_topic)
            print("[SUB] subscribed:", self.sub_topic)
        else:
            print("[SUB] connection failed")

    def on_message(self, client, userdata, msg):
        try:
            payload_text = msg.payload.decode("utf-8")
            data = json.loads(payload_text)
        except Exception as e:
            print("[SUB] failed to parse message:", e)
            return

        camera_id = self._topic_to_camera_id(msg.topic)
        with self._lock:
            buf = self._buffers.get(camera_id)
            if buf is None:
                buf = deque(maxlen=self.buffer_num)
                self._buffers[camera_id] = buf
            buf.append(data)

    def start(self):
        sub_client = mqtt.Client(client_id="local_test_subscriber")
        self._client = sub_client

        sub_client.on_connect = self.on_connect
        sub_client.on_message = self.on_message

        sub_client.connect(
            self.broker_host,
            self.broker_port,
            keepalive=60,
        )

        sub_client.loop_start()

    def stop(self) -> None:
        client = self._client
        if client is None:
            return
        client.loop_stop()
        client.disconnect()
        self.stop_event.set()

    def get_target_timestamp_tail(
        self, camera_id: str, target_timestamp: float
    ) -> Optional[list[dict]]:
        with self._lock:
            buf = self._buffers.get(camera_id)
            if not buf:
                return None
            if buf[-1]["timestamp"] < target_timestamp:
                return None
            for i, data in enumerate(buf):
                if data["timestamp"] >= target_timestamp:
                    return list(buf)[max(0, i - self.tail_frame_num) : i + 1]
            return None

    def get_camera_ids(self) -> list[str]:
        with self._lock:
            return list(self._buffers.keys())


if __name__ == "__main__":
    subscriber = Subscriber()
    subscriber.start()
