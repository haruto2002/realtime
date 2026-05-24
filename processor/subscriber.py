import json
from datetime import datetime

import paho.mqtt.client as mqtt

BROKER_HOST = "localhost"
BROKER_PORT = 1883

SUB_TOPIC = "camera/+"


class Subscriber:
    def __init__(
        self,
        broker_host: str = BROKER_HOST,
        broker_port: int = BROKER_PORT,
        sub_topic: str = SUB_TOPIC,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.sub_topic = sub_topic

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

        print("\n[SUB] message received")
        print("  topic     :", msg.topic)
        print("  camera_id :", data.get("camera_id"))
        print("  pc_id     :", data.get("pc_id"))
        print("  timestamp :", datetime.fromtimestamp(data.get("timestamp")))
        print("  frame_id  :", data.get("frame_id"))
        print("  objects   :", data.get("objects"))

    def start_subscriber(self):
        sub_client = mqtt.Client(client_id="local_test_subscriber")

        sub_client.on_connect = self.on_connect
        sub_client.on_message = self.on_message

        sub_client.connect(
            self.broker_host,
            self.broker_port,
            keepalive=60,
        )

        sub_client.loop_forever()


if __name__ == "__main__":
    subscriber = Subscriber()
    subscriber.start_subscriber()
