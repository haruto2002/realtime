import json
import socket

import paho.mqtt.client as mqtt

BROKER_HOST = "localhost"  # 集約PCのIPアドレス
BROKER_PORT = 1883

CAMERA_ID = "cam01"
PC_ID = socket.gethostname()


class Publisher:
    def __init__(
        self,
        broker_host: str = BROKER_HOST,
        broker_port: int = BROKER_PORT,
        camera_id: str = CAMERA_ID,
        pc_id: str = PC_ID,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.camera_id = camera_id
        self.pc_id = pc_id
        self.pub_topic = f"camera/{self.camera_id}"
        self.pub_client = mqtt.Client(
            client_id=f"{self.pc_id}_{self.camera_id}_publisher"
        )

    def publish_result(self, frame_id, timestamp, objects):
        payload = {
            "camera_id": self.camera_id,
            "pc_id": self.pc_id,
            "timestamp": timestamp,
            "frame_id": frame_id,
            "objects": objects,
        }

        payload_json = json.dumps(payload)

        self.pub_client.publish(
            self.pub_topic,
            payload_json,
            qos=0,
            retain=False,
        )

    def start(self):
        self.pub_client.connect(self.broker_host, self.broker_port, keepalive=60)
        self.pub_client.loop_start()
        print(f"[PUB] started camera_id={self.camera_id}")

    def stop(self):
        self.pub_client.loop_stop()
        self.pub_client.disconnect()
        print(f"[PUB] stopped camera_id={self.camera_id}")
