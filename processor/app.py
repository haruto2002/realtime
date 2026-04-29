import threading

from hydra.utils import instantiate


class MotApp:
    def __init__(
        self,
        time_counter,
        reader,
        displayer,
        worker,
        max_wall_seconds=None,
    ):
        self.time_counter = instantiate(time_counter)
        print("TimeCounter Setup Done")

        self.reader = instantiate(reader, time_counter=self.time_counter)
        print("Reader Setup Done")
        self.displayer = instantiate(displayer, time_counter=self.time_counter)
        print("Displayer Setup Done")

        self.worker = instantiate(
            worker,
            time_counter=self.time_counter,
            reader=self.reader,
            displayer=self.displayer,
        )
        print("Worker Setup Done")

        self.max_wall_seconds = max_wall_seconds

    def run(self):
        self.reader.start()
        self.worker.start()

        if self.max_wall_seconds is not None and self.max_wall_seconds > 0:
            threading.Timer(
                self.max_wall_seconds,
                self.displayer.request_stop,
            ).start()
            print(f"Will stop after {self.max_wall_seconds} s (wall clock)")

        try:
            self.displayer.run_loop()
        finally:
            self.worker.stop()
            self.reader.stop()
            self.time_counter.save()
