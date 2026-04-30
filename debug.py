from ultralytics import YOLO


def main():
    model_sizes = ["x", "n", "s", "m", "l"]
    for model_size in model_sizes:
        model = YOLO(f"yolo26{model_size}-pose.pt")


if __name__ == "__main__":
    main()
