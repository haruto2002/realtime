from ultralytics import YOLO


def main():
    # model_sizes = ["x", "n", "s", "m", "l"]
    # for model_size in model_sizes:
    #     model = YOLO(f"yolo26{model_size}-pose.pt")

    model = YOLO("weights/yolo26/yolo26x.pt")
    print(model.names)


if __name__ == "__main__":
    main()
