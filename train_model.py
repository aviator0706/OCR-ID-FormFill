from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"yolov8l.pt")

    results = model.train(
        data="data.yaml",
        epochs=100,imgsz=1024,batch=4,lr0=0.0001,
        lrf=0.01,
        weight_decay=0.001,
        momentum=0.937,
        warmup_epochs=3.0,
        hsv_h=0.002, hsv_s=0.6, hsv_v=0.3,
        degrees=0.0,
        translate=0.1,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.3,
        mosaic=0.4,
        mixup=0.0,
        copy_paste=0.0,
        device=0,
        save=True,
        val=True,
    )
