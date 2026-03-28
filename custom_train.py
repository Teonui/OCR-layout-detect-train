import os
from doclayout_yolo import YOLOv10
import torch

# Configuration
DATA_YAML = "D:/Yolo doc layout/DocLayout-YOLO/viet-pcc.yaml"
MODEL_PT = "viet-pcc-training/yolov10m_viet-pcc_v1_custom/weights/best.pt"
IMG_SIZE = 1024
BATCH_SIZE = 2
EPOCHS = 50 # Giảm epoch cho fine-tuning vì đã có base tốt
PROJECT = "viet-pcc-training"
NAME = "yolov10m_viet-pcc_v2_fined"

if __name__ == "__main__":
    # 1. Initialize Model
    print(f"Loading model: {MODEL_PT}")
    model = YOLOv10(MODEL_PT)

    # 2. Start Training
    print(f"Starting training on {DATA_YAML}")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT,
        name=NAME,
        device=0,  # RTX 4060
        workers=4,
        plots=True,
        save_period=10,
        val=False, # Disable validation for initial run to ensure stability
        exist_ok=True,
        optimizer='auto',
        lr0=0.001,
        warmup_epochs=3.0,
        mosaic=1.0,
    )
    
    print("Training finished successfully.")
