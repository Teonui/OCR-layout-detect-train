import os
import cv2
import torch
from doclayout_yolo import YOLOv10
from pathlib import Path

# Config
MODEL_PATH = r"viet-pcc-training/yolov10m_viet-pcc_v2_fined/weights/best.pt"
VAL_DIR = r"D:\Yolo doc layout\DocLayout-YOLO\layout_data\viet-pcc\images\val"
OUT_DIR = r"outputs_v2_fined"
IMG_SIZE = 1024
CONF = 0.25

def main():
    # Setup
    os.makedirs(OUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model: {MODEL_PATH}")
    model = YOLOv10(MODEL_PATH)
    
    # Predict
    images = list(Path(VAL_DIR).glob("*.jpg"))
    print(f"Found {len(images)} images in {VAL_DIR}")
    
    for img_path in images:
        print(f"Predicting: {img_path.name}")
        det_res = model.predict(
            str(img_path),
            imgsz=IMG_SIZE,
            conf=CONF,
            device=device,
        )
        
        # Plot and Save
        # font_size and line_width for visibility
        annotated_frame = det_res[0].plot(pil=True, line_width=4, font_size=18)
        output_path = os.path.join(OUT_DIR, img_path.name.replace(".jpg", "_res.jpg"))
        cv2.imwrite(output_path, annotated_frame)
        print(f"Saved: {output_path}")

    print(f"\nBatch prediction finished. View results in: {OUT_DIR}")

if __name__ == "__main__":
    main()
