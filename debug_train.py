import os
import traceback
from doclayout_yolo import YOLOv10

try:
    # Use absolute paths for everything
    model_path = r'D:\Yolo doc layout\DocLayout-YOLO\doclayout_yolo_docstructbench_imgsz1024.pt'
    data_path = r'D:\Yolo doc layout\DocLayout-YOLO\viet-pcc.yaml'
    
    model = YOLOv10(model_path, task='detect')
    print(f"Model loaded. Task: {model.task}")
    
    # Try to explicitly set the trainer to avoid yolov8n fallback
    results = model.train(
        data=data_path, 
        epochs=1, 
        imgsz=1024, 
        batch=2,
        device=0,
        project='viet-pcc-training',
        name='debug_run',
        val=False,
        save=True
    )
except Exception as e:
    traceback.print_exc()
