import os
import time
from pdf2image import convert_from_path
from concurrent.futures import ProcessPoolExecutor
from ultralytics import YOLO
import torch

# --- CẤU HÌNH ---
PDF_PATH = r'D:\Yolo doc layout\DocLayout-YOLO\images\train\ss3\pdf'
OUTPUT_FOLDER = r'D:\Yolo doc layout\DocLayout-YOLO\images\train\ss3\jpg'
MODEL_PATH = r'D:\path\to\your\best.pt' # Đường dẫn tới file best.pt của bạn
DPI = 200

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# 1. Bước CPU: Convert PDF sang JPG (Đa nhân)
def cpu_convert_pdf(filename):
    try:
        full_path = os.path.join(PDF_PATH, filename)
        images = convert_from_path(full_path, dpi=DPI, thread_count=4)
        saved_paths = []
        base_name = os.path.splitext(filename)[0]
        for i, image in enumerate(images):
            img_name = f"{base_name}_p{i}.jpg"
            save_path = os.path.join(OUTPUT_FOLDER, img_name)
            image.save(save_path, 'JPEG', quality=90)
            saved_paths.append(save_path)
        return saved_paths
    except Exception as e:
        print(f"Lỗi convert {filename}: {e}")
        return []

if __name__ == '__main__':
    # Kiểm tra GPU
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Đang sử dụng thiết bị: {'RTX 4060' if device == '0' else 'CPU'}")

    pdf_files = [f for f in os.listdir(PDF_PATH) if f.endswith(".pdf")]
    
    # CHẠY CPU: Convert hàng loạt
    print(f"⏳ Đang convert {len(pdf_files)} file PDF...")
    all_images = []
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(cpu_convert_pdf, pdf_files))
        for res in results:
            all_images.extend(res)

    # 2. Bước GPU: Auto-Labeling (Dùng RTX 4060)
    print(f"🤖 Khởi động GPU để tự động gán nhãn cho {len(all_images)} ảnh...")
    model = YOLO(MODEL_PATH)
    
    # Chạy inference trên GPU cho toàn bộ ảnh vừa tạo
    # save_txt=True sẽ tự tạo file .txt chuẩn YOLO cho AnyLabeling
    model.predict(
        source=all_images, 
        conf=0.25, 
        device=device, 
        save_txt=True, 
        imgsz=1024,
        project=OUTPUT_FOLDER, # Lưu trực tiếp vào thư mục ảnh
        name='labels',
        exist_ok=True
    )

    print(f"\n✅ Xong! Toàn bộ ảnh đã được tạo và gán nhãn tự động.")
    print(f"📂 Bạn chỉ cần mở AnyLabeling và bắt đầu kiểm tra.")