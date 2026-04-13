from ultralytics import YOLO

model = YOLO(r"D:\sampah\runs\train\sampah-yolo11s3\weights\best.pt")

model.predict(
    source=r"D:\sampah\images",  # <-- ganti ke folder gambar kamu
    device=0,        # GPU (RTX 4050)
    conf=0.25,
    imgsz=640,
    save=True,       # simpan hasil gambar dengan bbox
    save_txt=False,  # True kalau mau label .txt juga
    project=r"D:\sampah\inference11s",
    name="porn_batch",
    exist_ok=True
)
