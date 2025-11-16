import os
from ultralytics import YOLO

###############################################
# CONFIG
###############################################

DATA_YAML = "merged_yolo/data.yaml"  # путь к твоему merged датасету
MODEL = "yolov9c.pt"                 # можешь поменять на yolov9e.pt для макс качества
EPOCHS = 75
IMG_SIZE = 640
BATCH = 8
PROJECT = "runs_yolov9"              # куда сохранять результаты
NAME = "signature_stamp_qr3"          # имя эксперимента

###############################################
# TRAINING
###############################################

def main():
    print("📦 Loading model:", MODEL)
    model = YOLO(MODEL)

    print("🚀 Starting training...")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=PROJECT,
        name=NAME,
        device=0,               # 0 = GPU. можно поставить "cpu"
        workers=4,
        patience=50,
        cos_lr=True,            # smooth learning rate
        amp=True                # ускорение на FP16
    )

    print("\n🎉 Training complete!")
    print(f"🔍 Best weights saved in: {PROJECT}/{NAME}/weights/best.pt")


if __name__ == "__main__":
    main()
