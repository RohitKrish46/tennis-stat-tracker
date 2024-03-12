from ultralytics import YOLO

# model = YOLO('models/yolov5_last.pt')
model = YOLO('yolov8x.pt')

model.track('input/input_video.mp4', conf=0.2, save=True)