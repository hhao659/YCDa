import io

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from ultralytics import YOLO

# 初始化 FastAPI 应用
app = FastAPI(title="YOLO Model API", description="API for object detection using Ultralytics YOLO model")

# 加载 Ultralytics 中的训练好的模型
model = YOLO("/home/i/ultralytics/runs/detect/train/weights/best.pt")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """返回JSON格式的检测结果."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    results = model(image)

    predictions = []
    for result in results:
        for box in result.boxes:
            pred = {"class": result.names[int(box.cls)], "confidence": float(box.conf), "bbox": box.xyxy.tolist()}
            predictions.append(pred)

    return {"predictions": predictions}


@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    """返回带检测框的图片."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    results = model(image_cv)
    annotated_image = results[0].plot()

    annotated_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    img_byte_arr = io.BytesIO()
    annotated_pil.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    return StreamingResponse(
        img_byte_arr,
        media_type="image/jpeg",
        headers={"Content-Disposition": "attachment; filename=detected_image.jpg"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
