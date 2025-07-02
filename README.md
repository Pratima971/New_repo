# SATARK: Smart Assessment and Tracking for Accident Recognition and Knowledge.
A real-time car accident detection system leveraging YOLOv11 for fast and accurate identification of traffic incidents using advanced deep learning techniques.
![image](https://github.com/user-attachments/assets/1e8b142a-8b4c-4a71-b3f0-4439181fa61e)
https://github.com/user-attachments/assets/9d7c6d02-f2b0-4162-8087-0cd2e63d0f66
## 1. Extracting the dataset
```
!curl -L "https://universe.roboflow.com/ds/PkTjo0rocb?key=cfsz0255qM" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```
## 2. Installing the modules and packages
### Install Ultralytics (YOLO)
```
!pip install ultralytics
```
## 3. Check PyTorch + CUDA availability
```
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```
## 4. Load a model
```
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
```
## 5. Train the model
```
train_results = model.train(data = '/content/data.yaml', epochs=100 , imgsz=640 , device = 0)
```
## 6. Testing the model
```
from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt')
results = model('/content/test/images/Accident-390_jpg.rf.f704a92d23db7a2f0f9e6aa17db370bc.jpg')
results[0].show()
```
$ Clone the GitHub repository and the VS Code using Git(Which need to be installed)
## Install the Libraries
```
pip install flask opencv-python ultralytics torch torchvision pillow numpy
```
## Executing the code and Run the server
```
python app.py
```
