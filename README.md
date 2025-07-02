# Private_repo
# SATARK: SMART ASSESSMENT AND TRACKING FOR ACCIDENT RECOGNITION AND KNOWLEDGE.
Accident Recognitation Using Deep Learning and YOLO Algorithm.
![image](https://github.com/user-attachments/assets/1e8b142a-8b4c-4a71-b3f0-4439181fa61e)

# Extracting the dataset
!curl -L "https://universe.roboflow.com/ds/PkTjo0rocb?key=cfsz0255qM" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
# Installing the modules and packages
# Install Ultralytics (YOLO)
!pip install ultralytics
# Check PyTorch + CUDA availability
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
# Train the model
from ultralytics import YOLO
# Load a model
model = YOLO('yolo11n.pt')
# Train the model
train_results = model.train(data = '/content/data.yaml', epochs=70 , 
imgsz=640 , device = 0)
# Testing the best.pt
from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt')
results = model('/content/test/images/Accident-390
_jpg.rf.f704a92d23db7a2f0f9e6aa17db370bc.jpg')
results[0].show()
# Clone the GitHub repository and the VS Code using Git(Which need to be installed)
# Install the Libraries
pip install flask opencv-python ultralytics torch torchvision pillow numpy
# Executing the code and Run the server
python app.py
