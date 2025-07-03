# SATARK: Smart Assessment and Tracking for Accident Recognition and Knowledge.
A real-time car accident detection system leveraging YOLOv11 for fast and accurate identification of traffic incidents using advanced deep learning techniques.

![image](https://github.com/Pratima971/New_repo/blob/main/image_7.png)

![image](https://github.com/user-attachments/assets/1e8b142a-8b4c-4a71-b3f0-4439181fa61e)

![image](https://github.com/Pratima971/New_repo/blob/main/image_1.png)

https://github.com/user-attachments/assets/5fd60b88-b087-48fa-bb36-67e49aeaf7b7



## 1. Extracting the dataset
### Dataset
Train = 75%

Validation = 15%

Test = 10%
```
!curl -L "https://universe.roboflow.com/ds/PkTjo0rocb?key=cfsz0255qM" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```
## 2. Installing the modules and packages
### Install Ultralytics (YOLO)
```
!pip install ultralytics
```
## 3. Check PyTorch + CUDA availability

Verifying PyTorch and CUDA ensures GPU support for faster training and inference of deep learning models like YOLO.

```
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```
## 4. Load a model
DOWNLOAD AND ACCESS THE MODEL
https://drive.google.com/file/d/1OWXwoMoyhyrDcBqYxayWb-TR6gjcFCMq/view?usp=sharing

Load the pre-trained YOLO model to initialize it for inference or fine-tuning.

```
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
```
## 5. Train the model

Train the YOLO model on labeled data to adapt its weights for task-specific performance.

```
train_results = model.train(data = '/content/data.yaml', epochs=100 , imgsz=640 , device = 0)
```
## 6. Testing the model 

Evaluate the trained model on test data to assess detection performance and generalization.

```
from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt')
results = model('/content/test/images/Accident-390_jpg.rf.f704a92d23db7a2f0f9e6aa17db370bc.jpg')
results[0].show()
```
$ Clone the GitHub repository and the VS Code using Git(Which need to be installed)
## Install the Libraries

Installs all required libraries for building a real-time car accident detection system using Flask for the web interface, OpenCV for video processing, YOLOv11 via Ultralytics for object detection, and supporting deep learning and image handling libraries.

```
pip install flask opencv-python ultralytics torch torchvision pillow numpy
```
## Executing the code and Run the server

Start the application by executing the main script:

```
python app.py
```
##
##
