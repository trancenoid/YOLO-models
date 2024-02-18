import models
from importlib import reload
reload(models)
from models import YOLOv3
from torch import nn
import torch
import torchvision
from tqdm.notebook import tqdm
from util import predict_transform, write_results
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

device = 'cuda'
model = YOLOv3(in_channels=3)
model.load_weights("yolov3.weights")
model = model.to(device=device)

model_traced = torch.jit.trace(model, torch.ones(1,3,608,608).to(device='cuda'), strict=False)
_pred = torch.jit.script(YOLOv3._pred)

cap = cv2.VideoCapture("test.mp4")
frames = []
count = 0
pbar = tqdm()
device = 'cuda'
# model.to(device)
while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (608,608))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    img = torch.Tensor(frame).float().div(255.0).permute(2,0,1).unsqueeze(0).to(device)
    output = model_traced(img)
    detections = _pred(output)
    for det in detections:
        class_bboxes = det[:5]
        class_name = int(det[5].item())

        x1,y1,x2,y2,score = class_bboxes.cpu()
        x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
        frame = cv2.putText(frame, models.COCO_CLASSES[class_name],(x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3 )
        frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 1 )
    frames.append(frame)

    pbar.update(1)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('output.avi', fourcc, 25, (608,608))
_ = [writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)) for frame in frames[:-1]]
writer.release()