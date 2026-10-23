import warnings, os

warnings.filterwarnings('ignore')
from ultralytics import RTDETR



if __name__ == '__main__':
    
    model = RTDETR('ultralytics/cfg/models/rt‑detr/rtdetr‑CGRFPN.yaml')

  
    model.load('RTDETR‑main/weights/rtdetr‑l.pt')

    
    model.train(data='/root/RTDETR‑main/dataset/HRSC2016.yaml',
                cache=False,
                imgsz=640,
                epochs=100,
                batch=8,
                workers=4,
                project='runs/train_HRSC2016',
                name='exp',
                )

   
