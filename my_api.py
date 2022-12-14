
from flask import Flask, render_template, request, json
from flask import Markup
from flask_cors import CORS, cross_origin
import os
import base64

import random

from PIL import Image


import cv2
import torch
from numpy import random

from models.experimental import attempt_load
from utils.dataloaders import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_boxes, xyxy2xywh, strip_optimizer, set_logging, increment_path
    
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


# Khởi tạo Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
app.config['UPLOAD_FOLDER'] = "static"
cors = CORS(app)


# Load model
weights = 'best.pt'
set_logging()
device = select_device('')
half = device.type != 'cpu'
imgsz = 640

# Load model
model = attempt_load(weights, device=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size
if half:
    model.half()  # to FP16

desc_file = "xray_desc.csv"
#f = open(desc_file,  "r")
f = open(desc_file,  encoding="utf8")
desc = f.readlines()
f.close()
dict = {}
for line in desc:
    dict[line.split('|')[0]] = line.split('|')[1].replace("\n", "")


@app.route("/", methods=['POST'])
def home_page():
    image = request.files['file']
    if image:
        # Lưu file
        print(image.filename)
        print(app.config['UPLOAD_FOLDER'])
        source = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        print("Save = ", source)
        image.save(source)

        # source = "data/images/sample4.jpg"
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        _colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        conf_thres = 0.25
        iou_thres = 0.25
        
        # for path, img, im0s, vid_cap in dataset:
        for path, img, im0s, _cap, _s in dataset:
            
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # if len(img.shape) == 3:
            #     img = img[None]  # expand for batch dim

            # Inference
            pred = model(img, augment=False)[0]
            # pred = model(img, augment=False, visualize=False)
            # Apply NMS
            # pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
            
            extra = ""
            # Process detections
            # print(pred)
            
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                annotator = Annotator(im0, line_width=2, example=str(names))
                # p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                save_path = source
                print(det)
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
                    # Write results
                    explain = []
                    for *xyxy, conf, cls in reversed(det):
                        if save_img:  # Add bbox to image
                            # annotator.box_label(xyxy, label, color=colors(c, True))

                            
                            label = f'{names[int(cls)]} {conf:.2f}'
                            # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                            # print("dot1")
                            # print(_colors[1])
                            annotator.box_label(xyxy, label, color=colors(int(cls), True))
                            # print("dot2")
                            # extra += str(names[int(cls)]) + "(" + dict[names[int(cls)]] + ") với độ tin cậy {:.2f}".format(conf)
                            explain.append(str(names[int(cls)]) + "(" + dict[names[int(cls)]] + ") với độ tin cậy {:.2f}".format(conf))
                    if save_img:
                        if dataset.mode == 'image':
                            print(save_path)
                            cv2.imwrite(save_path, im0)
                            
                    with open(source, "rb") as f:
                        image_string = base64.b64encode(f.read())

                    data = {

                        'code': '1',
                        'image_base64': str(image_string),
                        'explain': str(explain)
                    }
                else:
                    data = {
                        'code': '0',
                        'message': 'Không tìm thấy tổn thương. Hãy thử với loại chẩn đoán khác.'
                    }   
                            
                
                # Save results (image with detections)


        response = app.response_class(response=json.dumps(data),
                    status=200,
                    mimetype='application/json')

        # Trả về kết quả
        return response

if __name__ == '__main__':
    app.run( port='3001', debug=True)