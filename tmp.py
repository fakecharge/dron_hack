import altair as alt
import pandas as pd
import numpy as np
import cv2
from skimage.metrics import structural_similarity
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict

colors = [(0,0,255), (0,255,0), (255,0,0),(0,255,255),(255,255,0)]
cap = cv2.VideoCapture('vid/2.mp4')
ret, frame_pred = cap.read()
frame_pred = cv2.cvtColor(frame_pred, cv2.COLOR_BGR2GRAY)
detection_model = AutoDetectionModel.from_pretrained(
                    model_type="yolov8",
                    model_path='models/+300n_640_emp.pt',
                    confidence_threshold=0.1,
                    device="cpu",  # or 'cuda:0'
                )

while True:
    ret, frame1 = cap.read()
    res_ = []
    if not ret:
        break
    frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    (score, diff) = structural_similarity(frame, frame_pred, gaussian_weights=True, sigma=3.5,
                                          data_range=355,
                                          use_sample_covariance=True, full=True)
    diff = (diff * 255).astype("uint8")
    ok_, diff = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    frame_pred = frame
    result = get_prediction(frame1, detection_model)
    for bb in result.object_prediction_list:
        res_.append(bb)
    for i in range((-1 * frame.shape[0] // 640 * -1)):
        for j in range((-1 * frame.shape[1] // 640 * -1)):
            x = i*640
            y = j*640
            tmp = diff[y:y+640, x:x+640]
            cnt = cv2.countNonZero(tmp)
            if cnt<10:
                tmp+=125
            else:
                result = get_prediction(frame1[y:y+640, x:x+640], detection_model)
                for bb in result.object_prediction_list:
                    bb.bbox.minx += x
                    bb.bbox.miny += y
                    bb.bbox.maxx += x
                    bb.bbox.maxy += y
                    res_.append(bb)
                    # print(bb)

                # res_.append()
            # print(cnt)
    for bb in res_:
        p0, p1, label = (int(bb.bbox.minx), int(bb.bbox.miny)), \
            (int(bb.bbox.maxx), int(bb.bbox.maxy)), \
            bb.category.name
        frame1 = cv2.rectangle(frame1, p0, p1, colors[bb.category.id], 3)
        cv2.putText(frame1, str(int(100 * bb.score.value)), p1,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame1, label, (p1[0], p1[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2, cv2.LINE_AA)
    diff = cv2.resize(frame1,(1024,720))
    cv2.imshow('1',diff)
    k = cv2.waitKey(1)
    if k == 27:
        break




# ObjectPrediction<
#     bbox: BoundingBox: <(212.4447, 558.1908, 285.6778, 617.2816), w: 73.23309326171875, h: 59.0908203125>,
#     mask: None,
#     score: PredictionScore: <value: 0.42090243101119995>,
#     category: Category: <id: 3, name: 3>>]