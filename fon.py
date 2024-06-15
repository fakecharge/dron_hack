import cv2
from skimage.metrics import structural_similarity
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
import numpy as np

from sahi.postprocess.combine import (
    GreedyNMMPostprocess,
    LSNMSPostprocess,
    NMMPostprocess,
    NMSPostprocess,
    PostprocessPredictions,
)

colors = [(0,0,255), (0,255,0), (255,0,0),(0,255,255),(255,255,0)]
cap = cv2.VideoCapture('vid/1.mp4')
# fore = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
# while True:
#     ret, image = cap.read()
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     mask = fore.apply(image)
#     diff = cv2.resize(mask, (1024, 720))
#     cv2.imshow('1', mask)
#     k = cv2.waitKey(100)
#     if k == 27:
#         break
# Randomly select 25 frames
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=30)
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
medianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

# medianFrame = cv2.resize(medianFrame, (1024, 720))
# cv2.imshow('1', medianFrame)
# cv2.waitKey(100)

detection_model = AutoDetectionModel.from_pretrained(
                    model_type="yolov8",
                    model_path='models/+300n_640_emp.pt',
                    confidence_threshold=0.25,
                    device="cuda:0",  # or 'cuda:0'
                )
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while True:
    ret, frame1 = cap.read()
    res_ = []
    if not ret:
        break
    frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    (score, diff) = structural_similarity(frame, medianFrame, gaussian_weights=True, sigma=3.5,
                                          data_range=255,
                                          use_sample_covariance=True, full=True)
    diff = (diff * 255).astype("uint8")
    ok_, diff = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
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
                frame1[y:y+640, x:x+640]*=0
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
    POSTPROCESS_NAME_TO_CLASS = {
        "GREEDYNMM": GreedyNMMPostprocess,
        "NMM": NMMPostprocess,
        "NMS": NMSPostprocess,
        "LSNMS": LSNMSPostprocess,
    }
    postprocess_constructor = POSTPROCESS_NAME_TO_CLASS['GREEDYNMM']
    postprocess = postprocess_constructor(
        match_threshold=0.5,
        match_metric="IOS",
        class_agnostic=True,
    )
    res_=postprocess(res_)
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
    tmp = cv2.split(frame1)
    frame_out = cv2.merge([tmp[0], tmp[1]+ 100 * diff, tmp[2] ])
    frame_out = cv2.resize(frame_out,(1024,720))
    cv2.imshow('Fon',frame_out)
    k = cv2.waitKey(1)
    if k == 27:
        break
