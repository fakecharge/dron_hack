import cv2
from sahi.predict import AutoDetectionModel, get_sliced_prediction
# import str_task4
import screeninfo
import time
from threading import Thread


def video_():
    global vid_, frame, image_h, image_w, trol, close_, key_yolo, yolo_res
    vid_ = False
    close_ = False
    key_yolo = False
    window_name = 'inNINO.cam'
    # cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    # screen = screeninfo.get_monitors()[0]
    # cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
    # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        if vid_:
            frame_out = frame.copy()



            if key_yolo:
                y_r = yolo_res.copy()
                for bbox_data in range(0, len(y_r)):
                    tmp = y_r[bbox_data].bbox
                    p0, p1, label = (int(tmp.minx), int(tmp.miny)), (int(tmp.maxx), int(tmp.maxy)), y_r[bbox_data].category.name

                    if 100 * y_r[bbox_data].score.value > 20:
                        # frame = cv2.rectangle(frame, p0, p1, (255, 255, 0), 10)
                        frame_out = cv2.rectangle(frame_out, p0, p1, (0,0,255), 3)
                        cv2.putText(frame_out, str(int(100 * y_r[bbox_data].score.value)), p1,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame_out, label, (p1[0], p1[1] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 2, cv2.LINE_AA)
            if key_yolo:
                cv2.putText(frame_out, 'Y - CNN (yolo)', (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame_out, 'Y - CNN (yolo)', (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_out, 'Esc - quit', (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(window_name, frame_out)
            k = cv2.waitKey(1)
            if (k % 256) == ord('y'):
                key_yolo = not key_yolo
            if k == 27:
                close_ = True
                break



def in_vid():
    global vid_, frame, image_h, image_w, trol, close_
    vid_ = False
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    image_h, image_w, trol = frame.shape
    close_ = False
    while True:
        ret, frame = cap.read()
        if ret:
            vid_ = True

        else:
            print('Error frame')
            time.sleep(1/25)

        if close_:
            break
    cap.release()

def yolo_pred():
    global yolo_res
    yolo_res = []
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='models/600n_640_emp.pt',
        confidence_threshold=0.01,
        device="cpu",  # or 'cuda:0'
    )
    while True:
        if close_:
            break
        if vid_ and (len(frame)>0):  # and num_>10:
            frame_pred = frame.copy()
            result = get_sliced_prediction(
                frame_pred,
                # "tmp/real_sob.png",
                detection_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.30,
                overlap_width_ratio=0.0,
                perform_standard_pred=False,
                postprocess_class_agnostic=True,
                postprocess_type='GREEDYNMM' # 'NMM', 'LSNMS', 'GREEDYNMM' or 'NMS'
            )
            time.sleep(1 / 40)
            yolo_res = result.object_prediction_list.copy()
            # print (yolo_res)


if __name__ == "__main__":
    Thread(target=in_vid, name='frame').start()
    Thread(target=video_, name='video', daemon=True).start()
    Thread(target=yolo_pred, name='yolo', daemon=True).start()