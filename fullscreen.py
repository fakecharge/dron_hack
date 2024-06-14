import socket
import struct
import ptz_control
import cv2
import time
from threading import Thread
import screeninfo
import numpy as np
from sahi.predict import AutoDetectionModel, get_sliced_prediction
import argparse


global key_pov, close_, ret, key_rec, vid_, file_, yolo_res
az_v = 0
az_o = 0
r = 8.73
global ret, frame, image_h, image_w, trol
# cap_adr = "rtsp://admin:123asdASD@192.1.1.163:554/Streaming/channels/103/"

key_pov = True
ptz = ptz_control.ptzControl()


def fromO_V(koor_v):
    dp = 360.0 - koor_v
    while dp > 360.0:
        dp = dp - 360.0
    while dp < 0.0:
        dp = dp + 360.0
    Av = 5.
    Ao = 164.43 + 216
    # Av = 191.2-180
    # Ao = 5.68
    # AoPoChas = 360-Ao
    dAoIst = 180 - Ao - Av
    if dAoIst < 0:
        dAoIst += 360
    dp -= dAoIst
    if dp < 0:
        dp += 360
    return dp, dAoIst

def prvd_r (azimut, um, d = 10000, alph = 185):
    xoes = r * np.cos(np.deg2rad(alph))
    yoes = r * np.sin(np.deg2rad(alph))
    Ltargt = d * np.cos(np.deg2rad(um))
    xtarg = Ltargt * (np.cos(np.deg2rad(azimut)))
    ytarg = Ltargt * (np.sin(np.deg2rad(azimut)))

    x_or = xtarg - xoes
    y_or = ytarg - yoes

    if x_or == 0:
        x_or = 0.000001

    if y_or == 0:
        y_or = 0.000001

    if y_or >= 0:
        if x_or >= 0:
            res = np.rad2deg(np.arctan(y_or / x_or))
        else:
            res = np.rad2deg(np.arctan(abs(x_or / y_or))) + 90.0
    else:
        if x_or <= 0:
            res = np.rad2deg(np.arctan(abs(y_or / x_or))) + 180.0
        else:
            res = np.rad2deg(np.arctan(abs(x_or / y_or))) + 270.0
    return res #+ 16 надо проверить +alph!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def perevod(koor, um, d = 10000):
    global key_pov


    # dp = az_o + az_v - koor #- np.rad2deg(np.arcsin(r/d))
    dp = koor
    while dp > 360.0:
        dp = dp - 360.0
    while dp < 0.0:
        dp = dp + 360.0

    koor_v = dp / 180 - 1.0
    koor_u = (um) / 45 -1.0
    kk = ptz.status()
    # rt = kk.Position.PanTilt.y
    zoom = kk.Position.Zoom.x
    ptz.move_abspantilt(koor_v, koor_u, zoom, 1)
    return koor_v, koor_u

def server():
    global close_, key_pov, dal_, n_cel_
    n_cel_ = 0
    dal_ = 10000
    close_ = False
    localIP = "192.1.1.2"
    localPort = 7803
    bufferSize = 1024
    UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    print('bind')
    UDPServerSocket.bind((localIP, localPort))

    delta_pred = 0.
    pred_d = [0., 0.]
    zoom = 0
    cmr = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    while (True):


            bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
            message = bytesAddressPair[0]
            # x = []
            real = [0. for j in range(15)]
            print(message, len(message))
            x = struct.unpack(">15h", message)

            for i in range(0, len(x)):
                real[i] = float(x[i]*cmr[i])

            if pred_d[0] != real[1]:
                pred_d[0] = real[1]
                pred_d[1] = real[2]
                delta_pred = 0.5

            delta_pred -= pred_d[1] - real[2]
            delta_pred /= 2
            pred_d[1] = real[2]
            dal_ = real[3]
            n_cel_ = real[1]
            x_ = real[2] + delta_pred

            print('X = '+str(x_)+' real = '+str(real[2])+'  Delta ='+str(delta_pred))
            if close_:
                break
            if key_pov:
                x_d, d = fromO_V(real[2])
                new_x = prvd_r(x_d, real[4], d=real[3], alph=185)
                perevod(new_x, real[4], zoom)

def video():
    global vid_, frame, image_h, image_w, trol, close_, key_rec, file_, yolo_res
    vid_ = False

    global key_pov, zoom_m
    window_name = 'Hack_DRON'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)

    screen = screeninfo.get_monitors()[0]
    cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    #mouse
    def mouse_circle(event, x, y, flags, param):
        global mouse, track_, zoom_m
        # zoom_m = 0.0
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse = x, y
            track_ = True

        global mouse_move_, pan, tilt
        if event == cv2.EVENT_RBUTTONDOWN:
            mouse_move_ = True
            pan = (x - 960) / 960
            tilt = -(y - 540) / 540
                # print(pan, tilt)
        if event == cv2.EVENT_RBUTTONUP:
            # cv2.Even
            mouse_move_ = False
        if event ==cv2.EVENT_MOUSEWHEEL:
            k_k = ptz.status()
            zo_p = k_k.Position.Zoom.x
            if flags>0:
                zo_p += 0.1
                # print(x,y,flags,param)
                if zoom_m > 1. and zoom_m < 1.69:
                    zoom_m+=0.1
                if zo_p>1. and zoom_m<=1.:
                    zoom_m = zo_p
                if zoom_m<=1:
                    ptz.movePTZ(0, 0, 0.1, 0.1)
            else:
                zo_p -= 0.1
                # print(x,y,flags,param)
                if zoom_m > 1. :
                    zoom_m -=0.1
                if zo_p > 0.:
                    zo_p = 0.
                if zoom_m<=1.:
                    ptz.movePTZ(0, 0, -0.1, 0.1)




            # print(mouse)
    cv2.setMouseCallback(window_name, mouse_circle)



    global mouse, track_, mouse_move_, pan, tilt
    mouse_move_ = False
    track_ = False
    mouse = (0,0)
    zoom_m = 0.0
    pan = 0.
    tilt = 0.
    tracker = cv2.TrackerMIL.create()
    ptz = ptz_control.ptzControl()
    k_k = ptz.status()

    clahe = cv2.createCLAHE(clipLimit=6, tileGridSize=(6, 6))
    mask = np.zeros((1080, 1920), dtype="uint8")
    cv2.circle(mask, (int(1920 / 2), int(1080 / 2)), 300, 255, -1)


    frame_count = 0
    time_ = time.strftime("%Y%m%d-%H%M%S")

    # f = open('D:\\' + time_ + '.txt', 'w')
    # f.write(str(az_v)+';'+str(az_o)+'\n')

    p_st = 0.
    t_st = 0.
    global file_
    kontrast_ = False
    file_ = True
    ok1 = False
    mov = False
    move_tilt = False
    key_rec = False
    key_yolo = False
    move1 = False
    frame_ = 0
    num_f = 1
    soprov = 201
    zoom_rezh = [[0.0160, 1.1], [0.3, 1.], [0.5, 1.], [0.7, 1.1], [1., 1.]]



    def detect_mov(img):
        _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
        rect_ = (0, 0, 0, 0)
        dilated = cv2.dilate(thresh, None, iterations=3)
        сontours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in сontours:

            if cv2.contourArea(contour) < 2 and cv2.contourArea(contour) >100:
                print(cv2.contourArea(contour))
                continue

            rect_tmp = cv2.boundingRect(contour)
            if (rect_tmp[0] < mouse[0] and rect_tmp[0]+rect_tmp[2] > mouse[0]) and (rect_tmp[1] < mouse[1] and rect_tmp[1]+rect_tmp[3] > mouse[1]):
                rect_ = rect_tmp
        return rect_

    def kontrast(img, mask):
        b, g, r = cv2.split(img)
        masked = cv2.bitwise_and(b, b, mask=mask)
        image_clahe_b = clahe.apply(masked)
        masked = cv2.bitwise_and(g, g, mask=mask)
        image_clahe_g = clahe.apply(masked)
        masked = cv2.bitwise_and(g, g, mask=mask)
        image_clahe_r = clahe.apply(masked)

        image_clahe = cv2.merge([image_clahe_b, image_clahe_g, image_clahe_r])
        mask = 255 - mask
        tmp = cv2.bitwise_and(img, img, mask=mask)
        # vis = np.concatenate((image, cv2.addWeighted(tmp, 1.0, image_clahe, 1.0, 0.0)), axis=1)
        return cv2.addWeighted(tmp, 1.0, image_clahe, 1.0, 0.0)



    while True:
        if vid_:
            frame_out = frame.copy()
            frame1 = frame.copy()
            if zoom_m < 0:
                zoom_ = 0

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

            if frame_ >= num_f and ok1:
                ok, bbox = tracker.update(frame_out)
                print(ok, bbox)
                t_s = 0.
                p_s = 0.
                polov = int(bbox[1] + bbox[3] / 2)
                if (polov > 1.05 * image_h / 2) or (polov < 0.95 * image_h / 2):
                    t_s = -1.5*(polov - image_h / 2) / image_h
                    move_tilt = True

                else:
                    if mov == True:
                        # ptz.stop()
                        move_tilt = False

                polov = int(bbox[0] + bbox[2] / 2)
                # print(polov)
                if (polov > 1.05 * image_w/ 2) or (polov < 0.95 * image_w / 2):
                    p_s = 1.5*(polov - image_w / 2) / image_w
                    mov = True
                else:
                    if mov == True:
                        # ptz.stop()
                        mov = False
                if p_s< 0.2 and p_s > 0.02:
                    p_s = 0.2
                if t_s< 0.2 and t_s>0.02:
                    t_s = 0.2
                if mov or move_tilt:
                    if abs(p_s - p_st) > 0.05 or abs(t_s - t_st) > 0.15:
                        ptz.move_tw(p_s, t_s)
                        p_st = p_s
                        t_st = t_s
                        str_ = str(p_s)+' ; '+str(t_s)
                        cv2.putText(frame_out, str_, (640, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 0), 2, cv2.LINE_AA)

                else:
                    ptz.stop()

                frame_ = 0




            if frame_ >= num_f:
                frame_ = 0

            zo = k_k.Position.Zoom.x

            if zo < zoom_rezh[2][0]:
                zoom_ = 1
            elif zo < zoom_rezh[3][0]:
                zoom_ = 2
            else:
                zoom_ = 3

            frame_ += 1

            if ok1:
                (x, y, w, h) = [int(v) for v in bbox]
                cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)


            frame_count +=1
            if frame_count >= 40:
                k_k = ptz.status()
                # time_ = time.strftime("%Y%m%d-%H%M%S")
                # koor_v = (k_k.Position.PanTilt.x + 1.0) * 180
                # dp = fromO_V(koor_v)
                # f.write(time_ + ';' + str(k_k.Position.PanTilt.x) + ';' + str(k_k.Position.PanTilt.y) + ';'
                #         + str(k_k.Position.Zoom.x)+ ';' + str(dp)+'\n')
                frame_count = 0

                # print (k_k)
            if zoom_m>1.:
                w_m = int(image_w * (2 - zoom_m))
                h_m = int(image_h * (2 - zoom_m))
                frame_out = frame_out[ int((image_h - h_m) / 2):int((image_h - h_m) / 2 +h_m),int((image_w - w_m) / 2):int((image_w - w_m) / 2 +w_m)]
                frame_out = cv2.resize(frame_out, (1920,1080))
                # cv2.resize(frame_out[int((image_w - w_m) / 2):int((image_w - w_m) / 2 +w_m), int((image_h - h_m) / 2):int((image_h - h_m) / 2 +h_m)], frame_out, frame1.size())
            if kontrast_:
                frame_out = kontrast(frame_out, mask)

            if key_rec:
                    cv2.putText(frame_out, 'REC', (1820, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2, cv2.LINE_AA)

            cv2.line(frame_out, (int(image_w / 2), int(image_h / 2 - 25)), (int(image_w / 2), int(image_h / 2 + 350)),
                     (125, 125, 125), 1)
            cv2.line(frame_out, (int(image_w / 2 - 25), int(image_h / 2)), (int(image_w / 2 + 25), int(image_h / 2)),
                     (125, 125, 125), 1)

            k_k = ptz.status()
            zo = k_k.Position.Zoom.x
            str_ = 'zoom = ' + str(k_k.Position.Zoom.x)
            cv2.putText(frame_out, str_, (200, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 3, cv2.LINE_AA)
            koor_v = (k_k.Position.PanTilt.x + 1.0) * 180
            dp2 = (k_k.Position.PanTilt.y + 1.0) * 45

            dp, d = fromO_V(koor_v)


            new_x = prvd_r(dp, dp2, d=800, alph=5) #alph=164.43

            str_ = 'azimutO = ' + str(koor_v)
            # print(koor_v)
            cv2.putText(frame_out, str_, (1600, 900), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 3, cv2.LINE_AA)



            str_ = 'Ugol = ' + str(int(dp2))
            cv2.putText(frame_out, str_, (1600, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(frame_out, 'W,S - verh/niz', (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame_out, 'A,D - pravo/levo', (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame_out, 'X - peredat cel', (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame_out, 'C - sbros skorosti/priem koor', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame_out, 'K - kontrast', (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame_out, 'M - otmena/priem koor', (0, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame_out, 'R - record video', (0, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame_out, 'Y - CNN (yolo)', (0, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2, cv2.LINE_AA)

            if key_pov:
                cv2.putText(frame_out, 'Priem', (0, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 3, cv2.LINE_AA)
            # frame_out = cv2.resize(frame_out,(640,480))
            cv2.imshow(window_name, frame_out)
            k = cv2.waitKey(1)

            # print(ret)
            if k == 27:
                close_ = True
                break

            if (k % 256) == ord('r'):  # record
                key_rec = not (key_rec)
                if not(key_rec):
                    file_ = not (file_)


            if (k % 256) == ord('a'):  # left
                # tilt = 0
                key_pov = False
                pan -= 0.1
                if pan > -0.1 and pan < 0.1:
                    pan = -0.1
                else:
                    # ptz.move_pan(pan)
                    move1 = True
                # ptz.move_pan(pan)

            if (k % 256) == ord('d'):  # right
                # tilt = 0
                key_pov = False
                pan += 0.1
                if pan > -0.1 and pan < 0.1:
                    pan = 0.1
                else:
                    # ptz.move_pan(pan)
                    move1 = True

            if (k % 256) == ord('c'):  # stop all
                ptz.stop()
                move1 = False
                tilt = 0.
                pan = 0.
                ok1 = False
                key_pov = True

            if (k % 256) == ord('e'):

                ptz.movePTZ(0, 0, zoom_rezh[zoom_ + 1][0], zoom_rezh[zoom_ + 1][1])

            if (k % 256) == ord('q'):

                ptz.movePTZ(0, 0, -zoom_rezh[zoom_ - 1][0], zoom_rezh[zoom_ - 1][1])


            if (k % 256) == ord('h'):
                ptz.move_tw(0.3, 0.1)
            if (k % 256) == ord('k'):
                kontrast_ = not kontrast_
            if (k % 256) == ord('m'):
                key_pov = not key_pov

            if (k % 256) == ord('y'):
                key_yolo = not key_yolo

            if (k % 256) == ord('w'):  # up
                # pan = 0
                key_pov = False
                tilt += 0.1
                if tilt > -0.1 and tilt < 0.1:
                    tilt = 0.1
                else:
                    # ptz.move_tilt(tilt)
                    move1 = True

            if (k % 256) == ord('s'):  # down
                # pan = 0
                key_pov = False
                tilt -= 0.1
                if tilt > -0.1 and tilt < 0.1:
                    tilt = -0.1
                else:
                    # ptz.move_tilt(tilt)
                    move1 = True
            if move1 or mouse_move_:
                if pan > 1.0:
                    pan = 1.
                if pan < -1.:
                    pan = -1
                if tilt > 1.0:
                    tilt = 1.
                if tilt < -1.:
                    tilt = -1
                ptz.move_tw(pan, tilt)
                key_pov = False
                move1 = False



            if k == 32 or track_:
                ptz.stop()
                key_pov = False
                tilt = 0.
                pan = 0.
                move1 = False
                diff = cv2.absdiff(frame1, frame2)
                gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (3, 3), 0)
                bbox = detect_mov(blur)
                if bbox[0] != 0:
                    ok = tracker.init(frame1, bbox)
                    ok1 = True
                else:
                    ok1 = False
                print(ok1)
                track_ = False

            if frame_count >= 39 and ok1:
                localPort = 7804
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                if n_cel_ > 0:
                    n_sopr = n_cel_
                else:
                    n_sopr = soprov
                    soprov += 1
                msg = struct.pack(">xbbbbbhhhhhhh", 1, int(time.strftime("%H")), int(time.strftime("%M")), int(time.strftime("%S")), 0, int(new_x),  int(dal_), int(dp2), 1, int(n_sopr), 201, int(10))
                localIP = '<broadcast>'
                sock.sendto(msg, (localIP, localPort))

            frame2 = frame1.copy()
    cv2.destroyAllWindows()

def in_vid():
    global vid_, frame, image_h, image_w, trol, close_
    vid_ = False
    cap = cv2.VideoCapture()
    cap.open(cap_adr)
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
            cap.open(cap_adr)

        if close_:
            break
    cap.release()

def record():
    global frame, key_rec, file_, image_w, image_h, close_
    close_ = False
    key_rec = False
    file_ = True
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    while True:
        if close_:
            break
        if key_rec:
            frame_rec = frame.copy()
            time_ = time.strftime("%Y%m%d-%H%M%S")
            if file_:
                vid = cv2.VideoWriter('tmp/' + str(time_) + '.avi', fourcc, 25, (image_w, image_h))
                file_ = not (file_)
            time_ = time.strftime("%H:%M:%S")
            cv2.putText(frame_rec, str(time_), (1800, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            vid.write(frame_rec)
            time.sleep(1/25)


        else:
            time.sleep(1)
    vid.release()


def yolo_pred():
    global yolo_res
    yolo_res = []
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='models/600n_640_emp.pt',
        confidence_threshold=0.25,
        device="cpu",  # or 'cuda:0'
    )
    while True:
        if close_:
            break
        if vid_ and (len(frame)>0):  # and num_>10:
            frame_pred = frame.copy()
            result = get_sliced_prediction(
                frame_pred,
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-cam", default="rtsp://admin:123asdASD@192.1.1.163:554/Streaming/channels/103/",
                        help="IP camera")
    parser.add_argument("-det", default="192.1.1.2", help="IP detector")
    args = parser.parse_args()
    cap_adr = args.cam


    Thread(target=in_vid, name='frame').start()
    Thread(target=video, name='video').start()
    Thread(target=record, name='record', daemon=True).start()
    Thread(target=server, name='server', daemon=True).start()
    Thread(target=yolo_pred, name='yolo', daemon=True).start()
