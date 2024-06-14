import streamlit as st
import os
import cv2
import os
import str_start
from sahi.predict import AutoDetectionModel, get_sliced_prediction
import pandas as pd
import altair as alt
import numpy as np


st.set_page_config(layout='wide', page_title='Hack.23 inNINO')

def save_uploadedfile(uploadedfile):
    with open(os.path.join("vid", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.sidebar.success("Файл загружен: "+uploadedfile.name+".".format(uploadedfile.name))

st.header('Обнаружение по видео')


datafile = st.sidebar.file_uploader("Upload")
if datafile is not None:
   file_details = {"FileName":datafile.name,"FileType":datafile.type}
   save_uploadedfile(datafile)

# def save_uploadedfile(uploadedfile):
    # with open(os.path.join("vid_input", uploadedfile.name), "wb") as f:
        # f.write(uploadedfile.getbuffer())
    # return st.sidebar.success("Файл загружен: "+uploadedfile.name+".".format(uploadedfile.name))
# cl = os.listdir('vid_input')
# vid_type = st.sidebar.selectbox(
#     'Выберите видео',
#     cl,
#     index=1,
#     format_func=lambda s: s.upper())

# datafile = st.sidebar.file_uploader("Upload")
# if datafile is not None:
   # file_details = {"FileName":datafile.name,"FileType":datafile.type}
   # save_uploadedfile(datafile)

cl = st.sidebar.text_input('введите путь к видео', 'vid')
model = st.sidebar.selectbox(
    'Выберите модель',
    os.listdir(str_start.cl_mod),
    index=None,
    format_func=lambda s: s.upper())
name_res = cl[:len(cl)-4]

@st.cache_data
def prd_yol(model, cl, name , slice_=8000):
    # model = YOLO(str_task4.cl_mod+'/'+model)    yolo_res = []
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=str_start.cl_mod+'/'+model,
        confidence_threshold=0.01,
        device="cpu",  # or 'cuda:0'
    )
    print(str_start.cl_mod + '/' + model)
    cap = cv2.VideoCapture(cl)
    ret, frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    print (fps)
    vid_res = cv2.VideoWriter(name_res+name+'_res.avi', fourcc, fps, (frame.shape[1], frame.shape[0]))
    len_vid = 0
    res_out = []
    # while True:
    for i in range(0,20):
        ret, frame =cap.read()
        if ret:
            result = get_sliced_prediction(
                frame,
                # "tmp/real_sob.png",
                detection_model,
                slice_height=slice_,
                slice_width=slice_,
                overlap_height_ratio=0.10,
                overlap_width_ratio=0.10,
                perform_standard_pred=False,
                postprocess_class_agnostic=True,
                postprocess_type='GREEDYNMM'  # 'NMM', 'LSNMS', 'GREEDYNMM' or 'NMS'
            )
            y_r = result.object_prediction_list.copy()
            res_out.append(y_r)
            for bbox_data in range(0, len(y_r)):
                tmp = y_r[bbox_data].bbox
                p0, p1, label = (int(tmp.minx), int(tmp.miny)), (int(tmp.maxx), int(tmp.maxy)), y_r[
                    bbox_data].category.name

                if 100 * y_r[bbox_data].score.value > 25:
                    # frame = cv2.rectangle(frame, p0, p1, (255, 255, 0), 10)
                    frame = cv2.rectangle(frame, p0, p1, (0, 0, 255), 3)
                    cv2.putText(frame, str(int(100 * y_r[bbox_data].score.value)), p1,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, label, (p1[0], p1[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2, cv2.LINE_AA)

            vid_res.write(frame)
            len_vid+=1
        else:
            vid_res.release()
            break
    return len_vid, res_out
try:
    len_vid, res_out = prd_yol(model, cl, '1')
    len_vid2, res_out2 = prd_yol(model, cl, '2', 640)
    cap = cv2.VideoCapture(name_res+'1_res.avi')
    cap2 = cv2.VideoCapture(name_res + '2_res.avi')

    sl_1 = st.slider('Текущий кадр '+ name_res+'_res.avi', 0, len_vid-1, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, sl_1)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, sl_1)
    res, frame = cap.read()
    res, frame2 = cap2.read()
    col1,col2 = st.columns(2)
    col1.image(frame, channels="BGR")
    col2.image(frame2, channels="BGR")
    # st.write(res_out)
    data1 = [[]]
    data2 = [[]]
    for i in range(0, len_vid):
        ids1 = [0,0,0,0,0,i]

        for data_ in res_out[i]:
            ids1[int(data_.category.id)] += 1
        for type_ in range(0,4):
            if ids1[type_]>0:
                data1.append([i,ids1[type_],str(type_)])
        # data1.append(ids1)

        ids2 = [0, 0, 0, 0, 0,i]

        for data_ in res_out2[i]:
            ids2[int(data_.category.id)] += 1
        for type_ in range(0,4):
            if ids2[type_]>0:
                data2.append([i,ids2[type_],str(type_)])

        # data2.append(ids2)
    # base = alt.Chart()
    # line = [[0, sl_1],[9, sl_1]]
    # line = pd.DataFrame(line, columns=['sl','num'])
    # st.write(line)
    xrule = (alt.Chart(pd.DataFrame({'Frame': [sl_1]})).mark_rule(color='red', strokeWidth=2).encode(x='Frame'))
    # xrule = (alt.Chart().mark_rule(color='cyan').encode(x = alt.datum(sl_1)))
    # line_ = alt.Chart(line).mark_line().encode(
    #     x = 'num',
    #     y = 'sl')
    # st.altair_chart(line_)
    data1=pd.DataFrame(data1[1:], columns=['Frame', 'Detection', 'Type'])
    chart = alt.Chart(data1).mark_line().encode(
            x = 'Frame',
            y = 'Detection',
            color = 'Type')


    # data1=pd.DataFrame(data1[1:], columns=['dron', 'plain', 'bird', 'bpla', 'heli','num'])
    #
    # chart = alt.Chart(data1).mark_line().encode(
    #     x = 'num',
    #     y = alt.Y(alt.repeat('layer'), aggregate='mean', title = 'predictions'),
    #     color = alt.datum(alt.repeat('layer'))
    # ).repeat(layer=['dron', 'plain', 'bird', 'bpla', 'heli'])

    # chart = alt.layer(
    #     chart, xrule
    # ).properties(
    #     title='Factors that Contribute to Life Expectancy in Malaysia',
    #     width=500, height=300
    # )

    col1.altair_chart(chart+xrule,use_container_width=True)

    # data2 = pd.DataFrame(data2[1:], columns=['dron', 'plain', 'bird', 'bpla', 'heli', 'num'])
    data2=pd.DataFrame(data2[1:], columns=['Frame', 'Detection', 'Type'])
    chart = alt.Chart(data2).mark_line().encode(
            x = 'Frame',
            y = 'Detection',
            color = 'Type')

    # chart = alt.Chart(data2).mark_line().encode(
    #     x='num',
    #     y=alt.Y(alt.repeat('layer'), aggregate='mean', title='predictions'),
    #     color=alt.datum(alt.repeat('layer'))
    # ).repeat(layer=['dron', 'plain', 'bird', 'bpla', 'heli'])
    col2.altair_chart(chart+xrule, use_container_width=True)
    col1.write(data1)   # res_out[i]
    col2.write(data2)

except:
    st.error('Видео не найдено')
# result = model.predict(cl)



# "ObjectPrediction<
#     bbox: BoundingBox: <(32.8602294921875, 75.12236022949219, 60.92374038696289, 98.1712875366211), w: 28.06351089477539, h: 23.048927307128906>,
#     mask: None,
#     score: PredictionScore: <value: 0.2737981975078583>,
#     category: Category: <id: 0, name: 0>>"