import streamlit as st
import os
import cv2
import os
import str_start
from sahi.predict import AutoDetectionModel, get_sliced_prediction
import pandas as pd
import altair as alt
import numpy as np
from sahi.postprocess.combine import (
    GreedyNMMPostprocess,
    LSNMSPostprocess,
    NMMPostprocess,
    NMSPostprocess,
)


st.set_page_config(layout='wide', page_title='Hack.24 inNINO')

cl = st.sidebar.text_input('введите путь к видео', 'vid/2.mp4')
# st.sidebar.write('Для примера можно ввести vid/2.mp4 или vid/1.mp4')
model = st.sidebar.selectbox(
    'Выберите модель',
    os.listdir(str_start.cl_mod),
    index=None,
    format_func=lambda s: s.upper())
name_res = cl[:len(cl)-4]

@st.cache_data
def prd_yol(model, cl, name , slice_=8000):
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=str_start.cl_mod+'/'+model,
        confidence_threshold=0.01,
        device="cuda:0",  # or 'cuda:0'
    )
    cap = cv2.VideoCapture(cl)
    len_vid = 0
    res_out = []
    while True:
    # for i in range(0,3):
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
                perform_standard_pred=True,
                postprocess_class_agnostic=True,
                postprocess_type='GREEDYNMM'  # 'NMM', 'LSNMS', 'GREEDYNMM' or 'NMS'
            )
            y_r = result.object_prediction_list.copy()
            res_out.append(y_r)
            len_vid+=1
        else:
            break
    return len_vid, res_out

def plot_bb(frame_, res_out, val):
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0)]
    for bb in res_out:
        p0, p1, label = (int(bb.bbox.minx), int(bb.bbox.miny)), \
            (int(bb.bbox.maxx), int(bb.bbox.maxy)), \
            bb.category.name

        if 100 * bb.score.value > val:
            frame_ = cv2.rectangle(frame_, p0, p1, colors[bb.category.id], 3)
            cv2.putText(frame_, str(int(100 * bb.score.value)), p1,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame_, label, (p1[0], p1[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2, cv2.LINE_AA)

    return frame_
        # frame = cv2.rectangle(frame, p0, p1, (255, 255, 0), 10)

def plot_alt(data_):
    xrule = (alt.Chart(pd.DataFrame({'Frame': [sl_1]})).mark_rule(color='red', strokeWidth=2).encode(x='Frame'))
    data_ = pd.DataFrame(data_, columns=['Frame', 'Detection', 'Type'])
    chart = alt.Chart(data_).mark_line().encode(
        x='Frame',
        y='Detection',
        color='Type')
    return chart+xrule

try:
    cap = cv2.VideoCapture(cl)
    st.title('Обнаружение на видеофайле')
    val = st.sidebar.slider('Порог вероятности', 0, 100, 25)
    len_vid, res_out = prd_yol(model, cl, '1', 1920)
    # st.write(res_out)
    len_vid2, res_out2 = prd_yol(model, cl, '2', 640)
    res_out_ = [i + x for i, x in zip(res_out, res_out2)]

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
    for i in range(0, len(res_out_)):
        res_out_[i] = postprocess(res_out_[i])

    sl_1 = st.slider('Текущий кадр', 0, len_vid - 1, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, sl_1)
    res, frame = cap.read()

    col = st.columns(3)
    col[0].write('Кадр с обработкой фрагментов 1280')
    col[1].write('Кадр с обработкой фрагментов 640')
    col[2].write('Кадр с итоговым обнаружением')

    frame_s = plot_bb(frame,res_out[sl_1],val)
    col[0].image(frame_s, channels="BGR")

    frame_b = plot_bb(frame,res_out2[sl_1],val)
    col[1].image(frame_b, channels="BGR")
    frame_a = plot_bb(frame,res_out_[sl_1],val)
    col[2].image(frame_a, channels="BGR")

    data1 = []
    data2 = []
    data_a = []

    for i in range(0, len_vid):
        ids = [0,0,0,0,0,i]

        for data_ in res_out[i]:
            if 100 * data_.score.value > val:
                ids[int(data_.category.id)] += 1
        for type_ in range(0,5):
            # if ids[type_]>0:
                data1.append([i,ids[type_],str(type_)])

        ids = [0, 0, 0, 0, 0, i]

        for data_ in res_out2[i]:
            if 100 * data_.score.value > val:
                ids[int(data_.category.id)] += 1
        for type_ in range(0,5):
            # if ids[type_]>0:
                data2.append([i,ids[type_],str(type_)])

        ids = [0, 0, 0, 0, 0, i]

        for data_ in res_out_[i]:
            if 100 * data_.score.value > val:
                ids[int(data_.category.id)] += 1

        for type_ in range(0, 5):
            # if ids[type_] > 0:
                data_a.append([i, ids[type_], str(type_)])
        # print('alt')
    col[0].write('График обнаружения 1280')
    col[1].write('График обнаружения 640')
    col[2].write('Итоговый график')
    col[0].altair_chart(plot_alt(data1))
    col[1].altair_chart(plot_alt(data2))
    col[2].altair_chart(plot_alt(data_a))





except:
    st.error('Видео или модель не выбраны')


st.title('Обнаружение на кадрах с помощью yolo и детектора движения')
st.write('Для снижения вычислений на каждый кадр предлагается корректировать количесво фрагментов обработки. Для этого '
         'предлагаем использовать детектор движения на основе сравнения соседних кадров с помощью индекса SSIM (клавиша '
         '"Детектор движения") или для работы стационарной камеры по усредненному фону (клавиша "Усреднение фона")')
col1, col2 = st.columns(2)
if col1.button("Детектор движения"):
    os.system("tmp.py")

if col2.button("Усреднение фона"):
    os.system("fon.py")