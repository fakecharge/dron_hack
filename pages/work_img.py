import os
import zipfile

import streamlit as st
import cv2
import str_start
from sahi.predict import AutoDetectionModel, get_sliced_prediction
import numpy as np

from sahi.postprocess.combine import (    GreedyNMMPostprocess,
    LSNMSPostprocess,    NMMPostprocess,
    NMSPostprocess,    PostprocessPredictions,
)

def write_t(text):
    st.markdown(f'<p style="font-size:20px; border: 2px solid #27ae60; padding : 10px; border-radius: 10px;">{text}</p>',
                unsafe_allow_html=True)
@st.cache_data
def prd_yol(model, img , slice_=8000):
    # model = YOLO(str_task4.cl_mod+'/'+model)    yolo_res = []
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=str_start.cl_mod+'/'+model,
        confidence_threshold=0.25,
        device="cpu",  # or 'cuda:0'
    )
    res_out = []

    result = get_sliced_prediction(
                img,
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
    return res_out

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

st.title('Обнаружение по фото')

st.header('Правила пользования')
write_t('Выберете изображения, которые хотите запустить для инференса')
write_t('Выберете модель, на которой будет происходить инференс (+300s_640_emp.pt - наша итоговая модель)')
write_t('После выполнения инференса, появится кнопка для скачивания архива с результатами распознавания')
write_t('Результаты распознавания представлены в формате YOLO (id;x_center;y_center;w;h)')

debug = st.checkbox('Вывод результата обнаружения', value=False)

datafiles = st.sidebar.file_uploader("Upload", accept_multiple_files=True, type=['jpg', 'png'])
model = st.sidebar.selectbox(
    'Выберите модель',
    os.listdir(str_start.cl_mod),
    index=None,
    format_func=lambda s: s.upper())


results = []
if datafiles is not None and len(datafiles) > 0:

    prg = st.progress(0)

    count_progress = 0
    per_sec = (100 / len(datafiles)) / 100
    for nextFile in datafiles:
        namefile = nextFile.name
        namefile = namefile[:len(namefile) - 4]

        bytes_data = nextFile.getvalue()

        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        try:
            res_out = prd_yol(model, img, slice_=1280)

            res_out_2 = prd_yol(model, img, 640)

            res_out_ = [i + x for i, x in zip(res_out, res_out_2)]
            POSTPROCESS_NAME_TO_CLASS = {"GREEDYNMM": GreedyNMMPostprocess,
                                         "NMM": NMMPostprocess, "NMS": NMSPostprocess,
                                         "LSNMS": LSNMSPostprocess, }
            postprocess_constructor = POSTPROCESS_NAME_TO_CLASS['GREEDYNMM']
            postprocess = postprocess_constructor(
                match_threshold=0.5, match_metric="IOS",
                class_agnostic=True, )
            for i in range(0, len(res_out_)):
                res_out_[i] = postprocess(res_out_[i])

            if debug:
                frame = plot_bb(img, res_out_[0], 25)
                st.image(frame)
                st.write(res_out[0])
            results.append([f'{namefile}.txt', res_out_, img.shape])

            # st.write(namefile)
            # st.image(img)
        except:
            pass

        count_progress += per_sec
        count_progress = min(count_progress, 1.0)
        prg.progress(count_progress)
        # st.write(namefile)
        # st.image(img)

if os.path.exists("archive.zip"):
    os.remove("archive.zip")


for result in results:
    with zipfile.ZipFile('archive.zip', 'a',
                         compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as zf:
        with open('test.txt', 'w') as f:
            img_shape = result[2]
            for j in result[1][0]:
                box = j.bbox
                id_class = j.category.id
                w = (box.maxx - box.minx) / img_shape[1]
                h = (box.maxy - box.miny) / img_shape[0]
                x = (box.minx + w / 2) / img_shape[1]
                y = (box.miny + h / 2) / img_shape[0]
                f.write(f'{int(id_class)};{x};{y};{w};{h}\n')

        zf.write("test.txt", arcname=f"{str(result[0])}")
if os.path.exists("test.txt"):
    os.remove('test.txt')

if os.path.exists("archive.zip"):
    with open('archive.zip', 'rb') as f:
       if st.sidebar.download_button('Download results', f, file_name='results.zip'):
           st.sidebar.success('Thanks for downloading!')

