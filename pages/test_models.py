import streamlit as st
import os
import time
from ultralytics import YOLO
import str_start

st.set_page_config(layout='wide', page_title='Hack.24 inNINO')
test_dir = st.sidebar.text_input('Введите путь к файлу .yaml', 'data.yaml')
st.header('Тест обученных моделей')
st.write('Сравнивались различные модели, обученные на аугментированных исходных данных - 414 изображениях.'
         'лучше всего по характеристикам показали себя yolov5n(30n5-cars.pt) и yolov8n(30n-cars_01.pt). '
         'Можно увидеть результаты при схожем обучении yolov3 (30n3-cars.pt) и Yolov6 (30n6-cars.pt).'
         'Есть возможность добавить обученный RTDETR-L в рассмотрение с помощью выбора на боковом меню')
# test_dir = 'data.yaml'
col1, col2, col3 = st.columns(3)
model1 = col1.selectbox(
    'Выберите модель 1',
    os.listdir(str_start.cl_mod),
    index=None,
    format_func=lambda s: s.upper())

model2 = col2.selectbox(
    'Выберите модель 2',
    os.listdir(str_start.cl_mod),
    index=None,
    format_func=lambda s: s.upper())

model3 = col3.selectbox(
    'Выберите модель 3',
    os.listdir(str_start.cl_mod),
    index=None,
    format_func=lambda s: s.upper())
# print(model1)
col = []
col = st.columns(8)
col[0].write('Name')
col[1].write('Fitness')
col[2].write('Pressision')
col[3].write('Recall')
col[4].write('mAp 50')
col[5].write('mAp')
col[6].write('Время выполнения, сек')
col[7].write('Confusion matrix')

@st.cache_data
def print_mod1(model):
    col = st.columns(8)
    start_time = time.time()
    model_1 = YOLO(str_start.cl_mod+'/'+model)
    result = model_1.val(data=test_dir, imgsz=640, batch=16, conf=0.25, iou=0.6, device='cpu', save=False)
    col[0].write(model)
    col[1].write(round(result.fitness, 6))
    k = result.mean_results()
    col[2].write(round(k[0], 6))
    col[3].write(round(k[1], 6))
    col[4].write(round(k[2], 6))
    col[5].write(round(k[3], 6))
    result.confusion_matrix.plot(normalize=False, save_dir='tmp', names=(), on_plot=None)
    col[6].write(round(time.time() - start_time, 6))
    col[7].image('tmp/confusion_matrix.png')

@st.cache_data
def print_mod2(model):
    col = st.columns(8)
    start_time = time.time()
    model_1 = YOLO(str_start.cl_mod+'/'+model)
    result = model_1.val(data=test_dir, imgsz=640, batch=16, conf=0.25, iou=0.6, device='cpu', save=False)
    col[0].write(model)
    col[1].write(round(result.fitness, 6))
    k = result.mean_results()
    col[2].write(round(k[0], 6))
    col[3].write(round(k[1], 6))
    col[4].write(round(k[2], 6))
    col[5].write(round(k[3], 6))
    result.confusion_matrix.plot(normalize=False, save_dir='tmp', names=(), on_plot=None)
    col[6].write(round(time.time() - start_time, 6))
    col[7].image('tmp/confusion_matrix.png')

@st.cache_data
def print_mod3(model):
    col = st.columns(8)
    start_time = time.time()
    model_1 = YOLO(str_start.cl_mod+'/'+model)
    result = model_1.val(data=test_dir, imgsz=640, batch=16, conf=0.25, iou=0.6, device='cpu', save=False)
    col[0].write(model)
    col[1].write(round(result.fitness, 6))
    k = result.mean_results()
    col[2].write(round(k[0], 6))
    col[3].write(round(k[1],6))
    col[4].write(round(k[2],6))
    col[5].write(round(k[3],6))
    result.confusion_matrix.plot(normalize=False, save_dir='tmp', names=(), on_plot=None)
    col[6].write(round(time.time() - start_time,6))
    col[7].image('tmp/confusion_matrix.png')

# @st.cache_data
# def print_detr():
#     col = st.columns(8)
#     start_time = time.time()
#     model = 'detr.pt'
#     model_1 = RTDETR(model)
#     result = model_1.val(data=test_dir, imgsz=640, batch=16, conf=0.1, iou=0.6, device='cpu', save=False)
#     col[0].write(model)
#     col[1].write(round(result.fitness, 6))
#     k = result.mean_results()
#     col[2].write(round(k[0], 6))
#     col[3].write(round(k[1],6))
#     col[4].write(round(k[2],6))
#     col[5].write(round(k[3],6))
#     result.confusion_matrix.plot(normalize=False, save_dir='tmp', names=(), on_plot=None)
#     col[6].write(round(time.time() - start_time,6))
#     col[7].image('tmp/confusion_matrix.png')

if model1!=None:
    print_mod1( model1)
if model2!=None:
    print_mod2( model2)
if model3!=None:
    print_mod3( model3)
# if k:
#     print_detr()