import streamlit as st
import os

st.set_page_config(layout='wide', page_title='Hack.24 inNINO')
st.header('Обнаружение на изображениях с камеры')
st.write('При нажатии кнопки запуск видеокамеры запускаетсся многопоточный скрипт воспроизведения видео со стандартной'
         ' видеокамеры. Во время работы приложения есть две активные клавиши:')
st.write('Y - включить/выключить обнаружение объектов в кадре. Если надпись горит красным, то обнаружение отключено.')
st.write('Esc - выход из программы.')
if st.button("Запуск видеокамеры"):
    os.system("cam.py")

st.header('Работа с управляемой видеокамерой')
st.write('Данный блок предназначен для работы с удаленной видеокамерой по протоколу ONVIF. В программе реализовано'
         'ручное управление (с помощью клавиатуры, или мышки), получение целеуказаний с обнаружителя (совмещение'
         ' тестировалось с РЛС и стационарной NIR камерой), запись видео, логирование и обнаружение '
         'с помощью Yolo8s c сопровождением.')
st.write('Нажатие правой кнопки мыши перемещает поле зрения, левая позволяет запустить слежение за объектом')
col1,col2 = st.columns(2)
cl = col1.text_input('Введите адрес камеры', "rtsp://user:user@192.1.1.163:554/Streaming/channels/103/")
cl2 = col2.text_input('Введите адрес обнаружителя', "192.1.1.2")
if st.button("Запуск управляемой видеокамеры"):
    os.system("fullscreen.py -cam "+cl+" -det "+cl2)

st.image('gifs/work_cam.gif')