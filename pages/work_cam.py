import streamlit as st
import os

st.set_page_config(layout='wide', page_title='Hack.23 inNINO')
st.header('Обнаружение на изображениях с камеры')
st.write('При нажатии кнопки запуск видеокамеры запускаетсся многопоточный скрипт воспроизведения видео со стандартной'
         ' видеокамеры. Во время работы приложения есть две активные клавиши:')
st.write('Y - включить/выключить обнаружение объектов в кадре. Если надпись горит красным, то обнаружение отключено.')
st.write('Esc - выход из программы.')
if st.button("Запуск видеокамеры"):
    os.system("cam.py")


cl = st.text_input('Введите адрес камеры', "rtsp://admin:123asdASD@192.1.1.163:554/Streaming/channels/103/")
cl2 = st.text_input('Введите адрес обнаружителя', "192.1.1.2")
if st.button("Запуск tmp"):
    os.system("fullscreen.py -cam "+cl+" -det "+cl2)