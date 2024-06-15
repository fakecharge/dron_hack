import streamlit as st
import pandas as pd

st.set_page_config(layout='wide', page_title='Hack.23 inNINO')
st.title('Система обнаружения объектов inNINO DRON')
global cl_mod

df = pd.read_csv("results.csv")

def write(title, text):
    st.markdown(f'<p style="font-size:20px; border-radius: 10px;"><dev style="background-color:#27ae60;font-size:24px;border-radius: 5px; padding: 2px 5px 2px 5px;">{title}</dev> - {text}</p>',
                unsafe_allow_html=True)
def write_t(text):
    st.markdown(f'<p style="font-size:20px; border: 2px solid #27ae60; padding : 10px; border-radius: 10px;">{text}</p>',
                unsafe_allow_html=True)
def main_ops(text):
    st.markdown(
        f'<p style="font-size:24px; border-radius: 10px;">{text}</p>',
        unsafe_allow_html=True)

st.line_chart(df, x="                  epoch")
cl_mod = st.sidebar.text_input('введите путь к обученным моделям', 'models')
main_ops('Данный веб-сервис позволяет пользователю обрабатывать данные, связанные с обнаружением объектов. '         
         'Доступна обработка кадров, видеофайлов, работа с видеокамерами, датасетом и моделями yolo. ')
st.header('Сервис разделен на 7 частей:')
write('str_start', 'описательная часть сервиса. Здесь можно задать путь к обученным моделям')
write('chg_img', 'приложение для работы с изображениями и набором кадров с просмотром результата.'         
                 'При его выборе в окне браузера будет возможность подгрузить изображение. '
         'При загрузке появляется возможность просмотреть изменения и влияние изменений на обнаружение.'        
                 'В данном разделе можно аугментировать данные, а также фрагментировать изображения для создания датасета')
st.image('gifs/aug_file.gif')
write('work_img', 'приложение для обработки изображений с помощью обученной модели. '
         'Для получения результирующего набора требуется загрузить изображения в формтае jpg или png, '
         'выбрать модель для обработки и в основном поле появится кнопка для скачивания архива '
         'с данными инференса в формате yolo.')
st.image('gifs/image_file.gif')
write('work_vid', 'приложение для обработки видео с помощью обученной модели. На экране можно выбрать модель и '
         'ввести путь до видео. После обработки появится слайдер с кадрами и обработанное видео. Результат сохранен '
         'там же, где и оригинал видео.')
st.image('gifs/video_file.gif')
write('test_models', 'приложение сравнения обученных моделей по тестовым данным. Позволяет выбрать до трех моделей '
         'и сравнить характеристики в блочном видео')
st.image('gifs/test_model.gif')
write('work_cam', 'запускает многопоточное приложение для обнаружения объектов на изображениях с подключенной '                    
                    'видеокамеры в реальном времени, а также подключить IP видеокамеру в систему с обнаружителем')
st.image('gifs/camera_file.gif')