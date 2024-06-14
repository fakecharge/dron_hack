import streamlit as st

st.set_page_config(layout='wide', page_title='Hack.23 inNINO')
st.header('VideoFrameProcessor')
global cl_mod
cl_mod = st.sidebar.text_input('введите путь к обученным моделям', 'models')
st.write('Данный веб-сервис позволяет пользователю загружать видео, '
         'разбивать его на кадры и обрабатывать каждый кадр. Сервис также предоставлять возможность '
         'объединения обработанных кадров обратно в видеофайл и сохранения его в указанную директорию, '
         'а также поддерживать режим реального времени для обработки видео с камеры.')
st.write('Сервис разделен на 7 частей:')
st.write('**str_task4** - описательная часть сервиса. Здесь можно задать путь к обученным моделям')
st.write('**chg_img** - приложение для работы с изображениями и набором кадров с просмотром результата.'
         'При его выборе в окне браузера будет возможность подгрузить изображение. '
         'При загрузке появляется возможность просмотреть изменения и влияние изменений на обнаружение')
# st.image('vid_input/gif/Video1.gif')
st.write('**detect_frames** - приложение для обработки с помощью обученной модели набора кадров. '
         'Для получения результирующего набора требуется выбрать путь до набора кадров, '
         'выбрать модель для обработки и в основном поле появится слайдер с выбором номера кадра и под ним '
         'его обработанное представление. Видео с обработанными кадрами сохранено в каталоге выше.'
         'Также возможно разметить изображения с помощью выбранной модели.')
# st.image('vid_input/gif/Video2.gif')
st.write('**detect_video** - приложение для обработки видео с помощью обученной модели. На экране можно выбрать модель и '
         'ввести путь до видео. После обработки появится слайдер с кадрами и обработанное видео. Результат сохранен '
         'там же, где и оригинал видео.')
# st.image('vid_input/gif/Video3.gif')
st.write('**get_frames** - приложение для извлечения кадров из набора видео. При выборе папки с видеоданными '
         'для каждого файла доступен выбор начального и конечного кадра для извлечения. '
         'По нажатию на кнопку "запись видео" формируется видеофайл с выбранными фрагментами. '
         'По нажатию на кнопку "запись кадров" создается подкаталог и записывается набор выбранных кадров в формате .png'
         'По нажатию на кнопку "запись видео из кадров" формируется видеофайл из изображений из каталога')
# st.image('vid_input/gif/Video4.gif')
st.write('**test_models** - приложение сравнения обученных моделей по тестовым данным. Позволяет выбрать до трех моделей '
         'и сравнить характеристики в блочном виде')
st.write('**detect_cam** - запускает многопоточное приложение для обнаружения с подключенной видеокамеры в реальном времени')
# st.image('vid_input/gif/Video5.gif')