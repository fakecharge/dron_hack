import streamlit as st
import numpy as np
import cv2
import os
import shutil
from sahi.predict import AutoDetectionModel, get_sliced_prediction
import str_start

st.set_page_config(layout='wide', page_title='Hack.24 inNINO')
st.header('Работа с изображениями')

img_det = st.sidebar.file_uploader(
    "Загрузить изображение",
    type=['png', 'jpg', 'jpeg', 'bmp'])
if st.sidebar.button('Очистить память'):
    st.cache_data.clear()


def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy

    elif noise_typ == "s&p":
        s_vs_p = 0.5
        amount = 0.04
        out = np.copy(image)

        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy

# model = YOLO('weights/'+model)
@st.cache_data
def pred (_img, _detection_model, _slice_x, _slice_y):
    return get_sliced_prediction(
        _img,
        _detection_model,
        slice_height=_slice_x,
        slice_width=_slice_y,
        overlap_height_ratio=0.1,
        overlap_width_ratio=0.1,
        perform_standard_pred=False,
        postprocess_class_agnostic=True,
        postprocess_type='GREEDYNMM'  # 'NMM', 'LSNMS', 'GREEDYNMM' or 'NMS'
    )

def change_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

if img_det is not None:
    bytes_data = img_det.getvalue()
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    # img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame,  cv2.COLOR_BGR2RGB)

    col1,col2,col3 = st.columns(3)
    col1.write('Изменение изображения')
    br = col1.slider('Изменение яркости', -100, 100, 0)
    angle = col1.slider('Поворот изображения', -180, 180, 0)
    kontr = col2.checkbox('Контрастировать', False, help='Повысить контрастность')

    col2.write('Зашумление изображения')
    gs = col2.checkbox('gauss', False)
    sp = col2.checkbox('s&p', False)
    poi = col2.checkbox('poisson', False)
    speck = col2.checkbox('speckle', False)

    # col3.write('Обнаружение')
    det = col3.checkbox('Обнаружение по модели', False)
    porog = col3.slider('Выберите порог обнаружения', 0, 100, 50)

    model = col3.selectbox(
        'Выберите модель',
        os.listdir(str_start.cl_mod),
        index=1,
        format_func=lambda s: s.upper())
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=str_start.cl_mod + '/' + model,
        confidence_threshold=0.25,
        device="cpu",  # or 'cuda:0'
    )


    # col1.write(br)
    img = change_brightness(frame, br)
    if kontr:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        b, g, r = cv2.split(img)
        b = clahe.apply(b)
        g = clahe.apply(g)
        r = clahe.apply(r)
        img = cv2.merge([b, g, r])
    if gs:
        img = noisy('gauss', img)
    if sp:
        img = noisy('s&p', img)
    if poi:
        img = noisy('poisson', img)
    if speck:
        img = noisy('speckle', img)
    # if kontr:
    #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #     b, g, r = cv2.split(img)
    #     b = clahe.apply(b)
    #     g = clahe.apply(g)
    #     r = clahe.apply(r)
    #     img = cv2.merge([b, g, r])
    img = rotate_image(img, angle)
    st.write(' ')
    col1, col2, col3 = st.columns(3)
    col1.write('Оригинальное')
    col1.image(frame)
    col2.write('Обработанное')
    cv2.imwrite('img_input/tmp2.png', img)
    img = cv2.imread('img_input/tmp2.png')
    col2.image(img)
    col3.write('Обнаружение')
    # col2.image(img)


    if det:
        result = pred(img, detection_model, 640,640)
        # result.export_visuals(export_dir="result/", file_name='res')
        # st.image('result/res.png')

        result_copy = result.object_prediction_list.copy()

        img_draw = img


        count_obn = 0
        for bbox_data in range(0, len(result_copy)):

            tmp = result.object_prediction_list[bbox_data].bbox
            p0, p1, label = (int(tmp.minx), int(tmp.miny)), (int(tmp.maxx), int(tmp.maxy)), \
                int(result.object_prediction_list[bbox_data].category.id)
            if 100*result.object_prediction_list[bbox_data].score.value > porog:
                img_draw = cv2.rectangle(img_draw, p0, p1, (255,0,0), 5)
                count_obn += 1


        col3.image(img_draw)
        col3.success(
            'На пороге '+str(porog)+'% обнаружено ' + str(count_obn) + ' объектов')

else:
    st.error('Загрузите изображение')

st.header('Аугментация датасет')
col1, col2, col3 = st.columns(3)
test_dir = col1.text_input('введите путь к датасету', 'data')
tmp_dir = col2.text_input('введите путь назначения', 'tmp')
col3.write(' ')
col3.write(' ')
if col2.button('Аугментировать датасет'):
        # test_dir = 'datast/train/'
        # tmp_dir = 'datast/tmp/train/'
        try:
            os.mkdir(tmp_dir)
        except:
            print('Folder not create')
        path = os.path.join(tmp_dir, 'images')
        try:
            os.mkdir(path)
        except:
            print('\images\ not create')
            # col3.write('Папка уже существует')
        path = os.path.join(tmp_dir, 'labels')
        try:
            os.mkdir(path)
        except:
            print('\labels\ not create')
            # col3.write('Папка уже существует')
        tmp_img = tmp_dir + '/images/'
        tmp_labels = tmp_dir + '/labels/'
        # print(test_dir[:-3])
        # print(test_dir[3:])
        all_frames = os.listdir(test_dir + '/images')
        all_ = []
        for img_ in all_frames:
            all_.append(img_[:-4])
            img = cv2.imread(test_dir + '/images/' + img_, cv2.IMREAD_UNCHANGED)

            chg_img = img * 1.2
            cv2.imwrite(tmp_img + img_[:-4] + '_br.png', chg_img)

            sourcePath = test_dir + '/labels/' + img_[:-4] + '.txt'
            destinationPath = tmp_labels + img_[:-4] + '_br.txt'
            shutil.copyfile(sourcePath, destinationPath)

            chg_img = img * 0.8
            cv2.imwrite(tmp_img + img_[:-4] + '_b.png', chg_img)

            sourcePath = test_dir + '/labels/' + img_[:-4] + '.txt'
            destinationPath = tmp_labels + img_[:-4] + '_b.txt'
            shutil.copyfile(sourcePath, destinationPath)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            b, g, r = cv2.split(img)
            b = clahe.apply(b)
            g = clahe.apply(g)
            r = clahe.apply(r)
            chg_img = cv2.merge([b, g, r])
            cv2.imwrite(tmp_img + img_[:-4] + '_cl.png', chg_img)

            sourcePath = test_dir + '/labels/' + img_[:-4] + '.txt'
            destinationPath = tmp_labels + img_[:-4] + '_cl.txt'
            shutil.copyfile(sourcePath, destinationPath)

            chg_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(tmp_img + img_[:-4] + '_bgr.png', chg_img)

            sourcePath = test_dir + '/labels/' + img_[:-4] + '.txt'
            destinationPath = tmp_labels + img_[:-4] + '_bgr.txt'
            shutil.copyfile(sourcePath, destinationPath)

            chg_img = noisy("gauss", img)
            cv2.imwrite(tmp_img + img_[:-4] + '_gs.png', chg_img)

            sourcePath = test_dir + '/labels/' + img_[:-4] + '.txt'
            destinationPath = tmp_labels + img_[:-4] + '_gs.txt'
            shutil.copyfile(sourcePath, destinationPath)

            # chg_img = noisy("s&p", img)
            # cv2.imwrite(tmp_img + img_[:-4] + '_sp.png', chg_img)
            #
            # sourcePath = test_dir + '/labels/' + img_[:-4] + '.txt'
            # destinationPath = tmp_labels + img_[:-4] + '_sp.txt'
            # shutil.copyfile(sourcePath, destinationPath)

            chg_img = noisy("poisson", img)
            cv2.imwrite(tmp_img + img_[:-4] + '_ps.png', chg_img)

            sourcePath = test_dir + '/labels/' + img_[:-4] + '.txt'
            destinationPath = tmp_labels + img_[:-4] + '_ps.txt'
            shutil.copyfile(sourcePath, destinationPath)

            chg_img = noisy("speckle", img)
            cv2.imwrite(tmp_img + img_[:-4] + '_sc.png', chg_img)

            sourcePath = test_dir + '/labels/' + img_[:-4] + '.txt'
            destinationPath = tmp_labels + img_[:-4] + '_sc.txt'
            shutil.copyfile(sourcePath, destinationPath)

        col3.success('Датасет из '+test_dir+' аугментирован в '+tmp_dir)

st.header('Фрагментирование датасета по размеру')
col1,col2,col3 = st.columns(3)
cl = col1.text_input('Введите адрес датасета', "data")
cl1 = col1.text_input('Введите целевой адрес', "tmp")

cl2 = col2.text_input('Разширение изображений', ".jpg")
cl3 = col2.text_input('Введите адрес для пустых фрагментов', "tmp/empt")

cl4 = col3.text_input('Введите размер фрагментов', "640")

if col2.button("Фрагментировать датасет"):
    try:
        os.system("tiler.py -source "+cl+" -target "+cl1+" -ext "+cl2+" -falsefolder "+cl3+" -size "+cl4)
        col3.success('Датасет фрагментирован')
    except:
        col3.error('Неправильно введены параметры')

