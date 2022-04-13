import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout,  GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator



class Preprocessor:
    def __init__(self, data=None):
        self.data = data

    def delete_duplicates(self):
        data = self.data.copy()
        data = data.drop_duplicates()
        return data

    def generate_image_list(self):
        train_bag = [
            2015, 2035, 2223, 2306, 2323, 2331, 2437, 2498, 2676, 2692,
            2941, 2973, 3202, 3198, 3237, 3247, 3248, 3253, 3421, 3546,
            3608, 3619, 3622, 3710, 3827, 3872, 3884, 3980, 4070, 24434,
            4372, 4385, 4409, 4432, 4440, 4625, 4929, 5447, 5532, 5560,
            5591, 5597, 5675, 5881, 6107, 6237, 6387, 6514, 6629, 6727,
            6767, 6914, 6947, 6995, 7269, 7340, 7373, 7429, 7478, 7547,
            7739, 8033, 8075, 8111, 8123, 8599, 8727, 8974, 8989, 8996,
            9003, 9058, 9352, 9372, 9548, 9634, 9810, 9973, 9980, 10024
        ]
        test_bag = [
            3, 12, 44, 88, 309, 541, 543, 556, 649, 734,
            773, 803, 1005, 1024, 1146, 1214, 1410, 1473, 1578, 1848
        ]
        train_shoes = [
            1042, 24840, 1271, 1276, 24596, 24417, 1453, 1533, 1550, 24205,
            23863, 1630, 1798, 1803, 1836, 1856, 2403, 2436, 2580, 2701,
            2795, 2989, 23629, 3073, 3077, 3140, 23494, 3322, 3328, 3349,
            23487, 3702, 3786, 3893, 3906, 3909, 23300, 4000, 4010, 4085,
            4143, 23004, 4266, 4267, 4319, 4321, 4375, 4607, 4631, 4695,
            4726, 5362, 5392, 5429, 5741, 5925, 6046, 6151, 6214, 6822,
            6964, 7162, 7247, 7384, 7428, 7647, 7648, 7867, 8016, 8073,
            8113, 8211, 8361, 8460, 8475, 8597, 8609, 8783, 8859, 8880
        ]
        test_shoes = [
            13, 21, 125, 133, 152, 318, 408, 592, 604,
            760, 769, 833, 850, 869, 883, 887, 901, 9162, 24915
        ]
        train_hood = [
            2764, 3292, 3561, 3738, 3790, 4003, 4068, 4485, 4992, 5071,
            5440, 5470, 5603, 5647, 8045, 8046, 8433, 8869, 10118, 10529,
            11303, 11979, 12047, 13290, 13950, 15425, 15732, 16212, 18246, 18698,
            18941, 19023, 19056, 19123, 19563, 20436, 21881, 22044, 22473, 22499,
            22684, 23088, 23645, 24685, 24964, 20739
        ]
        test_hood = [
            597, 438, 930, 945, 1269, 1455, 1549, 2003, 2174, 2344
        ]
        train_pants = [
            3963, 4191, 4327, 4600, 5012, 5286, 5341, 5439, 5662, 5959,
            6101, 6110, 6406, 6472, 7374, 7422, 7913, 7938, 8048, 8308,
            8368, 9044, 9147, 9298, 9380, 9865, 10186, 10469, 10625, 23949,
            10727, 10808, 11191, 11459, 11518, 11624, 11914, 12084, 11271,
            13369, 13444, 13732, 13874, 14156, 14162, 14271, 14464, 14507, 14819,
            15287, 15691, 16079, 16162, 16178, 16179, 16180, 16182, 16220, 16306,
            16445, 16627, 16675, 16695, 17159, 17204, 17535, 18064, 19159, 19601,
            20221, 20867, 20998, 21709, 22181, 22318, 22330, 22428, 22616, 23915
        ]
        test_pants = [
            168, 172, 183, 225, 823, 880, 1141, 1396, 1697, 1742,
            1763, 1817, 1929, 2021, 2055, 3201, 3330, 3487, 3515, 3604
        ]
        train_t_shirt = [
            7011, 7327, 7694, 8103, 8454, 8463, 8755, 8793, 8804, 8805,
            10224, 10480, 10546, 10559, 11051, 11053, 11245, 11489, 12181, 10121,
            12298, 12627, 12929, 13074, 13272, 13414, 13498, 14368, 14506, 24240,
            14762, 14806, 14964, 15092, 15136, 15160, 15168, 15285, 15313, 15452,
            15539, 15554, 15802, 15893, 15922, 15994, 16064, 16198, 16204, 16208,
            16209, 16263, 16845, 16941, 17055, 17156, 17339, 17545, 17955, 18399,
            18968, 19273, 20044, 20146, 20415, 20715, 21688, 22084, 22138, 22144,
            22360, 22381, 22456, 22703, 23011, 23012, 23438, 23497, 23809, 23812
        ]
        test_t_shirt = [
            134, 241, 577, 716, 902, 1461, 1730, 1784, 1868, 2574,
            4249, 4302, 4377, 4579, 4703, 5442, 5577, 5598, 5653, 6055
        ]

        return [train_bag, train_shoes, train_hood, train_pants, train_t_shirt], [test_bag, test_shoes, test_hood,
                                                                                  test_pants, test_t_shirt]

    def generate_dataframe(self, category_image):
        new_data = {'category': [], 'image': []}
        for i in range(len(category_image)):
            for j in category_image[i]:
                new_data['category'].append(i)
                new_data['image'].append(self.data['image'][j])
        df_data = pd.DataFrame(new_data)
        return df_data

    def iterate_requests(self, data):
        """
        data = {
            "category": ["shoes", "bag", "hood"],
            "image": ["http://danawa.com", "http://naver.com", "http://google.com"] 50000만개
        }
        """
        raw_image_list = []
        for index in tqdm(range(len(data['image']))):
            url = data['image'][index]
            raw_image = self.request_image_file(url)
            raw_image_list.append(raw_image)

        return raw_image_list

    def request_image_file(self, url):
        res = requests.get(url)
        raw_image = res.content
        return raw_image

    def iterate_save_image(self, data_type, data, raw_image_list):
        category_dict = {0: 'bag', 1: 'shoes', 2: 'hood', 3: 'pants', 4: 't-shirt'}
        for index in tqdm(range(len(raw_image_list))):
            raw_image = raw_image_list[index]
            category = category_dict[data['category'][index]]
            filename = "sample"
            filepath = self.generate_filepath(data_type, category, filename)
            self.save_image(raw_image, filepath)

    def generate_filepath(self, data_type, category, filename):
        return f'./flask/{data_type}_image_data/{category}/{filename}.png'

    def save_image(self, raw_image, filepath):
        img = Image.open(BytesIO(raw_image))
        img = img.convert("RGB")
        resize = img.resize((299, 299))
        resize.save(filepath)

    def do_save_images(self, data, data_type):
        raw_image_list = self.iterate_requests(data)
        self.iterate_save_image(data_type, data, raw_image_list)

    def generate_image(self, type):
        IMAGE_DIR = f'./flask/static/{type}_image_data/'
        data_generator = None
        if type == 'train':
            data_generator = self.make_train_image_data_generator(IMAGE_DIR)
        elif type == "test":
            data_generator = self.make_test_image_data_generator(IMAGE_DIR)
        return data_generator

    def make_train_image_data_generator(self, directory):
        datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=40,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        data_generator = datagen.flow_from_directory(
            directory,
            batch_size=32,
            target_size=(299, 299),
            class_mode='categorical',
        )
        return data_generator

    def make_test_image_data_generator(self, directory):
        datagen = ImageDataGenerator(rescale=1.0 / 255.0)
        data_generator = datagen.flow_from_directory(
            directory,
            batch_size=32,
            target_size=(299, 299),
            class_mode='categorical',
        )
        return data_generator


def inception_model(train, test, numclass):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(numclass, activation='softmax'))
    model.summary()
    early_stop = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint('flask/model/inception_best_model_5.h5', monitor='val_accuracy', mode='max', verbose=1,
                                       save_best_only=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit_generator(train, epochs=30, validation_data=test, callbacks=[model_checkpoint, early_stop])

    with open('train_history_5', 'wb') as f:
        pickle.dump(history.history, f)
    return model



def predict_image(data, category_dict, model):
    predicted = model.predict(data)
    idx = np.argmax(predicted[0])
    category = category_dict[idx]
    percentage = round(np.max(predicted[0]) * 100, 1)
    return idx, category, percentage


def show_loss(history):
    epochs = range(1, len(history['accuracy']) + 1)
    plt.plot(epochs, history['loss'])
    plt.plot(epochs, history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def show_accuracy(history):
    epochs = range(1, len(history['accuracy']) + 1)
    plt.plot(epochs, history['accuracy'])
    plt.plot(epochs, history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



if __name__ == '__main__':
    # data = pd.read_csv('data.csv')
    #
    # # 1. 데이터 전처리(1차) - 기본 전처리
    # pre = Preprocessor(data)
    # # 1-1) 중복값 제거
    # data = pre.delete_duplicates()
    # pre.data = data
    #
    # # 1-2) 빈값 제거
    # data = pre.delete_missing_value()
    # pre.data = data
    #
    # # 1-3) 데이터 저장
    # data.to_csv('data1.csv', encoding='utf-8-sig', index=False)

    # --------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------

    # 2. 데이터전처리(2차) - 이미지 및 학습 데이터 저장
    data = pd.read_csv('data1.csv')
    pre = Preprocessor(data)
    category_dict = {0: 'bag', 1: 'shoes', 2: 'hood', 3: 'pants', 4: 't-shirt'}

    # 2-1) 카테고리 별 선정한 이미지 인덱스 리스트 생성
    train_list, test_list = pre.generate_image_list()

    # 2-2) 선정한 이미지로 데이터 프레임 생성
    train_data = pre.generate_dataframe(train_list)
    test_data = pre.generate_dataframe(test_list)

    # 2-3) 이미지 저장
    pre.save_images(train_data, category_dict, 'train')
    pre.save_images(test_data, category_dict, 'test')

    # --------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------

    # 2-4) 이미지 증강(ImageGenerator)
    pre = Preprocessor()
    train_generator = pre.generate_image('train')
    test_generator = pre.generate_image('test')

    # 3 모델
    # 3-1) 모델 학습
    # model = inception_model(train_generator, test_generator, 5)
    model = load_model('flask/model/inception_best_model_2.h5')

    # # 3-2) history 그래프 확인
    with open('train_history_6', 'rb') as f:
        history = pickle.load(f)
    show_loss(history)
    show_accuracy(history)








