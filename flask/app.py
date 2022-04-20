from flask import Flask, request, render_template
import logging
import numpy as np
from werkzeug.utils import secure_filename
import os


from tensorflow.keras.models import load_model
from PIL import Image
import faiss

app = Flask(__name__)

model = load_model('./model/inception_best_model_2.h5')
category_dict = {0:'bag', 1: 'shoes', 2: 'hood', 3: 'pants', 4: 't-shirt'}

def parse_imgae_request(request):
    image = request.files['file']
    return image

def preprocess_image(target_image):
    img = Image.open(target_image)
    img = img.convert("RGB")
    resize_image = img.resize((299, 299))
    vector_image = np.asarray(resize_image)
    vector_image = np.expand_dims(vector_image, axis=0)
    vector_image = vector_image / 255.0
    vector_image = np.vstack([vector_image])
    return resize_image, vector_image

def predict_image(image):
    predicted = model.predict(image)
    category = np.argmax(predicted[0])
    return category_dict[category]

def save_image(image, category, file_name):
    path_dir = f'./static/train_image_data/{category}/'
    image.save(path_dir + secure_filename(file_name))
    image_path = 'train_image_data/' + category + '/' + file_name
    return image_path

def find_similar_image(category, image):
    group = []  # 검색한 이미지가 속한 카테고리 그룹
    path_dir = f'./static/train_image_data/{category}'
    img_name_list = os.listdir(path_dir)
    for i in range(len(img_name_list)):
        resize, vector_image = preprocess_image(path_dir + '/' +  img_name_list[i])
        group.append(vector_image[0].reshape(-1))
    group = np.array(group).astype('float32')
    index = faiss.IndexFlatL2(group.shape[1])
    index.add(group)
    distances, indices = index.search(image.reshape(1, -1).astype('float32'), 6)
    return indices

def make_img_path_list(indices, category):
    path_dir = f'./static/train_image_data/{category}'
    img_name_list = os.listdir(path_dir)
    similar_img_path_list = []
    for i in range(1, 6):
        if indices[0][i] != -1:
            file_name = img_name_list[indices[0][i]]
            image_path = 'train_image_data/' + category + '/' + file_name
            similar_img_path_list.append(image_path)
    return similar_img_path_list



@app.route('/')
def main():
    return render_template('main.html')

@app.route(rule='/getimage/', methods=['GET'])
def get_image():
    return render_template('get_image.html')

@app.route(rule='/postimage/', methods=['POST'])
def post_image():
    # 1. 데이터 파싱
    image = parse_imgae_request(request)

    # 2. 이미지 전처리
    resize_image, vector_image = preprocess_image(image)

    # 3. 이미지 분류 예측
    category = predict_image(vector_image)

    # 4. 이미지 저장 및 저장 위치 반환
    image_path = save_image(resize_image, category, image.filename)
    
    # 5. 유사 이미지 찾기
    indices = find_similar_image(category, vector_image)

    # 6. 유사 이미지 경로 반환
    similar_img_path_list = make_img_path_list(indices, category)

    return render_template(
        'post_image.html',
        image_path = image_path,
        similar_img_path_list = similar_img_path_list,
        )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=15000, debug=True)