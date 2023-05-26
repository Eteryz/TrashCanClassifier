#import matplotlib.pyplot as plt
import cv2
#import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.utils import normalize


def predict():
    # load model
    model = load_model('New_model2_50epochs_opt_rmsprop_result_img64_batchsize8_shuffle_True_best.h5')
    # image = Image.open("C:\\Users\\Vladislav\\PycharmProjects\\TrashCan\\Dataset\\full\\115.jpg", 'r')
    # image = Image.open("C:\\Users\\Vladislav\\PycharmProjects\\TrashCan\\Dataset\\empty\\1.jpg", 'r')
    # image = Image.open("Dataset/.jpg", 'r')
    image = cv2.imread("C:\\Users\\Vladislav\\PycharmProjects\\Diplom\\TrashCanClassifier\\Dataset_Old\\T1.png")
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    img = np.array(image)
    normImage = normalize(img, axis=1)
    # normImage = img / 255.0  # Нормализация значений пикселей

    # plt.imshow(normImage)
    # plt.show()
    # Расширение размерности массива для соответствия входному формату модели
    input_img = np.expand_dims(normImage, axis=0)
    res = model.predict(input_img)
    print("The prediction for this image is: ", res)
    # print("Result: ", res > 0.85)
    return res > 0.95


if __name__ == '__main__':
    print(predict())
