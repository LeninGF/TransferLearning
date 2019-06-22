'''
This script will load the h5 model and predict random image
This script will use a pre-trained model on cats-dogs-horse-human classifier
Author: Lenin G. Falconi
Date: 13 June 2019
'''

import numpy as np
import os
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import load_model


img_path = './test_cadohohu/horse.jpg'
#%% md
## Predictions:
# We print top 3 predictions with their % of probability
#%%

# print('Predicted: ', decode_predictions(preds, top=3)[0])


def process_image(path2img):
    img = image.load_img(path2img, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input Image Shape: ', x.shape)
    return x


def prediction(model, path2_folder, img_name):
    img_path = os.path.join(path2_folder, img_name)
    print(img_path)
    x = process_image(path2img=img_path)
    preds = model.predict(x)
    return preds


def class_prediction(model, path2_folder, img_name):
    img_path = os.path.join(path2_folder, img_name)
    print(img_path)
    x = process_image(path2img=img_path)
    # model.predict_classes no funciona
    class_preds = model.predict_classes(x)
    return class_preds


def main(path_to_h5, path_to_test_folder):
    # Loading model from h5 file
    new_model = load_model(path_to_h5)
    print(new_model.summary())
    # Prediction
    y1 = prediction(new_model, path_to_test_folder, img_name='cat1.jpg')
    # y1c = class_prediction(new_model, path_to_test_folder, img_name='cat1.jpg')
    print('must be a cat', y1)
    print(np.argmax(y1))
    # print('must be a cat', y1c)

    y1 = prediction(new_model, path_to_test_folder, img_name='cat2.jpg')
    print('must be a cat', y1)
    y1 = prediction(new_model, path_to_test_folder, img_name='cat3.jpg')
    print('must be a cat', y1)

    y2 = prediction(new_model, path_to_test_folder, img_name='dog1.jpg')
    print('must be a dog', y2)
    y2 = prediction(new_model, path_to_test_folder, img_name='dog2.jpg')
    print('must be a dog', y2)
    y2 = prediction(new_model, path_to_test_folder, img_name='dog3.jpg')
    print('must be a dog', y2)
    y2 = prediction(new_model, path_to_test_folder, img_name='dog10.jpg')
    print('must be a dog', y2)
    y2 = prediction(new_model, path_to_test_folder, img_name='dog11.jpg')
    print('must be a dog', y2)
    y2 = prediction(new_model, path_to_test_folder, img_name='dog12.jpg')
    print('must be a dog', y2)

    y3 = prediction(new_model, path_to_test_folder, img_name='horse.jpg')
    print('must be a horse', y3)
    y3 = prediction(new_model, path_to_test_folder, img_name='horse-1.jpg')
    print('must be a horse', y3)
    y3 = prediction(new_model, path_to_test_folder, img_name='horse-2.jpg')
    print('must be a horse', y3)
    y3 = prediction(new_model, path_to_test_folder, img_name='horse-3.jpg')
    print('must be a horse', y3)

    y4 = prediction(new_model, path_to_test_folder, img_name='person1.jpg')
    print('must be a human', y4)
    y4 = prediction(new_model, path_to_test_folder, img_name='person2.jpg')
    print('must be a human', y4)
    y4 = prediction(new_model, path_to_test_folder, img_name='person3.png')
    print('must be a human', y4)
    y4 = prediction(new_model, path_to_test_folder, img_name='person-180.jpg')
    print('must be a human', y4)
    y4 = prediction(new_model, path_to_test_folder, img_name='person-181.jpg')
    print('must be a human', y4)
    y4 = prediction(new_model, path_to_test_folder, img_name='person-182.jpg')
    print('must be a human', y4)

    return 0


if __name__ == '__main__':
    main(path_to_h5='cat-dog-horse-human-2.h5', path_to_test_folder='./test_cadohohu')
