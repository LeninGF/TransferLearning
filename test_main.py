'''
This script will load the h5 model and predict random image
This script will use a pre-trained model on cats-dogs-horse-human classifier
Author: Lenin G. Falconi
Date: 13 June 2019
'''

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import load_model

new_model = load_model('cat-dog-horse-human.h5')
print(new_model.summary())

img_path = './test_cadohohu/horse.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
x = preprocess_input(x)
print('Input Image Shape: ', x.shape)
#%% md
## Predictions:
# We print top 3 predictions with their % of probability
#%%
preds = new_model.predict(x)
# print('Predicted: ', decode_predictions(preds, top=3)[0])
print(preds)

