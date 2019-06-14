import numpy as np
import os
import time
import matplotlib.pyplot as plt
# plt.switch_backend('agg')     # This line to work in server with no display
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import  Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def read_dataset(data_location):
    '''
    :param data_location: '/data'
    :return:
    '''
    path = os.getcwd()
    data_path = path+data_location
    data_dir_list = os.listdir(data_path)
    # Create an array to store images from folder
    img_data_list=[]
    for dataset in data_dir_list:
        img_list = os.listdir(data_path+'/'+dataset)
        print('Loading images of dataset -'+'{}\n'.format(dataset))
        for img in img_list:
            img_path = data_path+'/'+dataset+'/'+img
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            # x = x/255
            print('Input image shape: ', x.shape)
            img_data_list.append(x)
    img_data = np.array(img_data_list)
    # img_data = img_data.astype('float32')
    print(img_data.shape)
    img_data = np.rollaxis(img_data,1,0)
    print(img_data.shape)
    img_data=img_data[0]
    print(img_data.shape)
    return img_data


def labelling_outputs(number_of_classes, number_of_samples):
    labels = np.ones((number_of_samples,), dtype='int64')
    labels[0:202] = 0
    labels[202:404] = 1
    labels[404:606] = 2
    labels[606:] = 3
    return labels


def labelling_mammo(number_of_classes, number_of_samples):
    labels = np.ones((number_of_samples,), dtype='int64')
    labels[0:903] = 0
    labels[903:] = 1
    return labels


def main(train_epochs):
    print('Hello Lenin Welcome to Transfer Learning with VGG16')
    # Reading images to form X vector
    img_data = read_dataset('/data_roi_single/train')
    categories_names = ['benign', 'malignant']
    num_classes = 2
    # labels = labelling_outputs(num_classes, img_data.shape[0])
    labels = labelling_mammo(num_classes, img_data.shape[0])
    # converting class labels to one-hot encoding
    y_one_hot =np_utils.to_categorical(labels, num_classes)
    #Shuffle data
    x,y = shuffle(img_data, y_one_hot, random_state=2)
    # Dataset split
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=2)

    #########################################################################################
    # Custom_vgg_model_1
    # Training the classifier alone
    image_input = Input(shape=(224, 224, 3))

    model = VGG16(input_tensor=image_input, include_top=True, weights='imagenet')
    model.summary()
    last_layer = model.get_layer('fc2').output
    out = Dense(num_classes, activation='softmax', name='vgg16TL')(last_layer)
    custom_vgg_model = Model(image_input, out)
    custom_vgg_model.summary()
    # until this point the custom model is retrainable at all layers
    # Now we freeze all the layers up to the last one
    for layer in custom_vgg_model.layers[:-1]:
        layer.trainable = False
    custom_vgg_model.summary()

    custom_vgg_model.layers[3].trainable
    # custom_vgg_model.layers[-1].trainable

    # Model compilation
    custom_vgg_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print('Transfer Learning Training...')
    t = time.time()

    num_of_epochs = train_epochs   # User defines number of epochs

    hist = custom_vgg_model.fit(xtrain, ytrain,
                                batch_size=64,
                                epochs=num_of_epochs,
                                verbose=1,
                                validation_data=(xtest, ytest))
    print('Training time: %s'%(time.time()-t))
    # Model saving parameters
    custom_vgg_model.save('vgg16_tf_bc.h5')


    print('Evaluation...')
    (loss, accuracy)  = custom_vgg_model.evaluate(xtest, ytest, batch_size=10, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy*100))
    print("Finished")

    # Model Training Graphics
    # Visualizing losses and accuracy
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['acc']
    val_acc = hist.history['val_acc']

    xc = range(num_of_epochs)    # Este valor esta anclado al numero de epocas

    plt.figure(1, figsize=(7, 5))
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('num of epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train', 'val'])
    plt.style.use(['classic'])    # revisar que mas hay
    plt.savefig('vgg16_train_val_loss.jpg')

    plt.figure(2, figsize=(7, 5))
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('num of epochs')
    plt.ylabel('accuracy')
    plt.title('train_accuracy vs val_accuracy')
    plt.grid(True)
    plt.legend(['train', 'val'], loc=4)
    plt.style.use(['classic'])  # revisar que mas hay
    plt.savefig('vgg16_train_val_acc.jpg')

    # plt.show()


if __name__ == '__main__':
    num_train_epochs = 1000
    main(train_epochs=num_train_epochs)
