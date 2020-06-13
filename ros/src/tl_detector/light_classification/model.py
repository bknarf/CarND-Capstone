



if __name__ == "__main__":
    import glob
    import sys
    import math
    import random

    import keras
    from keras.models import Sequential, Model
    from keras.layers import Cropping2D, AveragePooling2D, MaxPooling2D, Lambda, Conv2D, Dropout, Activation, Flatten, Dense
    import tensorflow as tf
    import math

    def prepare_images(imgps):
        print("preparing images")
        res = []
        for imgp in imgps:
            label = int(imgp.split("#")[1])
            one_hot = [0.0 , 0.0 , 0.0]
            one_hot[label] = 1.0
            res.append({"path" : imgp,"label" : one_hot,"mirror" : True})
            res.append({"path" : imgp, "label" : one_hot, "mirror" : False})
        return res


    image_paths = glob.glob("/home/workspace/CarND-Capstone/camera_imgs/*.jpg")

    data = prepare_images(image_paths)
    random.shuffle(data)
    start_validation = int(len(data)*0.7)
    start_test = int(len(data)*0.9)
    train_data = data[0:start_validation]
    validation_data = data[start_validation:start_test]
    test_data = data[start_test:]
    print("Samples: {}".format(len(data)))
    print("Training samples: {}".format(len(train_data)))
    print("Validation samples: {}".format(len(validation_data)))
    print("Testing samples: {}".format(len(test_data)))

    import cv2
    #import sklearn as skl
    import numpy as np

    def batch_data(data, batchsize=32):
        random.shuffle(data)
        num_samples = len(data)
        while 1:  # Loop forever so the generator never terminates
            for offset in range(0, num_samples, batchsize):
                #features/X
                images = []
                #labels/y
                labels = []
                for idx in range(offset, min(offset + batchsize, num_samples)):
                    sample = data[idx]
                    image=cv2.cvtColor(cv2.imread(sample["path"]), cv2.COLOR_BGR2RGB)
                    if sample["mirror"] :
                        #if mirroring is True, flip image, angle already flipped
                        image = cv2.flip(image, 1)
                    images.append(image)
                    labels.append(sample["label"])
                X_train = np.array(images)
                y_train = np.array(labels)
                #yield skl.utils.shuffle(X_train, y_train)
                yield X_train, y_train

    # Set the batchsize
    batchsize = 64

    #creating the generators later needed
    train_generator = batch_data(train_data, batchsize=batchsize)
    validation_generator = batch_data(validation_data, batchsize=batchsize)
    evaluate_generator = batch_data(test_data, batchsize=batchsize)






    #Start creating the model
    model = Sequential()

    #crop to interesting area of image
    #scale image down to half size
    model.add(AveragePooling2D(pool_size=2,input_shape=(600, 800, 3)))
    #normalize the data
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))

    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())

    model.add(Flatten())

    model.add(Dense(units=120, activation='relu'))

    model.add(Dense(units=84, activation='relu'))

    model.add(Dense(units=3, activation='softmax'))



    #mean absolute error to better handle 'outliers' like curves
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    epochs = 12
    #start the training
    history_fit = model.fit_generator(generator=train_generator, steps_per_epoch=math.ceil(len(train_data) / batchsize),
                                  validation_data=validation_generator,
                                  validation_steps=math.ceil(len(validation_data) / batchsize),
                                  epochs=epochs, verbose=1)

    #print the history
    for k,v in history_fit.history.items():
        print("History[{}]={}".format(k,v))

    #do the testing
    test_loss = model.evaluate_generator(evaluate_generator, steps=math.ceil(len(test_data) / batchsize))
    print("Test loss: {}".format(test_loss))
    print("Test samples: {}".format(len(test_data)))



    out = "/opt/carnd_p3/model.h5"
    if len(sys.argv) > 1 :
        #if provided, take path from commandline
        out = sys.argv[1]

    print("saving model to: {}".format(out))
    #save the model
    model.save(out)
    print("saved model to: {}".format(out))
