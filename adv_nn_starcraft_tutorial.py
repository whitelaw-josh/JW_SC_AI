import cv2
import keras
import keras.backend.tensorflow_backend as backend
import numpy as np
import os
import random
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard

def get_session(gpu_fraction=0.85):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

backend.set_session(get_session)

model = Sequential()

model.add(Conv2D(32, (7, 7), padding='same', input_shape=(176, 200, 1), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(14, activation='softmax'))

learning_rate = 0.001
opt = keras.optimizers.adam(lr=learning_rate, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="C:/Program Files (x86)/StarCraft II/logs/STAGE2-{}-{}".format(int(time.time()), learning_rate))

#TODO: Might want to pass the lists rather than referencing them
def check_data(choices):
    total_data = 0
    lengths = []

    for choice in choices:
        print("Length of {} is: {}".format(choice, len(choices[choice])))
        total_data += len(choices[choice])
        lengths.append(len(choices[choice]))

    print("Total data length now is: ", total_data)
    return lengths

#Looping through training data sets for up to 10 epochs of the NN by going
#200 training files a time
#TODO: Might be better way to code this portion of code
hm_epochs = 5000

for i in range(hm_epochs):
    current = 0
    not_maximum = True
    all_files = os.listdir("C:/Program Files (x86)/StarCraft II/train_data")
    maximum = len(all_files)
    random.shuffle(all_files)

    while not_maximum:
        try:
            print("WORKING ON {}:{}, EPOCH".format(current, current + 50))        
            choices = {
                0: [],
                1: [],
                2: [],
                3: [],
                4: [],
                5: [],
                6: [],
                7: [],
                8: [],
                9: [],
                10: [],
                11: [],
                12: [],
                13: [],
            }

            for file in all_files[current:current + 50]:
                try:
                    data = list(np.load(os.path.join("C:/Program Files (x86)/StarCraft II/train_data", file), allow_pickle=True))

                    for d in data:
                        choice = np.argmax(d[0])
                        choices[choice].append([d[0], d[1]])
                except Exception as e:
                    print(str(e))

            lowest_data = min(check_data(choices)) #Prevent bias on decision choices and grabbing the lowest length

            for choice in choices:
                random.shuffle(choices[choice])
                choices[choice] = choices[choice][:lowest_data]

            for choice in choices:
                for d in choices[choice]:
                    train_data.append(d)

            random.shuffle(train_data)
            #print(len(train_data))

            train_data = no_attacks + attack_closest_to_nexus + attack_enemy_structures + attack_enemy_start
            random.shuffle(train_data)

            #Feeding in data
            test_size = 100
            batch_size = 128
            x_train = np.array([i[1] for i in train_data[:-test_size]]).reshape(-1, 176, 200, 1) 
            y_train = np.array([i[0] for i in train_data[:-test_size]])
            
            x_test = np.array([i[1] for i in train_data[-test_size:]]).reshape(-1, 176, 200, 1) 
            y_test = np.array([i[0] for i in train_data[-test_size:]])

            model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_test, y_test), shuffle=True, verbose=1, callbacks=[tensorboard])
            model.save("BasicCNN-{}-epochs-{}-0.001-LR-STAGE2".format(hm_epochs, learning_rate))

        except Exception as e:
            print(str(e))

        current += 50
        if current > maximum:
            not_maximum = False