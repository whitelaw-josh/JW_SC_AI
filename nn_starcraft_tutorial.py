import keras
import numpy as np
import os
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(176, 200, 3), activation='relu'))
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
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

learning_rate = 0.0001
opt = keras.optimizers.adam(lr=learning_rate, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/STAGE1")

#TODO: Might want to pass the lists rather than referencing them
def check_data():
    choices = {
        "no_attacks": no_attacks,
        "attack_closest_to_nexus": attack_closest_to_nexus,
        "attack_enemy_structures": attack_enemy_structures,
        "attack_enemy_start": attack_enemy_start
    }

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
hm_epochs = 10

for i in range(hm_epochs):
    current = 0
    not_maximum = True
    all_files = os.listdir("C:/Program Files (x86)/StarCraft II/train_data")
    maximum = len(all_files)
    random.shuffle(all_files)

    while not_maximum:
        print("WORKING ON {}:{}".format(current, current + 200))        
        no_attacks = []
        attack_closest_to_nexus = []
        attack_enemy_structures = []
        attack_enemy_start = []

        for file in all_files[current:current + 200]:
            data = list(np.load(os.path.join("C:/Program Files (x86)/StarCraft II/train_data", file), allow_pickle=True))

            for d in data:
                choice = np.argmax(d[0])
                
                if choice == 0: #Didn't attack
                    no_attacks.append([d[0], d[1]])
                elif choice == 1: #Attack units closest to our nexus (random)
                    attack_closest_to_nexus.append([d[0], d[1]])
                elif choice == 2: #Attack enemy structures if known
                    attack_enemy_structures.append([d[0], d[1]])
                elif choice == 3: #Attack enemy start location
                    attack_enemy_start.append([d[0], d[1]])

        #lengths = check_data()
        lowest_data = min(check_data()) #Prevent bias on attack decision choices and grabbing the lowest length

        random.shuffle(no_attacks)
        random.shuffle(attack_closest_to_nexus)
        random.shuffle(attack_enemy_structures)
        random.shuffle(attack_enemy_start)

        no_attacks = no_attacks[:lowest_data]
        attack_closest_to_nexus = attack_closest_to_nexus[:lowest_data]
        attack_enemy_structures = attack_enemy_structures[:lowest_data]
        attack_enemy_start = attack_enemy_start[:lowest_data]

        check_data()

        train_data = no_attacks + attack_closest_to_nexus + attack_enemy_structures + attack_enemy_start
        random.shuffle(train_data)

        #Feeding in data
        test_size = 100
        batch_size = 128
        x_train = np.array([i[1] for i in train_data[:-test_size]]).reshape(-1, 176, 200, 3) 
        y_train = np.array([i[0] for i in train_data[:-test_size]])
        
        x_test = np.array([i[1] for i in train_data[-test_size:]]).reshape(-1, 176, 200, 3) 
        y_test = np.array([i[0] for i in train_data[-test_size:]])

        model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_test, y_test), shuffle=True, verbose=1, callbacks=[tensorboard])
        model.save("BasicCNN-{}-epochs-{}-LR-STAGE1".format(hm_epochs, learning_rate))

        current += 200
        if current > maximum:
            not_maximum = False