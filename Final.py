# IMPORT LIBRARY YANG DIBUTUHKAN

import matplotlib.pyplot as plt
import os 
import tensorflow as tf 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, Rescaling, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
"""
Sequential = untuk tumpukan polos lapisan dimana masing-masing lapisan memiliki tepat satu tensor input dan satu output tensor.
ImageDataGenerator = untuk memproses data sebelum di load
Dropout = biasanya digunakan untuk mencegah overfitting sehingga mengurangi koneksi dari neuron. biasanya ada nilai thresholdnya sebagai input parameter
Flatten = untuk membuat input yang memiliki banyak dimensi menjadi satu dimensi. biasanya digunakan sebelum ke fully connected
Dense = layer pada model arsitektur yang berisi neuron neuron(1 layer berisi banyak neuron)
"""

#IMPORT DATASET
mypath = 'Toyota_car/'

# MENGHITUNG JUMLAH GAMBAR KESELURUHAN YANG ADA PADA DATASET
number_label = {}
total_files = 0
for i in os.listdir(mypath):
    counting = len(os.listdir(os.path.join(mypath, i)))
    number_label[i] = counting
    total_files += counting
print("Total Files : " + str(total_files))
print(number_label.keys())
print(number_label.values())

# VISUALISASI JUMLAH DATA PER KELAS
import matplotlib.pyplot as plt

plt.bar(number_label.keys(), number_label.values());
plt.title("Jumlah Gambar Tiap Label");
plt.xlabel('Label');
plt.ylabel('Jumlah Gambar');
plt.show();

# MENAMPILKAN SAMPEL GAMBAR TIAP KELAS
import matplotlib.image as mpimg

img_each_class = 1
img_samples = {}
classes = list(number_label.keys())

for c in classes:
    temp = os.listdir(os.path.join(mypath, c))[:img_each_class]
    for item in temp:
        img_path = os.path.join(mypath, c, item)
        img_samples[c] = img_path

for i in img_samples:
    fig = plt.gcf()
    img = mpimg.imread(img_samples[i])
    plt.title(i)
    plt.imshow(img)
    plt.show()

## MEMPERSIAPKAN PARAMETER

IMAGE_SIZE = (300, 140)
BATCH_SIZE = 2
SEED = 999

# MENGGUNAKAN IMAGEDATAGENERATOR UNTUK PREPROCESSING
datagen = ImageDataGenerator(
    validation_split=0.3
)

# MENYIAPKAN DATA TRAIN DAN DATA VALIDATION
train_data = datagen.flow_from_directory(
    mypath,
    class_mode='categorical',
    subset='training',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

valid_data = datagen.flow_from_directory(
    mypath,
    class_mode='categorical',
    subset='validation',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

# IMAGE AUGMENTATION
data_augmentation = Sequential(
    [
        RandomFlip("horizontal", input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3)),
        RandomRotation(0.1),
        RandomZoom(0.1),
        Rescaling(1. / 255)
    ]
)

# MEMUAT MODEL VGG16
# LOAD MODEL VGG16
base_vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
base_vgg_model.trainable = False

# SUMMARY MODEL DASAR
base_vgg_model.summary()

# PREPROCESSING INPUT
vgg_preprocess = tf.keras.applications.vgg16.preprocess_input
train_data.preprocessing_function = vgg_preprocess

# TRANSFER LEARNING VGG16
vgg_model = Sequential([
    data_augmentation,
    base_vgg_model,
    Dropout(0.7),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax')
])

# COMPILE MODEL
vgg_model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

# SUMMARY MODEL VGG16 TRANSFER LEARNING
vgg_model.summary()

# MELATIH MODEL VGG16 TRANSFER LEARNING
vgg_hist = vgg_model.fit(
    train_data,
    epochs=10,
    validation_data=valid_data
)

# #LAPORAN/RANGKUMAN
# loss, acc = vgg_model.evaluate(train_data, steps=len(train_data), verbose=0)
# print('Accuracy on training data: {:.4f} \nLoss on training data: {:.4f}'.format(acc,loss),'\n')
 
# loss, acc = vgg_model.evaluate(valid_data, steps=len(valid_data), verbose=0)
# print('Accuracy on Validation data: {:.4f} \nLoss on Validation data: {:.4f}'.format(acc,loss),'\n')    

# MEMBUAT DIAGRAM AKURASI VGG16 
plt.figure(figsize=(10, 4))
plt.plot(vgg_hist.history['accuracy'])
plt.plot(vgg_hist.history['val_accuracy'])
plt.title('VGG16 model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(True)
plt.show()

# MEMBUAT DIAGRAM LOSS VGG16
plt.figure(figsize=(10, 4))
plt.plot(vgg_hist.history['loss'])
plt.plot(vgg_hist.history['val_loss'])
plt.title('VGG16 model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid(True)
plt.show()

# MODEL_BASE_PATH = "CODE"
# PROJECT_NAME = "project"
# SAVE_MODEL_NAME = "modelvgg.h5"
# save_model_path = os.path.join(MODEL_BASE_PATH, PROJECT_NAME, SAVE_MODEL_NAME)

# if os.path.exists(os.path.join(MODEL_BASE_PATH, PROJECT_NAME)) == False:
#     os.makedirs(os.path.join(MODEL_BASE_PATH, PROJECT_NAME))
    
# print('Saving Model At {}...'.format(save_model_path))
# vgg_model.save(save_model_path,include_optimizer=False)