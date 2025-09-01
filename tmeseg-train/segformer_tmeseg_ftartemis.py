import numpy as np
import tensorflow as tf
import random as rn
import os
from glob import glob
from tensorflow.keras import backend
from transformers import create_optimizer
from transformers import TFAutoModelForSemanticSegmentation
from transformers import DefaultDataCollator
from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback
from transformers import AutoImageProcessor
from tensorflow.keras.optimizers.legacy import Adam

np.random.seed(2023)
tf.random.set_seed(2023)
rn.seed(2023)



img = glob('/path_to_tme_annotations/image/*.png') #to setup
num_img = len(img)
print(num_img)
img_names = [path.split('/image/')[1][:-4] for path in img]
label = ['/path_to_tme_annotations/maskPng/' + 'mask_' + name + '.png' for name in img_names] #to setup
train_ds = tf.data.Dataset.from_tensor_slices((img, label))

image_size = 512
learning_rate = 0.0001
auto = tf.data.AUTOTUNE
batch_size = 8
num_epochs = 60


def read_image(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [image_size, image_size])
    return img

def read_mask(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, [image_size, image_size])
    return img

#######################
def rand_crop(img, mask):
    concat_img = tf.concat([img, mask], axis=-1)
    concat_img = tf.image.resize(concat_img, [280, 560], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    crop_img = tf.image.random_crop(concat_img, [256, 256, 4])
    return crop_img[:, :, :3], crop_img[:, :, 3:]

def norm(image, mask):
    image = tf.cast(image, tf.float32) / 0.5 - 1
    #mask = tf.cast(mask, tf.int32)
    return image, mask

def normalize(input_image, input_mask):
    input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    #input_mask /= 255. mutli class no need this step
    input_mask = tf.image.convert_image_dtype(input_mask, tf.int32)
    return input_image, input_mask
#######################



def aug_transforms(image):
    image /= 255.0
    image = tf.image.random_brightness(image, 0.25)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.8, 2.0)
    image = tf.image.random_hue(image, 0.1)
    return image

def load_img_train(img, mask):
    image = read_image(img)
    mask = read_mask(mask)
    image = aug_transforms(image)
    #image, mask = norm(image, mask)
    image = tf.transpose(image, (2, 0, 1)) #for segformer
    mask = tf.squeeze(mask, axis=-1)
    print(image.shape)
    print(mask.shape)
    return image, mask

def load_img_val(img, mask):
    image = read_image(img)
    mask = read_mask(mask)
    return normalize(image, mask)

train_ds = train_ds.map(load_img_train, num_parallel_calls=auto)
train_ds = train_ds.repeat()
train_ds = train_ds.batch(batch_size)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

id2label = {
    0:  'background',
    1:  'tumor',
    2:  'stroma',
    3:  'benign',
    4:  'necrosis',
    5:  'adipose'
} 
label2id = {label: id for id, label in id2label.items()}
num_labels = len(id2label)


model_checkpoint = "../tmeseg-infer/model/mit-b3-finetuned-TCGAbcss"  #artemis finetune on BRCA
#image_processor = AutoImageProcessor.from_pretrained(model_checkpoint), from scratch
optimizer = Adam(learning_rate=learning_rate)
model = TFAutoModelForSemanticSegmentation.from_pretrained(model_checkpoint, num_labels=num_labels,
                                                           id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
model.compile(optimizer=optimizer)
model_name = model_checkpoint.split("/")[-1]
model_id = f"../tmeseg-infer/model/{model_name}-finetunedBRCA-Artemis"
model.fit(train_ds,
          steps_per_epoch=num_img // batch_size,
          epochs=num_epochs, verbose=1)
model.save_pretrained(model_id)
