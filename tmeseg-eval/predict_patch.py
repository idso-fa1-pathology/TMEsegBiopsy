
import os
import numpy as np
import cv2
from PIL import Image
import platform
import math
from glob import glob
from transformers import TFAutoModelForSemanticSegmentation
import tensorflow as tf
import pandas as pd

##### this script is to generate patch-level masks, subject to ARTEMEIS TME annotations

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set GPU:1 as the visible device. 
        # This assumes GPU:1 is the second GPU (index 1) in your system.
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("GPU:1 is now the visible device for TensorFlow.")
    except RuntimeError as e:
        # Visible devices must be set at program startup
        print(e)


def to_categorical_mask(multi_label, nClasses):
    categorical_mask = np.zeros((multi_label.shape[0], multi_label.shape[1], nClasses))
    for c in range(nClasses):
        categorical_mask[:, :, c] = (multi_label == c).astype('float')
    return categorical_mask




#openCV: BGR
class_colors = [(0, 0, 0), (0, 0, 128), (0, 204, 255), (255, 255, 0), (255, 0, 255), (0, 128, 128)]

def get_colored_segmentation_image(seg_arr, n_classes, colors=class_colors):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')

    return seg_img


target_image = np.float32(cv2.cvtColor(cv2.imread('target.jpg'), cv2.COLOR_BGR2RGB)) / 255.0
target_lab = cv2.cvtColor(target_image, cv2.COLOR_RGB2Lab)
mt = np.mean(target_lab, axis=(0, 1))
stdt = np.std(target_lab, axis=(0, 1))
def pre_process_images(image, mt=mt, stdt=stdt):
    image = np.float32(image) / 255.0
    if np.any(image):
        image = norm_reinhard(image, mt, stdt)
    feat = 255.0 * image
    feat[feat < 0.0] = 0.0
    feat[feat > 255.0] = 255.0
    feat = np.round(feat)
    return feat

def norm_reinhard(source_image, mt, stdt):
    source_lab = cv2.cvtColor(source_image, cv2.COLOR_RGB2Lab)
    ms = np.mean(source_lab, axis=(0, 1))
    stds = np.std(source_lab, axis=(0, 1))
    if np.sum(stds)<=5:
        norm_image = source_image
    else:
        norm_lab = np.copy(source_lab)
        norm_lab[:, :, 0] = ((norm_lab[:, :, 0] - ms[0]) * (stdt[0] / stds[0])) + mt[0]
        norm_lab[:, :, 1] = ((norm_lab[:, :, 1] - ms[1]) * (stdt[1] / stds[1])) + mt[1]
        norm_lab[:, :, 2] = ((norm_lab[:, :, 2] - ms[2]) * (stdt[2] / stds[2])) + mt[2]
        norm_image = cv2.cvtColor(norm_lab, cv2.COLOR_Lab2RGB)
    return norm_image

class Patches:
    def __init__(self, img_patch_h, img_patch_w, stride_h=384, stride_w=384, label_patch_h=None, label_patch_w=None):
        assert img_patch_h > 0, 'Height of Image Patch should be greater than 0'
        assert img_patch_w > 0, 'Width of Image Patch should be greater than 0'
        assert label_patch_h > 0, 'Height of Label Patch should be greater than 0'
        assert label_patch_w > 0, 'Width of Label Patch should be greater than 0'
        assert img_patch_h >= label_patch_h, 'Height of Image Patch should be greater or equal to Label Patch'
        assert img_patch_w >= label_patch_w, 'Width of Image Patch should be greater or equal to Label Patch'
        assert stride_h > 0, 'Stride should be greater than 0'
        assert stride_w > 0, 'Stride should be greater than 0'
        assert stride_h <= label_patch_h, 'Row Stride should be less than or equal to Label Patch Height'
        assert stride_w <= label_patch_w, 'Column Stride should be less than or equal to Label Patch Width'
        self.img_patch_h = img_patch_h
        self.img_patch_w = img_patch_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.label_patch_h = label_patch_h
        self.label_patch_w = label_patch_w
        self.img_h = None
        self.img_w = None
        self.img_d = None
        self.num_patches_img = None
        self.num_patches_img_h = None
        self.num_patches_img_w = None
        self.label_diff_pad_h = 0
        self.label_diff_pad_w = 0
        self.pad_h = 0
        self.pad_w = 0

    @staticmethod
    def read_image(input_str):
        image = np.array(Image.open(input_str))
        return image

    def update_variables(self, image):
        self.img_h = np.size(image, 0)
        self.img_w = np.size(image, 1)
        self.img_d = np.size(image, 2)

    def extract_patches_img_label(self, input_img_value):
        if type(input_img_value) == str:
            image = self.read_image(input_img_value)
        elif type(input_img_value) == np.ndarray:
            image = input_img_value
        else:
            raise Exception('Please input correct image path or numpy array')
        self.update_variables(image)


        img_patch_h = self.img_patch_h
        img_patch_w = self.img_patch_w

        stride_h = self.stride_h
        stride_w = self.stride_w

        if image.shape[0] < img_patch_h:
            self.pad_h = img_patch_h - image.shape[0]
        else:
            self.pad_h = 0

        if image.shape[1] < img_patch_w:
            self.pad_w = img_patch_w - image.shape[1]
        else:
            self.pad_w = 0

        image = np.lib.pad(image, ((0, self.pad_h), (0, self.pad_w), (0, 0)), 'constant', constant_values=0)

        self.update_variables(image)
        img_h = self.img_h
        img_w = self.img_w

        self.num_patches_img_h = math.ceil((img_h - img_patch_h) / stride_h + 1)
        self.num_patches_img_w = math.ceil(((img_w - img_patch_w) / stride_w + 1))
        num_patches_img = self.num_patches_img_h*self.num_patches_img_w
        self.num_patches_img = num_patches_img
        iter_tot = 0
        img_patches = np.zeros((num_patches_img, 512, 512, image.shape[2]), dtype=image.dtype)
        
        for h in range(int(math.ceil((img_h - img_patch_h) / stride_h + 1))):
            for w in range(int(math.ceil((img_w - img_patch_w) / stride_w + 1))):
                start_h = h * stride_h
                end_h = (h * stride_h) + img_patch_h
                start_w = w * stride_w
                end_w = (w * stride_w) + img_patch_w
                if end_h > img_h:
                    start_h = img_h - img_patch_h
                    end_h = img_h

                if end_w > img_w:
                    start_w = img_w - img_patch_w
                    end_w = img_w


                img_patches[iter_tot, :, :, :] = cv2.resize(image[start_h:end_h, start_w:end_w, :], (512, 512))
                iter_tot += 1

        return img_patches


    def merge_patches(self, patches):
        img_h = self.img_h
        img_w = self.img_w
        img_patch_h = self.img_patch_h
        img_patch_w = self.img_patch_w
        label_patch_h = self.label_patch_h
        label_patch_w = self.label_patch_w
        stride_h = self.stride_h
        stride_w = self.stride_w
        num_patches_img = self.num_patches_img
        assert num_patches_img == patches.shape[0], 'Number of Patches do not match'
        image = np.zeros((img_h, img_w, patches.shape[3]), dtype=float)
        sum_c = np.zeros((img_h, img_w, patches.shape[3]), dtype=float)
        iter_tot = 0
        for h in range(int(math.ceil((img_h - img_patch_h) / stride_h + 1))):
            for w in range(int(math.ceil((img_w - img_patch_w) / stride_w + 1))):
                start_h = h * stride_h
                end_h = (h * stride_h) + img_patch_h
                start_w = w * stride_w
                end_w = (w * stride_w) + img_patch_w
                if end_h > img_h:
                    start_h = img_h - img_patch_h
                    end_h = img_h

                if end_w > img_w:
                    start_w = img_w - img_patch_w
                    end_w = img_w

                if self.label_diff_pad_h == 0 and self.label_diff_pad_w == 0:
                    image[start_h:end_h, start_w:end_w, :] +=cv2.resize(patches[iter_tot, :, :,:], (img_patch_h, img_patch_w))
                    sum_c[start_h:end_h, start_w:end_w, :] += 1.0
                else:
                    image[
                        start_h+self.label_diff_pad_h:start_h + label_patch_h + self.label_diff_pad_h,
                        start_w+self.label_diff_pad_w:start_w + label_patch_w + self.label_diff_pad_w] += \
                        patches[iter_tot, :, :]
                    sum_c[
                        start_h+self.label_diff_pad_h:start_h + label_patch_h + self.label_diff_pad_h,
                        start_w+self.label_diff_pad_w:start_w + label_patch_w + self.label_diff_pad_w] += 1.0
                iter_tot += 1

        if self.pad_h != 0 and self.pad_w != 0:
            sum_c = sum_c[:-self.pad_h, :-self.pad_w, :]
            image = image[:-self.pad_h, :-self.pad_w, :]

        if self.pad_h == 0 and self.pad_w != 0:
            sum_c = sum_c[:, :-self.pad_w, :]
            image = image[:, :-self.pad_w, :]

        if self.pad_h != 0 and self.pad_w == 0:
            sum_c = sum_c[:-self.pad_h, :, :]
            image = image[:-self.pad_h, :, :]

        assert (np.min(sum_c) >= 1.0)
        image = np.divide(image, sum_c)

        return image


#segformer
model_checkpoint = './model/mit-b3-finetunedBRCA-Artemis4Eval-discovery'
model = TFAutoModelForSemanticSegmentation.from_pretrained(model_checkpoint)
save_dir = './patch512artemis/result_validation_mit-b3-finetunedBRCA-Artemis4Eval-s512-20x512'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

datapath = './patch512artemis/image_validation726'
patch_size = 512
imgs = sorted(glob(os.path.join(datapath, '*.png')))
for im_f in imgs:
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    img_name = os.path.splitext(os.path.basename(im_f))[0]
    if not os.path.exists(os.path.join(save_dir, img_name + '.png')):
        testImgc = np.array(Image.open(im_f))
        testData_c = np.expand_dims(testImgc, axis=0)
        testData_c = testData_c.astype(np.float32)
        testData_c = testData_c / 255.0
        testData_c = tf.transpose(testData_c, (0, 3, 1, 2))

        outData = model.predict(testData_c, batch_size=8, verbose=1)
        logit_temp = np.array(tf.transpose(outData.logits, (0, 2, 3, 1)))
        merge_output = np.squeeze(logit_temp, axis=0)
        merge_output = merge_output.argmax(axis=2)

        seg_mask = get_colored_segmentation_image(merge_output, 6, colors=class_colors)
        seg_mask = cv2.resize(seg_mask, (testImgc.shape[1], testImgc.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(save_dir, img_name + '.png'), seg_mask)
    else:
        print('Already Processed %s\n' % os.path.join(save_dir, img_name + '.png'))

