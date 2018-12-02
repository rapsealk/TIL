#!/usr/bin/env python3
"""
코드 5-40: 사전 훈련된 가중치로 VGG16 네트워크 로드하기
"""
from keras.applications.vgg16 import VGG16

model = VGG16(weights='imagenet')

"""
코드 5-41: VGG16을 위해 입력 이미지 전처리하기
"""
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

img_path = './data/creative_commons_elephant.jpg'

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('preds:', preds)

"""
코드 5-42: Grad-CAM 알고리즘 설정하기
"""
import keras.backend as K

# np.argmax(preds[0])
african_elephant_output = model.output[:, 386]

last_conv_layer = model.get_layer('block5_conv3')
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)

"""
코드 5-43: 히트맵 후처리하기
"""
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

import matplotlib.pyplot as plt

plt.matshow(heatmap)

"""
코드 5-44: 원본 이미지에 히트맵 덧붙이기
"""
import cv2

img = cv2.imread(img_path)

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img

cv2.imwrite('./data/elephant_cam.jpg', superimposed_img)