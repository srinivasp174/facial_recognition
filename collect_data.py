import os
import cv2
import uuid
import tensorflow as tf
import numpy as np

# Define paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# Create directories if not exist
os.makedirs(POS_PATH, exist_ok=True)
os.makedirs(NEG_PATH, exist_ok=True)
os.makedirs(ANC_PATH, exist_ok=True)

# Data Augmentation
def data_aug(img):
    data = []
    for i in range(9):
        img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1, 2))
        img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1, 3))
        img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100), np.random.randint(100)))
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100), np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=0.9, upper=1, seed=(np.random.randint(100), np.random.randint(100)))
        data.append(img)
    return data

# Collect Images via Webcam
def collect_images():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[120:120 + 250, 200:200 + 250, :]
        cv2.imshow('Image Collection', frame)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgname, frame)

        if cv2.waitKey(1) & 0xFF == ord('p'):
            imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgname, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Augment Data
def augment_data():
    for path in [POS_PATH, ANC_PATH]:
        for file_name in os.listdir(path):
            img_path = os.path.join(path, file_name)
            img = cv2.imread(img_path)
            augmented_images = data_aug(img)
            for image in augmented_images:
                cv2.imwrite(os.path.join(path, '{}.jpg'.format(uuid.uuid1())), image.numpy())

# Move LFW Images to the Negative Path
def process_lfw():
    os.makedirs(NEG_PATH, exist_ok=True)
    for directory in os.listdir('lfw'):
        for file in os.listdir(os.path.join('lfw', directory)):
            EX_PATH = os.path.join('lfw', directory, file)
            NEW_PATH = os.path.join(NEG_PATH, file)
            os.replace(EX_PATH, NEW_PATH)

if __name__ == "__main__":
    collect_images()
    augment_data()
    process_lfw()
