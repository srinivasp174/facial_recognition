import tensorflow as tf
import os

# Paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# Preprocess Function
def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (105, 105))
    img = img / 255.0
    return img

# Prepare Dataset
def load_data():
    anchor = tf.data.Dataset.list_files(ANC_PATH + '/*.jpg').take(5000)
    positive = tf.data.Dataset.list_files(POS_PATH + '/*.jpg').take(5000)
    negative = tf.data.Dataset.list_files(NEG_PATH + '/*.jpg').take(5000)

    # Label the data
    positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    data = positives.concatenate(negatives)

    # Preprocess the data
    data = data.map(lambda x, y, z: (preprocess(x), preprocess(y), z))
    data = data.shuffle(buffer_size=1024).cache()
    train_data = data.take(round(len(data) * 0.7)).batch(16).prefetch(8)
    test_data = data.skip(round(len(data) * 0.7)).batch(16).prefetch(8)

    return train_data, test_data
