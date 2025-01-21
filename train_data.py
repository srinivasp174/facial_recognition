from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
from preprocess_data import load_data

# Build Embedding Model
def make_embedding():
    inp = Input(shape=(105, 105, 3), name='input_image')
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    return Model(inputs=[inp], outputs=[d1], name='embedding')

# Custom Distance Layer
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

# Build Siamese Model
def make_siamese_model():
    input_image = Input(name='input_image', shape=(105, 105, 3))
    validation_image = Input(name='validation_image', shape=(105, 105, 3))
    siamese_layer = L1Dist()
    distances = siamese_layer(make_embedding()(input_image), make_embedding()(validation_image))
    classifier = Dense(1, activation='sigmoid')(distances)
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='siamese_network')

# Training and Evaluation
if __name__ == "__main__":
    train_data, test_data = load_data()
    siamese_model = make_siamese_model()
    siamese_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    siamese_model.fit(train_data, epochs=10, validation_data=test_data)
    siamese_model.save('siamesemodel.h5')
