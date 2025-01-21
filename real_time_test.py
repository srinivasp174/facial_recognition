import cv2
import numpy as np
import tensorflow as tf
from preprocess_data import preprocess

def verify(model, detection_threshold, verification_threshold):
    results = []
    for image in os.listdir('application_data/verification_images'):
        input_img = preprocess('application_data/input_image/input_image.jpg')
        validation_img = preprocess(f'application_data/verification_images/{image}')
        result = model.predict(np.expand_dims([input_img, validation_img], axis=1))
        results.append(result)

    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(results)
    return verification > verification_threshold

def real_time_test(model):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[120:120 + 250, 200:200 + 250, :]
        cv2.imshow('Verification', frame)

        if cv2.waitKey(10) & 0xFF == ord('v'):
            cv2.imwrite('application_data/input_image/input_image.jpg', frame)
            verified = verify(model, 0.5, 0.7)
            print(f'Verified: {verified}')

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist': L1Dist})
    real_time_test(model)
