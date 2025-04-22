import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from PIL import Image


dataset, info = tfds.load('oxford_flowers102', with_info=True, as_supervised=True)

class_labels = info.features['label'].int2str


model = load_model('flower_classifier.keras')


def process_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0
    tensor_img = tf.convert_to_tensor(img)
    return tensor_img


image_path = './test_images/hard-leaved_pocket_orchid.jpg'
im = Image.open(image_path)
test_image = np.asarray(im)

processed_test_image = process_image(test_image)

fig, (ax1, ax2) = plt.subplots(figsize=(10,10), ncols=2)
ax1.imshow(test_image)
ax1.set_title('Original Image')
ax2.imshow(processed_test_image)
ax2.set_title('Processed Image')
plt.tight_layout()
plt.show()


def predict(image_path, model, top_k):
    processed_img = process_image(image_path)
    processed_img = np.expand_dims(processed_img.numpy(), axis=0)
    predictions = model.predict(processed_img)
    top_k_indices = np.argsort(predictions[0])[-top_k:][::-1]
    top_k_probs = predictions[0][top_k_indices]
    top_k_classes = [str(i) for i in top_k_indices]
    return top_k_classes, top_k_probs


def plot_with_predictions(image_path, model, top_k):
    img = process_image(image_path)

    top_k_classes, top_k_probs = predict(image_path,
                                         model, top_k,
                                         class_labels)

    disp_img = plt.imread(image_path)

    plt.figure(figsize=(8, 6))

    plt.imshow(disp_img)
    plt.axis('off')

    plt.title(f"Top {top_k} Predictions", fontsize=16)

    for i in range(top_k):
        plt.text(10, 220 + i * 30,
                 f"{i + 1}: Class {top_k_classes[i]} - Probability: {top_k_probs[i]:.4f}",
                 fontsize=12, color='white', backgroundcolor='black')
    plt.show()