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
    tensor_img = tf.expand_dims(tensor_img, axis=0)  
    return tensor_img


image_path = './test_images/hard-leaved_pocket_orchid.jpg'
im = Image.open(image_path)
test_image = np.asarray(im)

processed_test_image = process_image(image_path) 

fig, (ax1, ax2) = plt.subplots(figsize=(10,10), ncols=2)
ax1.imshow(test_image)
ax1.set_title('Original Image')
ax2.imshow(processed_test_image)
ax2.set_title('Processed Image')
plt.tight_layout()
plt.show()


def predict(image_path, model, top_k=5):
    processed_image = process_image(image_path)
    predictions = model.predict(processed_image)
    top_k_indices = np.argsort(predictions[0])[-top_k:][::-1]
    top_k_values = predictions[0][top_k_indices]
    
    return top_k_indices, top_k_values


def plot_predictions(image_path, model, class_names, top_k=5):
    processed_image = process_image(image_path)

    top_k_indices, top_k_values = predict(image_path, model, top_k)

    plt.imshow(Image.open(image_path))  
    plt.axis('off')

    plt.title("Top K Predictions:")
    for i in range(len(top_k_indices)):
        plt.text(0, i * 20, f"{class_names[top_k_indices[i]]}: {top_k_values[i]:.2f}", fontsize=12)
        
    plt.show()


