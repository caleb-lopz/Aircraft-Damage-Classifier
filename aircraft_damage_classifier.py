"""
Aircraft Damage Classification and Captioning
=============================================
Part 1: Binary classification (dent vs crack) using VGG16 feature extraction.
Part 2: Image captioning and summarization using the BLIP pretrained model.

Dataset: https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar
"""

# ── Dependencies ──────────────────────────────────────────────────────────────
# pip install tensorflow_cpu==2.17.1 pillow matplotlib transformers
# pip install torch==2.2.0+cpu torchvision==0.17.0+cpu torchaudio==2.2.0+cpu \
#     --index-url https://download.pytorch.org/whl/cpu

import os
import warnings
import random
import tarfile
import urllib.request
import shutil
import matplotlib

warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tf_keras as keras
from tf_keras.models import Sequential, Model
from tf_keras.layers import Dense, Dropout, Flatten
from tf_keras.applications import VGG16
from tf_keras.optimizers import Adam
from tf_keras.preprocessing.image import ImageDataGenerator

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


# ── Reproducibility ───────────────────────────────────────────────────────────
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

# ── 1.1 Dataset Preparation ───────────────────────────────────────────────────
batch_size = 32
n_epochs = 20
img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar"
tar_filename = "aircraft_damage_dataset_v1.tar"
extracted_folder = "aircraft_damage_dataset_v1"

urllib.request.urlretrieve(url, tar_filename)
print(f"Downloaded {tar_filename}.")

if os.path.exists(extracted_folder):
    shutil.rmtree(extracted_folder)

with tarfile.open(tar_filename, "r") as tar_ref:
    tar_ref.extractall()
print(f"Extracted {tar_filename}.")

extract_path = "aircraft_damage_dataset_v1"
train_dir = os.path.join(extract_path, "train")
valid_dir = os.path.join(extract_path, "valid")
test_dir  = os.path.join(extract_path, "test")


# ── 1.2 Data Preprocessing ────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
valid_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen  = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    seed=seed_value,
    class_mode="binary",
    shuffle=True,
)

# Task 1
valid_generator = valid_datagen.flow_from_directory(
    directory=valid_dir,
    class_mode="binary",
    seed=seed_value,
    batch_size=batch_size,
    shuffle=False,
    target_size=(img_rows, img_cols),
)

# Task 2
test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    class_mode="binary",
    seed=seed_value,
    batch_size=batch_size,
    shuffle=False,
    target_size=(img_rows, img_cols),
)


# ── 1.3 Model Definition ──────────────────────────────────────────────────────
# Task 3
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(img_rows, img_cols, 3))

output = base_model.layers[-1].output
output = keras.layers.Flatten()(output)
base_model = Model(base_model.input, output)

for layer in base_model.layers:
    layer.trainable = False

model = Sequential()
model.add(base_model)
model.add(Dropout(0.3))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(1, activation="sigmoid"))

# Task 4
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)




def main():
    # ── 1.4 Model Training ─────────────────────────────────────────────────────
    # Task 5
    try:
        steps_per_epoch = math.ceil(train_generator.samples / batch_size)
        validation_steps = math.ceil(valid_generator.samples / batch_size)

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1,
        )

        history = model.fit(
            train_generator,
            epochs=n_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=valid_generator,
            validation_steps=validation_steps,
            callbacks=[early_stopping],
            verbose=1,
            workers=1,
            use_multiprocessing=False,
        )

        train_history = history.history

        # ── 1.5 Visualizing Training Results ────────────────────────────────────
        plt.title("Training Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.plot(train_history["loss"])
        plt.show()

        plt.title("Validation Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.plot(train_history["val_loss"])
        plt.show()

        # Task 6
        plt.figure(figsize=(5, 5))
        plt.plot(train_history["accuracy"], label="Training Accuracy")
        plt.plot(train_history["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.legend()
        plt.show()

        # ── 1.6 Model Evaluation ───────────────────────────────────────────────
        test_steps = math.ceil(test_generator.samples / test_generator.batch_size)
        test_loss, test_accuracy = model.evaluate(
            test_generator,
            steps=test_steps,
        )
        print(f"Test Loss:     {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # ── 1.7 Visualizing Predictions ────────────────────────────────────────
        def plot_image_with_title(img_array, true_label, predicted_label, class_names):
            plt.figure(figsize=(6, 6))
            plt.imshow(img_array)
            plt.title(
                f"True: {class_names[true_label]}\nPred: {class_names[predicted_label]}"
            )
            plt.axis("off")
            plt.show()


        def test_model_on_image(test_gen, clf_model, index_to_plot=0):
            test_images, test_labels = next(test_gen)
            predictions = clf_model.predict(test_images)
            predicted_classes = (predictions > 0.5).astype(int).flatten()

            class_names = {v: k for k, v in test_gen.class_indices.items()}

            plot_image_with_title(
                img_array=test_images[index_to_plot],
                true_label=int(test_labels[index_to_plot]),
                predicted_label=predicted_classes[index_to_plot],
                class_names=class_names,
            )


        # Task 7
        test_model_on_image(test_generator, model, index_to_plot=1)

    except Exception:
        import traceback

        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — IMAGE CAPTIONING WITH BLIP
# ─────────────────────────────────────────────────────────────────────────────

# ── 2.1 Loading BLIP Model ────────────────────────────────────────────────────
processor = None
blip_model = None


class BlipCaptionSummaryLayer(tf.keras.layers.Layer):
    def __init__(self, blip_processor, blip_gen_model, **kwargs):
        super().__init__(**kwargs)
        self.processor = blip_processor
        self.model = blip_gen_model

    def call(self, image_path, task):
        return tf.py_function(self.process_image, [image_path, task], tf.string)

    def process_image(self, image_path, task):
        try:
            image_path_str = image_path.numpy().decode("utf-8")
            img = Image.open(image_path_str).convert("RGB")

            if task.numpy().decode("utf-8") == "caption":
                prompt = "This is a picture of"
            else:
                prompt = "This is a detailed photo showing"

            inputs = self.processor(images=img, text=prompt, return_tensors="pt")
            output = self.model.generate(**inputs)
            return self.processor.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Error: {e}")
            return "Error processing image"


# Task 8
def generate_text(image_path, task):
    blip_layer = BlipCaptionSummaryLayer(processor, blip_model)
    return blip_layer(image_path, task)


def run_blip_demo():
    global processor, blip_model

    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    sample_path = tf.constant(
        "aircraft_damage_dataset_v1/test/dent/144_10_JPG_jpg.rf.4d008cc33e217c1606b76585469d626b.jpg"
    )

    caption = generate_text(sample_path, tf.constant("caption"))
    print("Caption:", caption.numpy().decode("utf-8"))

    summary = generate_text(sample_path, tf.constant("summary"))
    print("Summary:", summary.numpy().decode("utf-8"))

    # Task 9 — caption for task image
    task_image_path = tf.constant(
        "aircraft_damage_dataset_v1/test/dent/149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg"
    )

    img_display = plt.imread(task_image_path.numpy().decode("utf-8"))
    plt.imshow(img_display)
    plt.axis("off")
    plt.show()

    caption_task = generate_text(task_image_path, tf.constant("caption"))
    print("Task 9 Caption:", caption_task.numpy().decode("utf-8"))

    # Task 10 — summary for task image
    summary_task = generate_text(task_image_path, tf.constant("summary"))
    print("Task 10 Summary:", summary_task.numpy().decode("utf-8"))


if __name__ == "__main__":
    main()
