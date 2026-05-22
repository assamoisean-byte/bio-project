import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Paths
RAW_DATASET = "dataset"
SPLIT_DIR = "dataset_split"
MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "classifier_model.h5")

IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)

def prepare_splits(test_size=0.1, val_size=0.1):
    """Create train/val/test splits under `dataset_split/` by copying files.
    If `dataset_split` already exists and is non-empty, this function does nothing.
    """
    if os.path.exists(SPLIT_DIR) and any(os.scandir(SPLIT_DIR)):
        print(f"Using existing splits in {SPLIT_DIR}")
        return

    classes = [d for d in os.listdir(RAW_DATASET) if os.path.isdir(os.path.join(RAW_DATASET, d))]
    if not classes:
        raise SystemExit(f"No class folders found in {RAW_DATASET}. Expected ./dataset/<class>/images")

    print("Preparing dataset splits...")
    for cls in classes:
        src_dir = os.path.join(RAW_DATASET, cls)
        images = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
        if not images:
            continue

        train_and_val, test = train_test_split(images, test_size=test_size, random_state=RANDOM_STATE, stratify=None)
        train, val = train_test_split(train_and_val, test_size=val_size/(1 - test_size), random_state=RANDOM_STATE, stratify=None)

        for split_name, split_list in (("train", train), ("val", val), ("test", test)):
            out_dir = os.path.join(SPLIT_DIR, split_name, cls)
            os.makedirs(out_dir, exist_ok=True)
            for src in split_list:
                shutil.copy2(src, out_dir)

    print("Dataset splits created under dataset_split/ (train/ val/ test)")


def build_model(input_shape=(*IMG_SIZE, 3)):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def get_generators():
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        rescale=1.0/255.0
    )

    test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(SPLIT_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_gen = test_datagen.flow_from_directory(
        os.path.join(SPLIT_DIR, 'val'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    test_gen = test_datagen.flow_from_directory(
        os.path.join(SPLIT_DIR, 'test'),
        target_size=IMG_SIZE,
        batch_size=1,
        class_mode='binary',
        shuffle=False
    )

    return train_gen, val_gen, test_gen


def compute_class_weights(generator):
    labels = generator.classes
    if len(np.unique(labels)) <= 1:
        return None
    weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = {i: w for i, w in enumerate(weights)}
    return class_weights


def main():
    prepare_splits()
    train_gen, val_gen, test_gen = get_generators()

    model = build_model()
    print(model.summary())

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(BEST_MODEL_PATH, monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]

    class_weights = compute_class_weights(train_gen)
    if class_weights:
        print("Using class weights:", class_weights)

    steps_per_epoch = max(1, train_gen.samples // BATCH_SIZE)
    validation_steps = max(1, val_gen.samples // BATCH_SIZE)

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Save final model
    model.save(FINAL_MODEL_PATH)
    print(f"Final model saved to {FINAL_MODEL_PATH}")

    # Also save best model (already saved via checkpoint)
    if os.path.exists(BEST_MODEL_PATH):
        print(f"Best model available at {BEST_MODEL_PATH}")


if __name__ == '__main__':
    main()
