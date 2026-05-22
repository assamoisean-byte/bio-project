import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

SPLIT_DIR = "dataset_split"
MODEL_PATH = os.path.join("models", "best_model.keras")
IMG_SIZE = (128, 128)

if not os.path.exists(MODEL_PATH):
    print(f"Best model not found at {MODEL_PATH}. Please train first.")
    raise SystemExit(1)

model = tf.keras.models.load_model(MODEL_PATH)
print(f"Loaded model from {MODEL_PATH}")

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

test_gen = test_datagen.flow_from_directory(
    os.path.join(SPLIT_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# Predict
preds = model.predict(test_gen, verbose=1)
pred_labels = (preds.ravel() > 0.5).astype(int)
true_labels = test_gen.classes

print(classification_report(true_labels, pred_labels, target_names=list(test_gen.class_indices.keys())))

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(test_gen.class_indices.keys()), yticklabels=list(test_gen.class_indices.keys()))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print('Saved confusion matrix to confusion_matrix.png')
