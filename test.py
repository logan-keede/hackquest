import numpy as np
import os
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import BinaryAccuracy
from keras.models import load_model
from keras.layers import Flatten, Dense, Dropout
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=3784)]
    )

# import tensorflow as tf

# Set the TF_GPU_ALLOCATOR environment variable
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# Load the model from the
def custom_loss(y_true, y_pred):
    # Define weights for different classes
    weight_fn = lambda x: 1.0 if x == 0 else 16.4298  # Assign higher weight to false negatives
    y_true_float = tf.cast(y_true, tf.float32)

    binary_loss = tf.keras.losses.binary_crossentropy(y_true_float, y_pred)


    weighted_loss = tf.reduce_mean(tf.multiply(binary_loss, tf.map_fn(weight_fn, y_true)))

    return weighted_loss
model = load_model("binary_classification_model_pretrained.h5", custom_objects={'custom_loss': custom_loss, 'FixedDropout': Dropout(0.5)})


df = pd.read_csv(".\\train\\train.csv")
df['label'] = df['label'].astype(str)
test_dir = ".\\test\\images"

train_dir = ".\\train\\images"
image_width = 240
image_height = 240
img_size = (image_width, image_height)

batch_size = 32

train_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3
)

train_generator = train_gen.flow_from_dataframe(
    directory=train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    dataframe=df,
    x_col="filename",
    y_col = "label",
    subset = "training",
    shuffle = False,
    pickle_safe = True,
    workers = 1
)

valid_generator = train_gen.flow_from_dataframe(
    directory = train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    dataframe = df,
    x_col = "filename",
    y_col = "label",
    subset = "validation",
    shuffle = False,
    pickle_safe = True,
    workers = 1
)
true_labels = valid_generator.classes
labels = list(true_labels)
for i in range(18,19):
    if len(str(i))!=2:
        model.load_weights(f"training\\cp-adam-000{i}.ckpt")
    else:
        model.load_weights(f"training\\cp-adam-00{i}.ckpt")
    predictions = (model.predict(valid_generator).flatten() > 0.5).astype(int)
    predictions = list(predictions)

    class_report = classification_report(true_labels, predictions, target_names=["Class 0", "Class 1"])
    print("Classification Report:\n", class_report)
    # evaluation_result = model.evaluate(valid_generator, batch_size=batch_size)
    # loss, accuracy, false_positives, true_negatives, true_positives, false_negatives = evaluation_result
    # tpr = true_positives / (true_positives + false_negatives)
    # tnr = true_negatives/ (true_negatives+false_positives)
    # print("True Positive Rate (TPR) at the end of fitting:", tpr)
    # print("True Negative Rate (TNR) at the end of fitting:", tnr)
#
# predictions = (model.predict(valid_generator).flatten() > 0.5).astype(int)
#
# true_labels = valid_generator.classes
#
# predictions = list(predictions)
# labels = list(true_labels)

# print("True Negative Rate (TNR):", tnr)
# print("True Positive Rate (TPR):", tpr)
#

class_report = classification_report(true_labels, predictions, target_names=["Class 0", "Class 1"])
print("Classification Report:\n", class_report)
