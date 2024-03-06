import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
import efficientnet.keras as efn

from sklearn.metrics import confusion_matrix
import os

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

df = pd.read_csv(".\\train\\train.csv")
df['label'] = df['label'].astype(str)
test_dir = ".\\test\\images"

train_dir = ".\\train\\images"
image_width = 240
image_height = 240
img_size = (image_width, image_height)
learning_rate = 0.0000075
batch_size = 32
epochs = 20

train_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    validation_split=0.4,
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
    # shuffle = False,
    # pickle_safe = True,
    # workers = 1
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
    # pickle_safe = True,
    # workers = 1
)

base_model = efn.EfficientNetB1(input_shape = (240, 240, 3), include_top = False, weights = 'imagenet')
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)

predictions = Dense(1, activation="sigmoid")(x)
model_final = Model(inputs = base_model.input, outputs = predictions)

def custom_loss(y_true, y_pred):
    weight_fn = lambda x: 1.0 if x == 0 else 15.789
    y_true_float = tf.cast(y_true, tf.float32)
    binary_loss = tf.keras.losses.binary_crossentropy(y_true_float, y_pred)
    weighted_loss = tf.reduce_mean(tf.multiply(binary_loss, tf.map_fn(weight_fn, y_true)))
    return weighted_loss

model_final.compile(optimizer=optimizers.Adam(learning_rate = learning_rate, decay=1e-6), loss=custom_loss, metrics=['accuracy',tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.TruePositives(),tf.keras.metrics.FalseNegatives()])

checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_freq='epoch',
                                                 verbose=1)


model_final.fit(train_generator, epochs=20, validation_data=valid_generator,callbacks=[cp_callback])
evaluation_result = model_final.evaluate(valid_generator, batch_size=batch_size)
loss, accuracy, false_positives, true_negatives, true_positives, false_negatives = evaluation_result
tpr = true_positives / (true_positives + false_negatives)


print("True Positive Rate (TPR) at the end of fitting:", tpr)
model_final.save("binary_classification_model_pretrained.h5")
predictions = (model_final.predict(valid_generator).flatten() > 0.5).astype(int)

true_labels = valid_generator.classes
conf_matrix = confusion_matrix(true_labels, predictions)
tn, fp, fn, tp = conf_matrix.ravel()

tnr = tn / (tn + fp)
tpr = tp / (tp + fn)
print("True Negative Rate (TNR):", tnr)
print("True Positive Rate (TPR):", tpr)

df = pd.read_csv(".\\test\\test.csv")
df[''] = df['image_id'].astype(str)
test_dir = ".\\test\\images"

train_dir = ".\\train\\images"

train_gen = ImageDataGenerator(
    rescale=1./255,
)

train_generator = train_gen.flow_from_dataframe(
    directory=test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode=None,
    dataframe=df,
    x_col="filename",
    y_col=None,
    shuffle = False,
    # pickle_safe = True,
    # workers = 1
)

features = model_final.predict(train_generator)
# features = (model.predict(train_generator).flatten() > 0.5).astype(int)
print(features.shape)
features_df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])


result_df = pd.concat([df, features_df], axis=1)
result_df["label"] = result_df["feature_0"]
result_df["image_id"] = result_df[""]
result_df = result_df[["image_id", "label"]]
result_df["label"] = (result_df["label"]>0.5).astype(int)
result_df.to_csv('predictions_with_data.csv', index=False)