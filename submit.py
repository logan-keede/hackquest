import numpy as np
import os
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
gpus = tf.config.list_physical_devices('GPU')
from keras.layers import Flatten, Dense, Dropout
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=3784)]
    )


os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def custom_loss(y_true, y_pred):
    # Define weights for different classes
    weight_fn = lambda x: 1.0 if x == 0 else 15.789  # Assign higher weight to false negatives
    y_true_float = tf.cast(y_true, tf.float32)

    binary_loss = tf.keras.losses.binary_crossentropy(y_true_float, y_pred)


    weighted_loss = tf.reduce_mean(tf.multiply(binary_loss, tf.map_fn(weight_fn, y_true)))

    return weighted_loss
model = load_model("binary_classification_model_pretrained.h5", custom_objects={'custom_loss': custom_loss, "FixedDropout": Dropout(0.5)})
# for i in range(19,21):
#     if len(str(i))!=2:
#         model.load_weights(f"training\\cp-adam-000{i}.ckpt")
#     else:
#         model.load_weights(f"training\\cp-adam-00{i}.ckpt")
model.load_weights(f"training\\cp-adam-0011.ckpt")
df = pd.read_csv(".\\test\\test.csv")
df[''] = df['image_id'].astype(str)
test_dir = ".\\test\\images"

train_dir = ".\\train\\images"
image_width = 240
image_height = 240
img_size = (image_width, image_height)

batch_size = 64

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
    y_col=None
)

features = model.predict(train_generator)
# features = (model.predict(train_generator).flatten() > 0.5).astype(int)
print(features.shape)
features_df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])


result_df = pd.concat([df, features_df], axis=1)
result_df["label"] = result_df["feature_0"]
result_df["image_id"] = result_df[""]
result_df = result_df[["image_id", "label"]]
result_df["label"] = (result_df["label"]>0.5).astype(int)
result_df.to_csv('predictions_with_data.csv', index=False)
