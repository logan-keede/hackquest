import numpy as np
import os
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
# import tensorflow as tf
from keras.callbacks import LearningRateScheduler
import os
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=3784)]
    )

# import tensorflow as tf

# Set the TF_GPU_ALLOCATOR environment variable
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
df = pd.read_csv(".\\train\\train.csv")
df['label'] = df['label'].astype(str)
test_dir = ".\\test\\images"

train_dir = ".\\train\\images"
image_width = 256
image_height = 256
img_size = (image_width, image_height)

batch_size = 32

train_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
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
)


#
# test_gen = ImageDataGenerator(rescale=1./255)

# test_generator = test_gen.flow_from_directory(
#     test_dir,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode='binary'
# )

model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(
            32, (4, 4), activation="relu", input_shape=(image_width, image_height, 3)
        ),

        tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),

        tf.keras.layers.Conv2D(
            32, (4, 4), activation="relu",
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),
        #
        tf.keras.layers.Conv2D(
            32, (4, 4), activation="relu",
        ),

        tf.keras.layers.MaxPooling2D(pool_size=(4, 4)),

        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(2084, activation="relu"),
        # tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(1, activation="hard_sigmoid"),
        ])


def custom_loss(y_true, y_pred):
    # Define weights for different classes
    weight_fn = lambda x: 1.0 if x == 0 else 15.789  # Assign higher weight to false negatives
    y_true_float = tf.cast(y_true, tf.float32)

    binary_loss = tf.keras.losses.binary_crossentropy(y_true_float, y_pred)


    weighted_loss = tf.reduce_mean(tf.multiply(binary_loss, tf.map_fn(weight_fn, y_true)))

    return weighted_loss

# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, decay=1e-5)
def lr_schedule(epoch, initial_lr=0.0001, end_lr=0.00001):
    return initial_lr - ((initial_lr-end_lr)/8 * epoch)

# Create a Learning Rate Scheduler
lr_scheduler = LearningRateScheduler(lr_schedule)
initial_learning_rate = 0.0001
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

# optimizer = tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
model.compile(optimizer="adam", loss=custom_loss, metrics=['accuracy',tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.TruePositives(),tf.keras.metrics.FalseNegatives()])
model.fit(train_generator, epochs=8, validation_data=valid_generator,callbacks=[lr_scheduler])
model.evaluate(valid_generator, batch_size=batch_size)
# Save the model to a file
model.save("binary_classification_model1.h5")



df = pd.read_csv(".\\test\\test.csv")
df[''] = df['image_id'].astype(str)
test_dir = ".\\test\\images"

train_dir = ".\\train\\images"
image_width = 256
image_height = 256
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


features_df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])


result_df = pd.concat([df, features_df], axis=1)
result_df["label"] = result_df["feature_0"]
result_df["image_id"] = result_df[""]
result_df = result_df[["image_id", "label"]]
result_df["label"] = (result_df["label"]>0.5).astype(int)

result_df.to_csv('predictions_with_data.csv', index=False)
