import tensorflow as tf
import tensorflow_hub as hub

def load_and_tune(data_dir, img_height, img_width, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

def model_select(model, img_height, img_width):

    model_list =['EfficientNet']

    if model in model_list:
        if model == 'EfficientNet':
            # URL EfficientNet
            model_url = "https://tfhub.dev/google/efficientnet/b0/feature-vector/1"

            # Model declaring
            model = tf.keras.Sequential([
                hub.KerasLayer(model_url, input_shape=(img_height, img_width, 3)),
                tf.keras.layers.Dense(8, activation='softmax')
            ])
            return model
    
    else:
        raise Exception("Model not implemented!")