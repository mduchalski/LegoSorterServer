import tensorflow as tf

# create the base pre-trained model
from tensorflow.keras import Model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense


class Inception():
    @staticmethod
    def prepare_model(cls_count, weights=None):
        base_model = InceptionV3(weights=weights or 'imagenet', include_top=False)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)

        predictions = Dense(cls_count, activation='softmax')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.CategoricalAccuracy()])
        return model