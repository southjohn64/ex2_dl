import tensorflow as tf
import os
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.layers import Lambda
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense

print('TF version installed: ', tf.__version__)


class SiameseNet:
    def __init__(self, input_shape, num_train_samples, batch_size):
        '''

        :param image_shape: (w,h,c)
        '''

        # params on net
        self.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=10e-2)
        self.bias_initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=10e-2)
        self.fully_connected_kernel_initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=2 * 10e-1)

        self.kernel_regularizer = regularizers.l2(2 * 1e-1)

        self.initial_learning_rate = 1e-4
        self.learning_rate_decay = 0.99
        self.decay_step = num_train_samples // batch_size

        #input_shape = (250, 250, 1)
        input_image_siam1 = tf.keras.Input(input_shape)
        input_image_siam2 = tf.keras.Input(input_shape)

        model = models.Sequential()

        # start of one siam structure
        model.add(layers.Conv2D(64, (10, 10), activation='relu', input_shape=input_shape,
                                kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
                                kernel_regularizer=self.kernel_regularizer, name='Layer_1'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(128, (7, 7), activation='relu', input_shape=(48, 48, 64),
                                kernel_initializer=self.kernel_initializer,
                                bias_initializer=self.bias_initializer,
                                kernel_regularizer=self.kernel_regularizer, name='Layer_2'))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(128, (4, 4), activation='relu', input_shape=(21, 21, 128),
                                kernel_initializer=self.kernel_initializer,
                                bias_initializer=self.bias_initializer,
                                kernel_regularizer=self.kernel_regularizer))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Conv2D(256, (4, 4), activation='relu', input_shape=(9, 9, 128),
                                kernel_initializer=self.kernel_initializer,
                                bias_initializer=self.bias_initializer,
                                kernel_regularizer=self.kernel_regularizer))
        model.add(layers.MaxPooling2D((2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation='sigmoid',
                               kernel_initializer=self.fully_connected_kernel_initializer,
                               bias_initializer=self.bias_initializer,
                               kernel_regularizer=self.kernel_regularizer,
                               name="layer_dense_4096"))

        ############ END OF SIAM CNN PART  ######################################################################

        # split to 2 siams net
        emb1 = model(input_image_siam1)
        emb2 = model(input_image_siam2)

        # calc the dist of embbedings
        Emb_dist_layer = Lambda(lambda embeddings: tf.math.abs(embeddings[0] - embeddings[1]))
        Emb_distance = Emb_dist_layer([emb1, emb2])

        # calc [0-1] for the distance , this is what we are learning
        outputs = Dense(1, activation='sigmoid',
                        kernel_initializer=self.fully_connected_kernel_initializer,
                        bias_initializer=self.bias_initializer)(Emb_distance)

        # consolidate the 2 siames to one model with 2 inputs
        self.siam_model = models.Model([input_image_siam1, input_image_siam2], outputs)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.initial_learning_rate,
            decay_steps=self.decay_step,
            decay_rate=self.learning_rate_decay,
            staircase=True)

        self.siam_model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
                                loss=tf.keras.losses.BinaryCrossentropy(),
                                metrics=['accuracy'],
                                )

        self.siam_model.summary()

    def train(self, train_dataset, y_array_train, val_dataset, y_array_val, save_model_path, batch_size=2, epoch=200):
        #batches_per_epoch = len(train_dataset[0])//batch_size
        early_stopping_callback = EarlyStopping(monitor='loss', patience=20)
        # change mode path if you are in linux or windows
        file_path = os.path.join(save_model_path,'model.{epoch:02d}-{val_loss:.4f}.h5')
        checkpoint_callback = ModelCheckpoint(file_path,
                                              monitor='val_loss', verbose=1,
                                              save_best_only=True, mode='min')#, save_freq=batches_per_epoch*2)

        history = self.siam_model.fit(x=train_dataset, y=y_array_train, batch_size=batch_size, epochs=epoch,
                                      validation_data=(val_dataset, y_array_val),
                                      callbacks=[early_stopping_callback, checkpoint_callback])
