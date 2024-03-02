import tensorflow as tf
from tensorflow.keras import layers

def create_generator_model():
    # Создаем пустую модель Sequential
    model = tf.keras.Sequential()

    # Добавляем полносвязный слой с размером выходного тензора 7*7*256 и без использования bias
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())  # Добавляем слой BatchNormalization для нормализации данных
    model.add(layers.LeakyReLU())  # Добавляем слой LeakyReLU для активации

    # Изменяем форму тензора на (7, 7, 256)
    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Проверяем правильность размера выходного тензора

    # Добавляем слой Conv2DTranspose с 128 фильтрами, ядром 5x5, шагом 1 и сохраняя размер, без использования bias
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)  # Проверяем правильность размера выходного тензора
    model.add(layers.BatchNormalization())  # Добавляем слой BatchNormalization
    model.add(layers.LeakyReLU())  # Добавляем слой LeakyReLU

    # Добавляем слой Conv2DTranspose с 64 фильтрами, ядром 5x5, шагом 2 и сохраняя размер, без использования bias
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)  # Проверяем правильность размера выходного тензора
    model.add(layers.BatchNormalization())  # Добавляем слой BatchNormalization
    model.add(layers.LeakyReLU())  # Добавляем слой LeakyReLU

    # Добавляем слой Conv2DTranspose с 1 фильтром, ядром 5x5, шагом 2 и сохраняя размер, без использования bias, и функцией активации tanh
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)  # Проверяем правильность размера выходного тензора

    return model  # Возвращаем созданную модель