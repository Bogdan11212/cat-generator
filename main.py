import tensorflow as tf
from tensorflow import keras
from flask import Flask, render_template_string
import numpy as np

# Загрузка предварительно обученной модели
model = keras.models.load_model('path/to/your/model.h5')

# Создание веб-сервера
app = Flask(__name__)

# Маршрут для отображения котика
@app.route('/')
def index():
    # Генерация случайных входных данных
    random_input = np.random.rand(1, input_shape)

    # Получение предсказания нейронной сети
    prediction = model.predict(random_input)

    # Преобразование предсказания в HTML-код для отображения изображения котика
    image_html = '<img src="data:image/png;base64,{0}">'.format(prediction)

    return render_template_string(image_html)

if __name__ == "__main__":
    # Запуск веб-сервера
    app.run()
