from enum import Enum


class ModelOutputs(str, Enum):
    INCORRECT_EXTENSION = "Файл должен иметь расширение jpg.",
    INCORRECT_RESOLUTION = "Разрешение изображения должно быть не меньше, чем 700x700.",
    PEOPLE_ABSENCE = "На фотографии нет людей. Сделайте снимок, где есть ваше лицо.",
    DARK_LIGHTING = "Слишком темное изображение. Сделайте снимок ещё раз.",
    BRIGHT_LIGHTING = "Слишком яркое изображение. Сделайте снимок ещё раз.",
    SEVERAL_PEOPLE = "На фотографии несколько людей. Сделайте снимок, на котором находитесь только вы.",
    CLOSED_FACE = "Лицо не видно полностью, сделайте снимок ещё раз."
