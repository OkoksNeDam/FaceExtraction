from enum import Enum


class ModelOutputs(str, Enum):
    INCORRECT_EXTENSION = "Файл должен иметь расширение jpg.",
    INCORRECT_RESOLUTION = "Разрешение изображения должно быть не меньше, чем 750x750.",
    PEOPLE_ABSENCE = "На фотографии нет людей. Сделайте снимок, где есть ваше лицо.",
    SEVERAL_PEOPLE = "На фотографии несколько людей. Сделайте снимок, на котором находитесь только вы.",
    CLOSED_FACE = "Лицо не видно полностью, сделайте снимок ещё раз."
