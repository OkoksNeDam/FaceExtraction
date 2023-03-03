import pytest

from src.FaceExtraction.face_extractor import FaceExtractor
from src.FaceExtraction.model_outputs_enum import ModelOutputs

# Running all tests.
# python -m pytest src/tests/test_face_extractor.py


@pytest.fixture
def face_extractor():
    return FaceExtractor(face_part_classifier_filepath="downloaded_models/face_part_classifier.pt")


def test_incorrect_extension(face_extractor):
    image_path = "test_images/incorrect_extension/test.txt"
    model_output = face_extractor.extract_face_from(image_path)

    assert isinstance(model_output, ModelOutputs) and model_output.value == "Файл должен иметь расширение jpg."


def test_incorrect_resolution(face_extractor):
    image_path = "src/tests/test_images/incorrect_resolution/1.jpg"
    model_output = face_extractor.extract_face_from(image_path)

    assert isinstance(model_output, ModelOutputs) and \
           model_output.value == "Разрешение изображения должно быть не меньше, чем 750x750."
