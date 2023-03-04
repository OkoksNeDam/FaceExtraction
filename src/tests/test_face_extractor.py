import pytest
import torch

from src.FaceExtraction.face_extractor import FaceExtractor
from src.FaceExtraction.model_outputs_enum import ModelOutputs


# Running all tests.
# python -m pytest src/tests/test_face_extractor.py


@pytest.fixture
def face_extractor():
    """
    Initialization of class FaceExtractor.

    :return:instance of FaceExtractor class.
    """
    return FaceExtractor(face_part_classifier_filepath="downloaded_models/face_part_classifier.pt")


# Data for the function test_extract_face, that is under test.
# Format of data: tuple(image path, expected model output).
tests_for_extract_face = \
    [("test_images/incorrect_extension/test.txt", ModelOutputs.INCORRECT_EXTENSION.value),
     ("src/tests/test_images/incorrect_resolution/1.jpg", ModelOutputs.INCORRECT_RESOLUTION.value),
     ("src/tests/test_images/people_absence/1.jpg", ModelOutputs.PEOPLE_ABSENCE.value),
     ("src/tests/test_images/dark_image/1.jpg", ModelOutputs.DARK_LIGHTING.value),
     ("src/tests/test_images/bright_image/1.jpg", ModelOutputs.BRIGHT_LIGHTING.value),
     ("src/tests/test_images/several_people/1.jpg", ModelOutputs.SEVERAL_PEOPLE.value),
     ("src/tests/test_images/closed_face/1.jpg", ModelOutputs.CLOSED_FACE.value),
     ("src/tests/test_images/standard_face/1.jpg", torch.Size([3, 360, 360]))]


@pytest.mark.parametrize("image_path,expected_model_output", tests_for_extract_face)
def test_extract_face(face_extractor, image_path, expected_model_output):
    """
    Testing the extract_face_from function from the FaceExtractor class.

    :param face_extractor:initialized class FaceExtractor.
    :param image_path:path to test image.
    :param expected_model_output:response expected from the model.
    """
    model_output = face_extractor.extract_face_from(image_path)

    if isinstance(model_output, ModelOutputs):
        assert model_output.value == expected_model_output
