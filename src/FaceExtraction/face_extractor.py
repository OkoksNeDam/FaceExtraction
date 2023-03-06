from facenet_pytorch import MTCNN as Facenet_MTCNN  # Image cropping directly.

from src.FaceExtraction.utils import *


class FaceExtractor:

    MIN_IMAGE_WIDTH = 700
    MIN_IMAGE_HEIGHT = 700

    def __init__(self, face_part_classifier_filepath):
        # Loading face part classifier.
        self.face_parts_classifier = load_face_parts_classifier(filepath=face_part_classifier_filepath)

        self.yolov5_algorithm = init_yolov5()
        self.face_parts_detector = init_face_detector("mtcnn")

    def extract_face_from(self, filepath):
        """
        A function that crops a person's face from an image.

        :param filepath:path to image.
        :return:tensor that stores the cut out image of the face.
        """

        # Checking if a file has a .jpg extension.
        if get_file_extension(filepath) not in [".jpg", ".jpeg"]:
            return ModelOutputs.INCORRECT_EXTENSION

        image = cv2.imread(filepath)

        # Check bounds of input image.
        # Should be higher than MIN_IMAGE_WIDTH and MIN_IMAGE_HEIGHT.
        if image.shape[0] < FaceExtractor.MIN_IMAGE_WIDTH or image.shape[1] < FaceExtractor.MIN_IMAGE_HEIGHT:
            return ModelOutputs.INCORRECT_RESOLUTION

        # Check if image is too bright or dark.
        image_brightness = check_image_brightness(image)
        if isinstance(image_brightness, ModelOutputs):
            return image_brightness

        # The case when there are no people in the image is not suitable.
        # Apply MTCNN and yolov5.
        if check_people_absence_on_image(image, self.face_parts_detector) and \
           check_people_absence_on_image(image, self.yolov5_algorithm):
            return ModelOutputs.PEOPLE_ABSENCE

        # The case when there are several people in the image is not suitable.
        if get_number_of_people_on_image(image, self.yolov5_algorithm) > 1:
            return ModelOutputs.SEVERAL_PEOPLE

        # Cropping a face from an image using MTCNN algorithm.
        cropped_face = crop_face_from_image(image,
                                            Facenet_MTCNN(image_size=360, select_largest=False, post_process=False))

        # If cropped_face is None than there were no detections of face, but person is on the image.
        if cropped_face is None:
            return ModelOutputs.CLOSED_FACE

        # Get coordinates of face parts.
        face_parts_boxes = get_face_parts_boxes(cropped_face, self.face_parts_detector)

        # If face_parts_boxes is None than there were no detections of face, but person is on the image.
        if not face_parts_boxes:
            return ModelOutputs.CLOSED_FACE

        # Changing coordinates of face parts to crop them.
        mouth_top = change_coordinate(face_parts_boxes[2], 0, 50, 0, 360)
        mouth_bottom = change_coordinate(face_parts_boxes[3], 0, -50, 0, 360)
        left_eye_top = change_coordinate(face_parts_boxes[0], -50, -50, 0, 360)
        left_eye_bottom = change_coordinate(face_parts_boxes[0], 50, 50, 0, 360)
        right_eye_top = change_coordinate(face_parts_boxes[1], -50, -50, 0, 360)
        right_eye_bottom = change_coordinate(face_parts_boxes[1], 50, 50, 0, 360)
        nose_top = change_coordinate(face_parts_boxes[4], -50, -70, 0, 360)
        nose_bottom = change_coordinate(face_parts_boxes[4], 50, 30, 0, 360)

        result = torch.tensor(torch.tensor(cropped_face).permute(1, 2, 0).int().numpy())

        left_eye_img = result[left_eye_top[1]:left_eye_bottom[1], left_eye_top[0]:left_eye_bottom[0]]
        right_eye_img = result[right_eye_top[1]:right_eye_bottom[1], right_eye_top[0]:right_eye_bottom[0]]
        mouth_img = result[mouth_bottom[1]:mouth_top[1], mouth_top[0]:mouth_bottom[0]]
        nose_img = result[nose_top[1]:nose_bottom[1], nose_top[0]:nose_bottom[0]]

        left_eye_pred, left_eye_prediction_label = get_face_part_prediction(left_eye_img, self.face_parts_classifier)
        right_eye_pred, right_eye_prediction_label = get_face_part_prediction(right_eye_img, self.face_parts_classifier)
        mouth_pred, mouth_prediction_label = get_face_part_prediction(mouth_img, self.face_parts_classifier)
        nose_pred, nose_prediction_label = get_face_part_prediction(nose_img, self.face_parts_classifier)

        if right_eye_prediction_label != 0 or left_eye_prediction_label != 0 or \
                mouth_prediction_label != 1 or nose_prediction_label != 2 or \
                left_eye_pred[0][0] < 1 or right_eye_pred[0][0] < 1 or mouth_pred[0][1] < 1 or \
                nose_pred[0][2] < 1:
            return ModelOutputs.CLOSED_FACE

        return torch.tensor(torch.tensor(result).permute(2, 0, 1).int().numpy())
