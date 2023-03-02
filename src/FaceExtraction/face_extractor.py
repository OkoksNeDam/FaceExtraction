from torchvision import transforms
from facenet_pytorch import MTCNN as Facenet_MTCNN  # Image cropping directly.

from src.FaceExtraction.model_outputs_enum import ModelOutputs
from src.FaceExtraction.utils import *


class FaceExtractor:
    def __init__(self, model_result_folder, face_part_classifier_filepath):
        self.model_result_folder = model_result_folder
        self.face_part_classifier_filepath = face_part_classifier_filepath

    def extract_face_from(self, image_to_crop):
        face_parts_classifier = init_face_parts_classifier(filepath=self.face_part_classifier_filepath)

        if image_to_crop.shape[0] < 750 or image_to_crop.shape[1] < 750:
            return ModelOutputs.INCORRECT_RESOLUTION

        check_image_brightness(image_to_crop)

        yolov5_algorithm = init_yolov5()
        detector = init_face_detector("mtcnn")

        if check_people_absence_on_image(image_to_crop, detector) and check_people_absence_on_image(image_to_crop,
                                                                                                    yolov5_algorithm):
            return ModelOutputs.PEOPLE_ABSENCE

        if get_number_of_people_on_image(image_to_crop, yolov5_algorithm) > 1:
            return ModelOutputs.SEVERAL_PEOPLE

        # Cropping a face from an image.
        cropped_face = crop_face_from_image(image_to_crop,
                                            Facenet_MTCNN(image_size=360, select_largest=False, post_process=False))

        if cropped_face is None:
            return ModelOutputs.CLOSED_FACE

        # Get coordinates of face parts.
        face_parts_boxes = get_face_parts_boxes(cropped_face, detector)

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

        left_eye_img = Image.fromarray(left_eye_img.int().numpy().astype('uint8'), 'RGB')
        left_eye_img = transforms.ToTensor()(left_eye_img)
        left_eye_prediction = face_parts_classifier(left_eye_img.unsqueeze(0).float())
        left_eye_prediction_label = torch.argmax(left_eye_prediction)

        right_eye_img = Image.fromarray(right_eye_img.int().numpy().astype('uint8'), 'RGB')
        right_eye_img = transforms.ToTensor()(right_eye_img)
        right_eye_prediction = face_parts_classifier(right_eye_img.unsqueeze(0).float())
        right_eye_prediction_label = torch.argmax(right_eye_prediction)

        mouth_img = Image.fromarray(mouth_img.int().numpy().astype('uint8'), 'RGB')
        mouth_img = mouth_img.resize((100, 100), Image.LANCZOS)
        mouth_img = transforms.ToTensor()(mouth_img)
        mouth_prediction = face_parts_classifier(mouth_img.unsqueeze(0).float())
        mouth_prediction_label = torch.argmax(mouth_prediction)

        nose_img = Image.fromarray(nose_img.int().numpy().astype('uint8'), 'RGB')
        nose_img = transforms.ToTensor()(nose_img)
        nose_prediction = face_parts_classifier(nose_img.unsqueeze(0).float())
        nose_prediction_label = torch.argmax(nose_prediction)

        if right_eye_prediction_label != 0 or left_eye_prediction_label != 0 or \
                mouth_prediction_label != 1 or nose_prediction_label != 2 or \
                left_eye_prediction[0][0] < 1 or right_eye_prediction[0][0] < 1 or mouth_prediction[0][1] < 1 or \
                nose_prediction[0][2] < 1:
            return ModelOutputs.CLOSED_FACE

        return torch.tensor(torch.tensor(result).permute(2, 0, 1).int().numpy())
