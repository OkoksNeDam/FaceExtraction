import glob

from PIL import Image
import torch
from mtcnn import MTCNN  # Used to trim parts of the face.
import yolov5
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from torchvision import transforms

from src.FaceExtraction.ModelBaseline import ModelBaseline
from src.FaceExtraction.model_outputs_enum import ModelOutputs


def init_face_detector(name):
    """
    Initializing face detector.

    :param name: face detector name.
    :return: instance of face detector.
    """
    if name == "mtcnn":
        return MTCNN()


def init_yolov5():
    """
    Initialize yolov5 model.

    :return: instance of yolov5.
    """
    return yolov5.load('../../downloaded_models/yolov5s.pt')


def get_file_extension(filepath):
    """
    Get the extension of the provided file.

    :param filepath:path of the provided file.
    :return:extension of the provided file.
    """
    return os.path.splitext(p=filepath)[1]


def check_image_brightness(image):
    """
    Check if the image is too bright or dark.
    Looks at the ratio of bright pixels to dark ones.

    :param image:image to check,
    :return:ModelOutputs instance if image is too dark of bright, otherwise return True.
    """

    bright_thres = 0.5
    dark_thres = 0.4
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark_part = cv2.inRange(gray, 0, 30)
    bright_part = cv2.inRange(gray, 220, 255)
    total_pixel = np.size(gray)
    dark_pixel = np.sum(dark_part > 0)
    bright_pixel = np.sum(bright_part > 0)

    if dark_pixel / total_pixel > bright_thres:
        return ModelOutputs.DARK_LIGHTING
    if bright_pixel / total_pixel > dark_thres:
        return ModelOutputs.BRIGHT_LIGHTING
    return True


def check_people_absence_on_image(image_to_check, algorithm):
    """
    Checking for the absence of people in the photo.

    :param image_to_check: image to check.
    :param algorithm: algorithm to apply.
    :return: return True if there are people in the photo, return False otherwise.
    """

    # If algorithm is MTCNN.
    if type(algorithm).__name__ == "MTCNN":
        face_detections = algorithm.detect_faces(image_to_check)
        # Return True if no people were found in the photo, return false otherwise.
        return len(face_detections) == 0

    # if algorithm is yolov5.
    if str(type(algorithm)) == "<class 'yolov5.models.common.AutoShape'>":
        # Model prediction.
        predictions = algorithm(image_to_check).pred[0]
        # Classes that were found on image (0 depends to human).
        categories = predictions[:, 5]

        # Return True if no people were found in the photo, return False otherwise.
        return categories.tolist().count(0) == 0


def get_number_of_people_on_image(image_to_check, algorithm):
    """
    Get number of people in the photo.

    :param image_to_check: image to check.
    :param algorithm: algorithm to apply.
    :return: return number of people on image.
    """
    if str(type(algorithm)) == "<class 'yolov5.models.common.AutoShape'>":
        # Предсказание модели.
        predictions = algorithm(image_to_check).pred[0]
        # Классы, найденные на изображении.
        categories = predictions[:, 5]
        return torch.sum(categories == 0)


def crop_face_from_image(image_to_crop, algorithm):
    """
    Cropping face from image.

    :param image_to_crop: image to crop.
    :param algorithm: cropping algorithm.
    :return: cropped face.
    """
    if type(algorithm).__name__ == "MTCNN":
        image_to_crop = cv2.cvtColor(image_to_crop, cv2.COLOR_BGR2RGB)
        return algorithm(image_to_crop)


def get_face_parts_boxes(face, algorithm):
    """
    Get boxes of face parts: eyes, mouth, nose.

    :param face: image from which parts of the face are cut out.
    :param algorithm: algorithm to use.
    :return: box of left eye, right eye, mouth, nose.
    """
    if type(algorithm).__name__ == "MTCNN":
        face = Image.fromarray(face.permute(1, 2, 0).int().numpy().astype('uint8'), 'RGB')
        face = cv2.cvtColor(np.float32(face), cv2.COLOR_BGR2RGB)

        detections = algorithm.detect_faces(face)
        if not detections:
            return []
        l_eye = detections[0]['keypoints']['left_eye']
        r_eye = detections[0]['keypoints']['right_eye']
        mouth_t = detections[0]['keypoints']['mouth_left']
        mouth_b = detections[0]['keypoints']['mouth_right']
        nose = detections[0]['keypoints']['nose']

        return [l_eye, r_eye, mouth_t, mouth_b, nose]


def save_image(img, path):
    im = Image.fromarray(img.permute(1, 2, 0).int().numpy().astype('uint8'), 'RGB')
    im.save(path)


def change_coordinate(coord, x, y, left_bound, right_bound):
    """
    Change coordinate "coord" adding to it x and y. Check if left and right bounds fit.

    :param coord:coordinate to change.
    :param x:add to coord[0].
    :param y:add to coord[1].
    :param left_bound:check if coord[0] < left_bound and coord[1] < left_bound.
    :param right_bound:check if coord[0] > right_bound and coord[1] > right_bound.
    :return:changed coordinate.
    """
    coord = list(coord)
    coord[0] += x
    if coord[0] < left_bound:
        coord[0] = left_bound + 1
    if coord[0] > right_bound:
        coord[0] = right_bound - 1
    coord[1] += y
    if coord[1] < left_bound:
        coord[1] = left_bound + 1
    if coord[1] > right_bound:
        coord[1] = right_bound - 1
    return tuple(coord)


def load_face_parts_classifier(filepath):
    """
    Initialize the classifier which will classify different parts of the face (eyes, nose, mouth).

    :param filepath:path to classifier to be loaded.
    :return:initialized classifier.
    """

    if get_file_extension(filepath) == ".pt":
        model = ModelBaseline()
        model.load_state_dict(torch.load(filepath))
        model.eval()
        return model


def save_message(filepath, message):
    """
    Save provided message on path.

    :param filepath:the path where the message should be saved
    :param message:message to be saved.
    :return:
    """
    with open(filepath, 'w') as f:
        f.write(message)


def delete_files_from_folder(folder):
    """
    Delete all files from granted folder.

    :param folder:folder to delete files from.
    """
    files = glob.glob(folder)
    for f in files:
        os.remove(f)


def chose_file():
    """
    Get path of file, that was chosen.

    :return: file path.
    """

    root = tk.Tk()
    root.withdraw()

    return filedialog.askopenfilename()


def get_face_part_prediction(face_part_img, algorithm):
    """
    Get prediction for granted face part.

    :param face_part_img:get prediction of this img.
    :param algorithm:algorithm that is used to get prediction.
    :return: tuple(float prediction of face part, prediction label of face part)/
    """
    face_part_img = Image.fromarray(face_part_img.int().numpy().astype('uint8'), 'RGB')
    face_part_img = face_part_img.resize((100, 100), Image.LANCZOS)
    face_part_img = transforms.ToTensor()(face_part_img)
    face_part_prediction = algorithm(face_part_img.unsqueeze(0).float())
    face_part_prediction_label = torch.argmax(face_part_prediction)
    return face_part_prediction, face_part_prediction_label
