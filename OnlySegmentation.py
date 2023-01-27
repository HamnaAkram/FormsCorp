import os
import cv2
import numpy as np
import dlib

TooManyFace='More than 1 face in image'
NoFace='No face detected in image'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/hamna/DFRC/pythonProject/shape_predictor_68_face_landmarks.dat")

FEATHER_AMOUNT = 11
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
FULL_OUTER_CONTOUR_POINTS = [list(range(27)), ]
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    if isinstance(landmarks, list):
        for item in landmarks:
            for group in FULL_OUTER_CONTOUR_POINTS:
                    draw_convex_hull(im,
                                 item[group],
                                 color=1)
                    cv2.imshow('draw', im)
                    cv2.waitKey(0)
    else:
        for group in FULL_OUTER_CONTOUR_POINTS:
            draw_convex_hull(im,
                             landmarks[group],
                             color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)
def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFace
    if len(rects) == 0:
        raise NoFace

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
def read_im_and_landmarks(fname):
    # if not os.path.exists(fname):
    #     raise Exception('Cannot find image file: {}'.format(fname))

    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (256,256))
    s = get_landmarks(im)

    return im, s


def main(path):
    for images in os.listdir(path):
        img = os.path.join(path,images)
        im, landmarks = read_im_and_landmarks(img)
        mask = get_face_mask(im,landmarks)
        output_im = mask * im
        # put original image and output image horizontally
        output_im = np.hstack([im, output_im])

        output_im=annotate_landmarks(im,landmarks)
        save_name = 'annotated_'+images
        # cv2.imshow('output',output_im)
        # cv2.waitKey(0)
        cv2.imwrite(save_name,output_im)


if __name__ == "__main__":
    main('/home/hamna/DFRC/pythonProject/Task2/Real')