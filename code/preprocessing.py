import cv2
import numpy as np
import copy

def get_flipped_copies(_X, _y):
        # Flip the image horizontally
        _X_flipped = list(map(lambda img: cv2.flip(img, 1), _X))
        _y_flipped = list(map(lambda angl: -angl, _y))
        return np.array(_X_flipped), _y_flipped

def get_color_inverted_copies(_X, _y):
        # Invert each channel of RGB image
        _X_inverted = list(map(lambda img: cv2.bitwise_not(img), _X))
        _y_inverted = copy.copy(_y)
        return np.array(_X_inverted), _y_inverted

def get_cropped_copies(_X, _y):
        # Only leave the part with the street
        _X_cropped = list(map(lambda img: img[70:-25,:,:].copy(), _X))
        return _X_cropped, _y


def get_Gaussian_blurred(_imgs, kernel=5):
        # apply guassian blur on src image
        for _img in _imgs:
                print(_img)
        _X_blurred = list(map(lambda img: cv2.GaussianBlur(np.array(img),(kernel,kernel),cv2.BORDER_DEFAULT), _imgs))
        return np.array(_X_blurred)

"""
Performs a histogram equalization after YcbCr conversion for contrast improvement.
ASSUMPTION: THIS HAS A NEGATIVE EFFECT AS THE STREET GETS VERY GRAINY
"""
def get_histogram_equalization(_imgs, source_format="BGR"):
        # Support color conversion for different input formats
        assert source_format in ["RGB", "BGR"]
        color_mappings = {
                "BGR": [cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR],
                "RGB": [cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB]
        }
        result = []
        for _img in _imgs:
                # Apply the color conversion
                # Get Y channel
                img_HSV_copy = cv2.cvtColor(_img, color_mappings[source_format][0])
                H, S, V = cv2.split(img_HSV_copy)

                # Equalize S channel.
                S_equalized = cv2.equalizeHist(S)
                #V_equalized = cv2.equalizeHist(V)

                # Concert back to input format
                img_HSV_equalized = cv2.merge((H, S_equalized, V))
                img_equalized = cv2.cvtColor(img_HSV_equalized, color_mappings[source_format][1])
                result.append(img_equalized)
        return np.array(result)

def preprocessing_pipeline(_X):
        _X = get_Gaussian_blurred(_X)
        _X = get_histogram_equalization(_X)
        return _X