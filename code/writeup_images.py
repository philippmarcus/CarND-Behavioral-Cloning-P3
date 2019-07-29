import preprocessing
import csv
import cv2

class ImageLoader(object):
    def __init__(self):
        laps=["lap1", "lap2", "lap3"]
        self.samples = []
        for lap in laps:
                with open("data/{}/driving_log.csv".format(lap)) as csvfile:
                        reader = csv.reader(csvfile)
                        for line in reader:
                                line.append(lap)
                                self.samples.append(line)

    def load_image(self, sample_id, camera_angle="CENTER"):
        
        camera_angles = {"CENTER":0, "LEFT":1, "RIGHT":2}
        assert camera_angle in camera_angles.keys()
        measurement_correction = {0:0., 1:0.2, 2:-0.2}
        
        SAMPLE_ID = 300
        i = camera_angles[camera_angle]
        sample = self.samples[SAMPLE_ID]
        source_path = sample[i]
        lap = sample[-1]
        filename = source_path.split("/")[-1]
        current_path = 'data/{}/IMG/'.format(lap) + filename
        img = cv2.imread(current_path)
        assert img is not None
        measurement = float(sample[3]) + measurement_correction[i]
        return img, measurement

img_loader = ImageLoader()

# generated a original and cropped image version
img_original, m = img_loader.load_image(300)
X, y = preprocessing.get_cropped_copies([img_original], [m])
img_cropped, _ = img_loader.load_image(300)
cv2.imwrite("track_1_original.png", img_original)
cv2.imwrite("track_1_cropped.png", X[0])

# generate an original and flipped image version
img_original = X[0]
X, y = preprocessing.get_flipped_copies([img_original], [m])
img_cropped, _ = img_loader.load_image(300)
cv2.imwrite("track_1_flipped.png", X[0])

# generate a color inverted version of the image
img_original = X[0]
X, y = preprocessing.get_color_inverted_copies([img_original], [m])
img_cropped, _ = img_loader.load_image(300)
cv2.imwrite("track_1_flipped_inverted.png", X[0])

img_original, m = img_loader.load_image(300)
X, y = preprocessing.get_color_inverted_copies([img_original], [m])
X, y = preprocessing.get_cropped_copies(X, [m])
cv2.imwrite("track_1_original_inverted.png", X[0])

# camera angles
img, m = img_loader.load_image(300, camera_angle="CENTER")
X, y = preprocessing.get_cropped_copies([img], [m])
cv2.imwrite("track_1_center.png", X[0])

img, m = img_loader.load_image(300, camera_angle="LEFT")
X, y = preprocessing.get_cropped_copies([img], [m])
cv2.imwrite("track_1_left.png", X[0])

img, m = img_loader.load_image(300, camera_angle="RIGHT")
X, y = preprocessing.get_cropped_copies([img], [m])
cv2.imwrite("track_1_right.png", X[0])
