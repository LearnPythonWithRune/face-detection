import os
import cv2


class FaceDetector:
    def __init__(self, scale_factor=2, min_neighbors=5):
        self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.img = None

    def read_image(self, filename):
        self.img = cv2.imread(filename)

    def detect_faces(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors
        )

        # drawing a rectangle to the image.
        # for loop is used to access all the coordinates of the rectangle.
        for x, y, w, h in faces:
            cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 0), 5)

        return self.img


face_detector = FaceDetector()

for filename in os.listdir('sample_images/'):
    print(filename)
    face_detector.read_image(f'sample_images/{filename}')
    img = face_detector.detect_faces()

    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
