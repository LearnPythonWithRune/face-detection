# importing opencv
import cv2

# using cv2.CascadeClassifier
# See https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
# See more Cascade Classifiers https://github.com/opencv/opencv/tree/4.x/data/haarcascades
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("sample_images/sample-00.jpg")

# changing the image to gray scale for better face detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

multi_scale = detector.detectMultiScale(
    gray,
    scaleFactor=2,  # Big reduction
    minNeighbors=5  # 4-6 range
)

# drawing a rectangle to the image.
# for loop is used to access all the coordinates of the rectangle.
for x, y, w, h in multi_scale:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)

# showing the detected face followed by the waitKey method.
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
