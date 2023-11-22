import cv2
import imutils
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
scales = [1.0, 1.05, 1.1]
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = imutils.resize(frame, width=min(400, frame.shape[1]))
    regions = []
    for scale in scales:
        (regions_at_scale, _) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(4, 4), scale=scale)
        regions.extend(regions_at_scale)
    for (x, y, w, h) in regions:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow("Webcam Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
