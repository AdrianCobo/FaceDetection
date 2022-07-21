# minimal code for running our program
# api: https://google.github.io/mediapipe/solutions/face_detection.html

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("videos/1.mp4")
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.9)

while cap.isOpened():
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection) # draw the bounding boxes and relevant points
            # print(id, detection)
            # print(detection.score) # face probability
            # print(detection.location_data.relative_bounding_box) # information about the bounding box (datos normalizados)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape  # ih = image high,...
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)

            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)

    # if frame is read correctly ret is True
    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(10)
    # exit pressing q
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
