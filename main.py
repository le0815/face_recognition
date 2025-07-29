import threading

import cv2
from deepface import DeepFace
from tensorflow.python.ops.gen_functional_ops import While

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
input_img = cv2.imread("assets/elon_musk.jpg")
face_match = False


def check_face(frame):
    global face_match
    try:
        if DeepFace.verify(frame, input_img.copy(), enforce_detection=False)["verified"]:
            face_match = True
        else:
            face_match = False
    except Exception as err:
        raise err


while 1:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
                # break
            except Exception as err:
                print(Exception)

        counter += 1

        if face_match:
            cv2.putText(frame, "Match", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cap.release()
        break

# while 1:
#     print(end="", )

cv2.destroyAllWindows()
