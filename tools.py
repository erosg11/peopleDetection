import cv2
from tqdm import tqdm


__all__ = ['iter_cel_phone_frames']


url = 'http://192.168.1.87:8080/video'
cap = cv2.VideoCapture(url)
tq = tqdm()


def iter_cel_phone_frames():
    while True:
        tq.update(1)
        ret, frame = cap.read()
        if frame is not None:
            frame = cv2.flip(frame, 1)
            cv2.imshow('frame', frame)
            frame.flags.writeable = False
            yield frame
        q = cv2.waitKey(1)
        if q == ord("q"):
            break
    cv2.destroyAllWindows()
