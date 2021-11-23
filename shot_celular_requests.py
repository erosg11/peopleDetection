import cv2
import requests
import numpy as np
from tqdm import tqdm


url = 'http://192.168.1.87:8080/shot.jpg'
tq = tqdm()

while True:
    tq.update(1)
    with requests.get(url) as r:
        raw = np.frombuffer(r.content, dtype='uint8')
        img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        cv2.imshow('image', img)
        q = cv2.waitKey(1)
        if q == ord("q"):
            break
cv2.destroyAllWindows()