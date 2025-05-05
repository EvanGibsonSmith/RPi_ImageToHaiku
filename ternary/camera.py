import cv2
import numpy as np

class Camera:
    def __init__(self, camera_index: int = 0):
        self.cam_idx = camera_index
        self.cam = cv2.VideoCapture(self.cam_idx)

        self.frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert self.frame_width != 0
        assert self.frame_height != 0

    def __call__(self):
        ret, frame = self.cam.read()

        # bin frame down to 32x32x3
        if ret:
            frame = cv2.resize(frame, (32, 32))

 #       cv2.imshow("Camera", frame)
#        cv2.waitKey(0)
        return np.array([frame], dtype=np.float32)

    def __del__(self):
        self.cam.release()
        cv2.destroyAllWindows()
