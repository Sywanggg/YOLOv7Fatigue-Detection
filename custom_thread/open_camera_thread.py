import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from numpy import ndarray


class OpenCameraThread(QThread):
    frame_processed = pyqtSignal(QPixmap, ndarray)

    def __init__(self):
        super(OpenCameraThread, self).__init__()
        self.capture = None
        self.is_running = False

    def run(self):
        # 创建视频捕获对象
        self.capture = cv2.VideoCapture(0)

        if not self.capture.isOpened():
            print("Failed to open camera")
            return

        print("Camera opened successfully")

        # 循环捕获和处理视频帧
        self.is_running = True
        while self.is_running:
            # 捕获视频帧
            ret, frame = self.capture.read()

            if not ret:
                print("Failed to capture frame")
                break

            # 将视频帧转换为PyQt5图像
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(image)

            # 发送处理完成的视频帧到主线程
            self.frame_processed.emit(pixmap, frame)

        # 释放视频捕获对象
        self.capture.release()

    def stop(self):
        # 停止线程
        self.is_running = False
        self.wait()
