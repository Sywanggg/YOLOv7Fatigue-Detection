import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog

from custom_thread.age_detect_thread import AgeDetectThread
from custom_thread.detect_thread import DetectThread
from custom_thread.open_camera_thread import OpenCameraThread
from custom_thread.play_voice_thread import PlayVoiceThread
from view.main_window_ui import Ui_MainWindow


# 主窗口
class Window(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(Window, self).__init__()
        # 设置主界面的UI
        self.age_image = None
        self.param = None
        self.timer = None
        self.capture = None
        self.setupUi(self)
        self.open_camera_thread = OpenCameraThread()
        self.detect_thread = DetectThread()
        self.age_detect_thread = AgeDetectThread()
        self.play_voice_thread = PlayVoiceThread()
        self.source = None

        # 连接信号与槽
        self.signal()

    # 处理信号机制
    def signal(self):
        # 退出系统按钮事件
        self.btn_exit.clicked.connect(self.close)
        # 上传视频按钮事件
        self.btn_upload.clicked.connect(self.upload)
        # 打开摄像头按钮事件
        self.btn_open_camera.clicked.connect(self.open_camera)
        # 开始推断事件
        self.btn_start.clicked.connect(self.detect)

    def detect(self):
        self.textEdit_log.setText("")
        self.param = [self.source,  # 数据源
                      True if self.checkBox_voice_broadcast.isChecked() else False,  # 是否语音播报
                      self.doubleSpinBox_eye_threshold.value(),  # 眼部开合阈值
                      self.doubleSpinBox_mouth_threshold.value(),  # 嘴部开合阈值
                      self.doubleSpinBox_perclos_threshold.value(),  # PERCLOS阈值
                      self.spinBox_frames_delay.value()]  # 延迟显示帧数
        self.detect_thread.get_param(self.param)
        self.detect_thread.frame_processed.connect(self.show_frame)
        self.detect_thread.frame_signal.connect(lambda frame: self.label_frame.setText(str(frame)))
        self.detect_thread.face_signal.connect(
            lambda face_confidence: self.label_face.setText(str(face_confidence)[:4]))
        self.detect_thread.log_signal.connect(self.show_log)
        self.detect_thread.eye_mouth_signal.connect(self.show_eye_mouth)
        self.detect_thread.fatigue_signal.connect(self.play_voice)
        self.detect_thread.start()
        self.age_detect_thread.get_image(self.age_image)
        self.age_detect_thread.age_sex_processed.connect(self.show_age_sex)
        self.age_detect_thread.start()

    # 语音播报
    def play_voice(self):
        self.play_voice_thread.start()

    def show_log(self, log):
        self.textEdit_log.insertPlainText(log + '\n')
        self.textEdit_log.verticalScrollBar().setValue(self.textEdit_log.verticalScrollBar().value() + 1000)

    def show_eye_mouth(self, eye_mouth_list):
        self.label_left_eye_status.setText(str(eye_mouth_list[0]))
        self.label_left_eye_confidence.setText(str(eye_mouth_list[1])[:4])
        self.label_ritht_eye_status.setText(str(eye_mouth_list[2]))
        self.label_right_eye_confidence.setText(str(eye_mouth_list[3])[:4])
        self.label_mouth_status.setText(str(eye_mouth_list[4]))
        self.label_mouth_confidence.setText(str(eye_mouth_list[5])[:4])

    def show_age_sex(self, msg_list):
        self.label_sex.setText(str(msg_list[0]))
        self.label_age.setText(str(msg_list[1]))
        self.label_sex_confidence.setText(str(msg_list[2])[:4])
        self.label_age_confidence.setText(str(msg_list[3])[:4])

    # 打开摄像头事件
    def open_camera(self):
        self.source = '0'
        self.detect_thread.stop()
        self.open_camera_thread.frame_processed.connect(self.show_frame)
        self.open_camera_thread.start()

    def show_frame(self, pixmap, frame):
        # 在标签中显示视频帧
        self.label_photo.setPixmap(pixmap)
        self.age_image = frame

    # 上传本地视频事件
    def upload(self):
        # 在打开本地视频前关闭摄像头
        self.open_camera_thread.stop()
        self.detect_thread.stop()
        # 读取本地视频，返回的第一个参数是文件路径
        video_file, _ = QFileDialog.getOpenFileName(self, '打开文件', './video', 'MP4 Files (*.mp4)')

        if video_file:
            self.line_video_path.setText(video_file)
            self.source = video_file
            self.capture = cv2.VideoCapture(video_file)
            if self.capture:
                ret, frame = self.capture.read()
                self.age_image = frame

                # 将视频帧转换为PyQt5图像
                image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(image)
                self.label_photo.setPixmap(pixmap)
