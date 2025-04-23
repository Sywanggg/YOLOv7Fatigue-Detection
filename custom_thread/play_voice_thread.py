import pyttsx3
from PyQt5.QtCore import QThread



class PlayVoiceThread(QThread):

    def __init__(self):
        super(PlayVoiceThread, self).__init__()

        # 初始化pyttsx3引擎
        self.engine = pyttsx3.init('dummy')

    def run(self):
        # 获取要播放的文字
        text = 'fatigue'

        # 使用pyttsx3引擎播放文字
        self.engine.say(text)
        self.engine.runAndWait()

