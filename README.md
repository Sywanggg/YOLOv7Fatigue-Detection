百度网盘YOLOv7Fatigue Detection.zip
链接: https://pan.baidu.com/s/1v_nR4baMXyjnEpFMxY8nbw?pwd=5qfw 提取码: 5qfw

环境配置：
本系统编写了requirement.txt，文件中包含了系统所需的python第三方库。Windows系统在有anaconda的情况下双击start.bat会自动安装所需环境，也可在PyCharm中自动安装。（Mac系统由于第三方库权限问题无法发出语音警报）。

系统主界面：
本系统在PyCharm中运行。打开系统，进入系统主界面，如下图所示：
![Image text](https://github.com/Sywanggg/YOLOv7Fatigue-Detection/blob/main/IMG/%E5%9B%BE%E7%89%871.png)

系统功能实现：
本系统的数据传入提供了本地视频上传和摄像头实时采集两种数据传入按钮。
点击“上传视频”按钮并选取指定视频进行测试，界面中会显示本地文件路径，同时提供实时显示视频状态的窗口，如图所示：
![Image text](https://github.com/Sywanggg/YOLOv7Fatigue-Detection/blob/main/IMG/%E5%9B%BE%E7%89%872.png)

参数设置部分可以设置指定的阈值和延迟显示帧数，延迟显示帧数为指定多少帧后重新测定疲劳状态，避免持续报警。
其实时状态同步在界面的“实时状态”框中。
检测目标达到疲劳状态时会触发疲劳警报`Fatigue!`，同时会发出语音警报。

