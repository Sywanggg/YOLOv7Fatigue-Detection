import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from numpy import random, ndarray

from models.experimental import attempt_load
from utils import cv2pil
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, set_logging, increment_path
from utils.general import strip_optimizer
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


class DetectThread(QThread):
    frame_processed = pyqtSignal(QPixmap, ndarray)
    frame_signal = pyqtSignal(int)
    face_signal = pyqtSignal(float)
    log_signal = pyqtSignal(str)
    eye_mouth_signal = pyqtSignal(list)
    fatigue_signal = pyqtSignal(bool)

    def __init__(self):
        super(DetectThread, self).__init__()
        self.param = None
        self.is_detect = True

    def detect(self, opt, save_img=False):
        source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
        save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # 初始化
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'

        # 加载YOLOv7模型
        model = attempt_load(weights, map_location=device)
        stride = int(model.stride.max())
        imgsz = check_img_size(imgsz, s=stride)

        if trace:
            model = TracedModel(model, device, opt.img_size)

        if half:
            model.half()  # to FP16

        # 调用分类器
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # 设置数据加载器
        vid_path, vid_writer = None, None
        if webcam:
            cudnn.benchmark = False  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # 获取类名
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # 开始推断
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()

        # 定义眼部和嘴部开合阈值
        eye_threshold = self.param[2]
        mouth_threshold = self.param[3]

        # 定义PERCLOS阈值
        perclos_threshold = self.param[4]

        # 定义计数器和时间间隔
        frame_count = 0

        # 延迟显示帧数
        frames_delay = 0

        # 当前帧数，返回主窗口
        k = 1

        for path, img, im0s, vid_cap in dataset:  # 处理每帧图像
            self.frame_signal.emit(k)
            k += 1
            if self.is_detect:

                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                if device.type != 'cpu' and (
                        old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                    old_img_b = img.shape[0]
                    old_img_h = img.shape[2]
                    old_img_w = img.shape[3]
                    for i in range(3):
                        model(img, augment=opt.augment)[0]

                # 开始推断
                t1 = time_synchronized()
                with torch.no_grad():
                    pred = model(img, augment=opt.augment)[0]
                t2 = time_synchronized()

                # 使用NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                           agnostic=opt.agnostic_nms)
                t3 = time_synchronized()

                # 调用分类器
                if classify:
                    pred = apply_classifier(pred, modelc, img, im0s)

                eye_mouth_list = ['' for i in range(6)]
                # 处理检测到的目标狂
                for i, det in enumerate(pred):  # 处理每帧图像中的每个框
                    configdence = det[:, -2].mean().item()
                    self.face_signal.emit(configdence)
                    eye_weary = False
                    mouth_weary = False
                    if webcam:
                        p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                    else:
                        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                    p = Path(p)
                    save_path = str(save_dir / p.name)
                    txt_path = str(save_dir / 'labels' / p.stem) + (
                        '' if dataset.mode == 'image' else f'_{frame}')
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                    if len(det):  # 如果当前帧检测到了框
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                            # 写入结果，遍历每帧中的每个框
                        for *xyxy, conf, cls in reversed(det):  # 遍历每个框

                            if save_txt:
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    -1).tolist()
                                line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if (names[int(cls)] == 'open_eye' and conf < eye_threshold) or (
                                    names[int(cls)] == 'closed_eye' and conf > eye_threshold):
                                eye_weary = True
                            if (names[int(cls)] == 'open_mouth' and conf > mouth_threshold) or (
                                    names[int(cls)] == 'closed_mouth' and conf < mouth_threshold):
                                mouth_weary = True

                            # 将框绘制到当前帧图像中
                            if save_img or view_img:
                                label = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                            if 'mouth' in names[int(cls)]:
                                eye_mouth_list[4] = names[int(cls)]
                                eye_mouth_list[5] = conf.item()
                            elif eye_mouth_list[0] == '':
                                eye_mouth_list[0] = names[int(cls)]
                                eye_mouth_list[1] = conf.item()
                            else:
                                eye_mouth_list[2] = names[int(cls)]
                                eye_mouth_list[3] = conf.item()

                    # 打印推断时间 (inference + NMS)
                    print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                    self.log_signal.emit(
                        f'【frame: {frame}】 {s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                    if eye_weary and mouth_weary:
                        frame_count += 1
                    print(frame_count)
                    perclos = frame_count / 50

                    if perclos > perclos_threshold:
                        frames_delay = self.param[5]
                        frame_count = 0
                        print('疲劳状态! ')
                        if self.param[1]:
                            self.fatigue_signal.emit(True)

                    if frames_delay > 0:
                        frames_delay -= 1
                        im0 = cv2pil.cv2_chinese_text(im0, 'fatigue!', (im0.shape[1] - 400, 50), (0, 0, 255), 80)

                    if view_img:
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 millisecond

                    if save_img:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
                            print(f" The image with the result is saved in: {save_path}")
                        else:
                            if vid_path != save_path:
                                vid_path = save_path
                                if isinstance(vid_writer, cv2.VideoWriter):
                                    vid_writer.release()
                                if vid_cap:
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                    save_path += '.mp4'
                                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            vid_writer.write(im0)

                    # 将视频帧转换为PyQt5图像
                    image = QImage(im0, im0.shape[1], im0.shape[0], QImage.Format_RGB888).rgbSwapped()
                    pixmap = QPixmap.fromImage(image)

                    # 发送处理完成的视频帧到主线程
                    self.frame_processed.emit(pixmap, img.cpu().numpy())

                self.eye_mouth_signal.emit(eye_mouth_list)
            else:
                break

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''

        print(f'Done. ({time.time() - t0:.3f}s)')

    def get_param(self, param):
        self.param = param

    def run(self):
        self.is_detect = True
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='./checkpoints/best.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default="./video/test2.mp4",
                            help='source')
        parser.add_argument('--img-size', type=int, default=480, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
        opt = parser.parse_args()
        opt.source = self.param[0]
        print(opt)

        with torch.no_grad():
            if opt.update:
                for opt.weights in ['yolov7.pt']:
                    self.detect(opt)
                    strip_optimizer(opt.weights)
            else:
                self.detect(opt)

    def stop(self):
        # 停止线程
        self.is_detect = False
