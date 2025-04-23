import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def cv2_chinese_text(img, text, position, textColor=(0, 0, 255), textSize=80):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)

    # fontStyle = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
    fontStyle = ImageFont.truetype("Monaco.ttf", textSize, encoding="utf-8")

    draw.text(position, text, textColor, font=fontStyle)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
