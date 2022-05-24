# -*- coding: utf-8 -*-

import os
import sys

import argparse
import numpy as np

from PIL import Image, ImageFont, ImageDraw
import json
import collections
import re
from fontTools.ttLib import TTFont
from tqdm import tqdm
import random

from torch import nn
from torchvision import transforms

from utils.charset_util import processGlyphNames

CN_CHARSET = None
CN_T_CHARSET = None
JP_CHARSET = None
KR_CHARSET = None

DEFAULT_CHARSET = "./charset/cjk.json"


def load_global_charset():
    global CN_CHARSET, JP_CHARSET, KR_CHARSET, CN_T_CHARSET
    cjk = json.load(open(DEFAULT_CHARSET))
    CN_CHARSET = cjk["gbk"]
    JP_CHARSET = cjk["jp"]
    KR_CHARSET = cjk["kr"]
    CN_T_CHARSET = cjk["gb2312_t"]


def draw_single_char(ch, font, canvas_size, x_offset=0, y_offset=0):
    img = Image.new("L", (canvas_size * 2, canvas_size * 2), 0)
    draw = ImageDraw.Draw(img)
    try:
        draw.text((10, 10), ch, 255, font=font)
    except OSError:
        return None
    bbox = img.getbbox()
    if bbox is None:
        return None
    l, u, r, d = bbox
    l = max(0, l - 5)
    u = max(0, u - 5)
    r = min(canvas_size * 2 - 1, r + 5)
    d = min(canvas_size * 2 - 1, d + 5)
    if l >= r or u >= d:
        return None
    img = np.array(img)
    img = img[u:d, l:r]
    img = 255 - img
    img = Image.fromarray(img)
    # img.show()
    width, height = img.size
    # Convert PIL.Image to FloatTensor, scale from 0 to 1, 0 = black, 1 = white
    try:
        img = transforms.ToTensor()(img)
    except SystemError:
        return None
    img = img.unsqueeze(0)  # 加轴
    pad_len = int(abs(width - height) / 2)  # 预填充区域的大小
    # 需要填充区域，如果宽大于高则上下填充，否则左右填充
    if width > height:
        fill_area = (0, 0, pad_len, pad_len)
    else:
        fill_area = (pad_len, pad_len, 0, 0)
    # 填充像素常值
    fill_value = 1
    img = nn.ConstantPad2d(fill_area, fill_value)(img)
    # img = nn.ZeroPad2d(m)(img) #直接填0
    img = img.squeeze(0)  # 去轴
    img = transforms.ToPILImage()(img)
    img = img.resize((canvas_size, canvas_size), Image.ANTIALIAS)
    return img


def draw_font2imgs_example(ch, src_font, dst_img, canvas_size, x_offset, y_offset):
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    dst_img = dst_img.resize((canvas_size, canvas_size), Image.ANTIALIAS).convert('RGB')
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255))
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    # convert to gray img
    example_img = example_img.convert('L')
    return example_img

def filter_recurring_hash(charset, font, canvas_size, x_offset, y_offset):
    """ Some characters are missing in a given font, filter them
    by checking the recurring hashes
    """
    _charset = charset.copy()
    np.random.shuffle(_charset)
    sample = _charset[:2000]
    hash_count = collections.defaultdict(int)
    for c in sample:
        img = draw_single_char(c, font, canvas_size, x_offset, y_offset)
        hash_count[hash(img.tobytes())] += 1
    recurring_hashes = filter(lambda d: d[1] > 2, hash_count.items())
    return [rh[0] for rh in recurring_hashes]

def hangeol_font2imgs(src, dst_img_dir, char_size, charset, label, canvas_size,
                      x_offset, y_offset, sample_count, sample_dir):
    src_font = ImageFont.truetype(src, size=char_size)
    count = 0
    for c in tqdm(charset):
        if count == sample_count:
            break
        img_path = os.path.join(dst_img_dir, c)
        dst_img = Image.open(img_path + '.png')
        e = draw_font2imgs_example(c, src_font, dst_img, canvas_size, x_offset, y_offset)
        if e:
            e.save(os.path.join(sample_dir, "%d_%04d.jpg" % (label, count)))
            count += 1
            if count % 500 == 0:
                print("processed %d chars" % count)


load_global_charset()
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['hangeol_font2imgs'], required=True)
parser.add_argument('--src_font', type=str, default=None, help='path of the source font')
parser.add_argument('--src_fonts_dir', type=str, default=None, help='path of the source fonts')
parser.add_argument('--src_imgs', type=str, default=None, help='path of the source imgs')
parser.add_argument('--dst_font', type=str, default=None, help='path of the target font')
parser.add_argument('--dst_imgs', type=str, default=None, help='path of the target imgs')

parser.add_argument('--filter', default=False, action='store_true', help='filter recurring characters')
parser.add_argument('--charset', type=str, default='CN',
                    help='charset, can be either: CN, JP, KR or a one line file. ONLY VALID IN font2font mode.')
parser.add_argument('--shuffle', default=False, action='store_true', help='shuffle a charset before processings')
parser.add_argument('--char_size', type=int, default=256, help='character size')
parser.add_argument('--canvas_size', type=int, default=256, help='canvas size')
parser.add_argument('--x_offset', type=int, default=0, help='x offset')
parser.add_argument('--y_offset', type=int, default=0, help='y_offset')
parser.add_argument('--sample_count', type=int, default=5000, help='number of characters to draw')
parser.add_argument('--sample_dir', type=str, default='sample_dir', help='directory to save examples')
parser.add_argument('--label', type=int, default=0, help='label as the prefix of examples')

args = parser.parse_args()

if __name__ == "__main__":
    if not os.path.isdir(args.sample_dir):
        os.mkdir(args.sample_dir)
    if args.mode == 'hangeol_font2imgs':
        if args.src_font is None or args.dst_imgs is None:
            raise ValueError('src_font and dst_imgs are required.')
        charset = list(open(args.charset, encoding='utf-8').readline().strip())
        hangeol_font2imgs(args.src_font, args.dst_imgs, args.char_size, charset, args.label,
                          args.canvas_size, args.x_offset, args.y_offset,
                          args.sample_count, args.sample_dir)
    else:
        raise ValueError('mode should be hangeol_font2imgs')
