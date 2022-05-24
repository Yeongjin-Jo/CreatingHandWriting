from tqdm import tqdm

from data import DatasetFromObj
from torch.utils.data import DataLoader, TensorDataset
from model import Zi2ZiModel
import os
import argparse
import torch
from torch import nn
import random
import time
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from data import DatasetFromObj
import torchvision.utils as vutils
import numpy as np
from PIL import ImageEnhance
from cv2 import bilateralFilter
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import time
from model.model import chk_mkdir


parser = argparse.ArgumentParser(description='Infer')
parser.add_argument('--experiment_dir', required=True,
                    help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--start_from', type=int, default=0)
parser.add_argument('--gpu_ids', default=[], nargs='+', help="GPUs")
parser.add_argument('--image_size', type=int, default=256,
                    help="size of your input and output image")
parser.add_argument('--L1_penalty', type=int, default=100, help='weight for L1 loss')
parser.add_argument('--Lconst_penalty', type=int, default=15, help='weight for const loss')
# parser.add_argument('--Ltv_penalty', dest='Ltv_penalty', type=float, default=0.0, help='weight for tv loss')
parser.add_argument('--Lcategory_penalty', type=float, default=1.0,
                    help='weight for category loss')
parser.add_argument('--embedding_num', type=int, default=40,
                    help="number for distinct embeddings")
parser.add_argument('--embedding_dim', type=int, default=128, help="dimension for embedding")
parser.add_argument('--batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--resume', type=int, default=None, help='resume from previous training')
parser.add_argument('--obj_path', type=str, default='./experiment/data/val.obj', help='the obj file you infer')
parser.add_argument('--input_nc', type=int, default=1)

parser.add_argument('--from_2350', action='store_true')
parser.add_argument('--src_txt', type=str, default='大威天龍大羅法咒世尊地藏波若諸佛')
parser.add_argument('--canvas_size', type=int, default=256)
parser.add_argument('--char_size', type=int, default=256)
parser.add_argument('--label', type=int, default=0)
parser.add_argument('--src_font', type=str, default='charset/gbk/方正新楷体_GBK(完整).TTF')


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

# def draw_single_char(ch, font, canvas_size):
#     img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
#     draw = ImageDraw.Draw(img)
#     draw.text((0, 0), ch, (0, 0, 0), font=font)
#     img = img.convert('L')
#     return img


def main():
    args = parser.parse_args()
    data_dir = os.path.join(args.experiment_dir, "data")
    checkpoint_dir = os.path.join(args.experiment_dir, "checkpoint")
    sample_dir = os.path.join(args.experiment_dir, "sample")
    infer_dir = os.path.join(args.experiment_dir, "infer")
    chk_mkdir(infer_dir)

    # train_dataset = DatasetFromObj(os.path.join(data_dir, 'train.obj'), augment=True, bold=True, rotate=True, blur=True)
    # val_dataset = DatasetFromObj(os.path.join(data_dir, 'val.obj'))
    # dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    t0 = time.time()

    model = Zi2ZiModel(
        input_nc=args.input_nc,
        embedding_num=args.embedding_num,
        embedding_dim=args.embedding_dim,
        Lconst_penalty=args.Lconst_penalty,
        Lcategory_penalty=args.Lcategory_penalty,
        save_dir=checkpoint_dir,
        gpu_ids=args.gpu_ids,
        is_training=False
    )
    model.setup()
    model.print_networks(True)
    model.load_networks(args.resume)

    t1 = time.time()

    if args.from_2350:
        f = open('상용한글2350자.txt', 'r', encoding='UTF8')
        hangeol_2350 = f.readline()
        hangeol_2350 = hangeol_2350[1:-1]
        src = hangeol_2350

        # src = args.src_txt
        # font = ImageFont.truetype(args.src_font, size=args.char_size)
        font = ImageFont.truetype(args.src_font, size=230)
        img_list = [transforms.Normalize(0.5, 0.5)(
            transforms.ToTensor()(
                draw_single_char(ch, font, args.canvas_size)

            )
        ).unsqueeze(dim=0) for ch in src]
        label_list = [args.label for _ in src]

        img_list = torch.cat(img_list, dim=0)
        label_list = torch.tensor(label_list)

        dataset = TensorDataset(label_list, img_list, img_list)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    else:
        val_dataset = DatasetFromObj(os.path.join(data_dir, 'val.obj'),
                                     input_nc=args.input_nc,
                                     start_from=args.start_from)
        dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    global_steps = 0
    batch_cnt = 0
    for batch in tqdm(dataloader):
        #model.set_input(batch[0], batch[2], batch[1])
        #model.optimize_parameters()
        #img = vutils.make_grid(tensor_to_plot)
        #model.sample(batch, infer_dir)
        #model.infer_sample(batch, infer_dir, batch_cnt, args.label)
        model.infer_sample(batch, infer_dir, batch_cnt)
        global_steps += 1
        batch_cnt += args.batch_size

    t_finish = time.time()

    print('cold start time: %.2f, hot start time %.2f' % (t_finish - t0, t_finish - t1))


if __name__ == '__main__':
    with torch.no_grad():
        main()
