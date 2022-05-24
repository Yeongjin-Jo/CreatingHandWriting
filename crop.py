# -*- coding: utf-8 -*-
import os
import argparse
from PIL import Image
from PIL import ImageOps
from PIL import ImageFilter
from PIL import ImageEnhance
from cv2 import bilateralFilter
import numpy as np
from tqdm import tqdm


def crop_image_uniform(src_dir, dst_dir):
    f = open("399ForCrop.txt", "r", encoding='UTF-8')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for page in range(1,4):
        img = Image.open(src_dir + "/" + str(page) +"-uniform.png").convert('L')

        # 여백삭제
        for i in range(img.size[0]):
            if np.array(img).sum(axis=0)[i] < img.size[1] * 255 * 0.7:
                whitespace_left = i
                break
        for i in range(img.size[0] - 1, 0, -1):
            if np.array(img).sum(axis=0)[i] < img.size[1] * 255 * 0.7:
                whitespace_right = i
                break
        for i in range(img.size[1]):
            if np.array(img).sum(axis=1)[i] < img.size[0] * 255 * 0.7:
                whitespace_upper = i
                break
        for i in range(img.size[1] - 1, 0, -1):
            if np.array(img).sum(axis=1)[i] < img.size[0] * 255 * 0.7:
                whitespace_lower = i
                break

        img = img.crop((whitespace_left, whitespace_upper, whitespace_right, whitespace_lower))

        header_ratio = 16.5 / (16.5 + 42)

        line_ratio = 4 / img.size[0]

        width, height = img.size
        cell_width = width / float(cols)
        cell_height = height / float(rows)
        header_offset = height / float(rows) * header_ratio
        width_margin = line_ratio * img.size[0]
        height_margin = line_ratio * img.size[1]
        cnt = 0
        for j in range(0, rows):
            for i in range(0, cols):
                left = i * cell_width
                upper = j * cell_height + header_offset
                right = left + cell_width
                lower = (j + 1) * cell_height

                center_x = (left + right) / 2
                center_y = (upper + lower) / 2

                crop_width = right - left - 2 * width_margin
                crop_height = lower - upper - 2 * height_margin

                size = 0
                if crop_width > crop_height:
                    size = crop_height / 2
                else:
                    size = crop_width / 2

                left = center_x - size;
                right = center_x + size;
                upper = center_y - size;
                lower = center_y + size;

                code = f.readline()
                if not code:
                    break
                else:
                    name = dst_dir + '/' + code.strip() + ".png"
                    cropped_image = img.crop((left, upper, right, lower))
                    cropped_image = cropped_image.resize((128,128), Image.LANCZOS)
                    # 가장자리만 짜르기
                    edge = 0
                    for i in range(round(np.array(cropped_image).shape[0] / 2)):
                        if np.array(cropped_image).sum(axis=0)[i] == 128 * 255 and \
                                np.array(cropped_image).sum(axis=1)[i] == 128 * 255 and \
                                np.array(cropped_image).sum(axis=0)[-(i + 1)] == 128 * 255 and \
                                np.array(cropped_image).sum(axis=1)[-(i + 1)] == 128 * 255:
                            if i >= edge:
                                edge = i
                    cropped_image = cropped_image.crop(
                        (edge, edge, cropped_image.size[1] - edge, cropped_image.size[1] - edge))
                    cropped_image = cropped_image.resize((128, 128), Image.LANCZOS)
                    # Increase constrast
                    enhancer = ImageEnhance.Contrast(cropped_image)
                    cropped_image = enhancer.enhance(1.5)
                    opencv_image = np.array(cropped_image)
                    opencv_image = bilateralFilter(opencv_image, 9, 30, 30)
                    cropped_image = Image.fromarray(opencv_image)
                    cropped_image.save(name)
        print("Processed uniform page " + str(page))

    # path = dst_dir
    # file_list = os.listdir(path)
    # for file in tqdm(file_list):
    #     dst_img = Image.open(path + '/' + file)
    #     # 여백삭제
    #     for i in range(dst_img.size[0]):
    #         if np.array(dst_img).sum(axis=0)[i] != dst_img.size[1] * 255:
    #             whitespace_left = i
    #             break
    #     for i in range(dst_img.size[0] - 1, 0, -1):
    #         if np.array(dst_img).sum(axis=0)[i] != dst_img.size[1] * 255:
    #             whitespace_right = i
    #             break
    #     for i in range(dst_img.size[1]):
    #         if np.array(dst_img).sum(axis=1)[i] != dst_img.size[0] * 255:
    #             whitespace_upper = i
    #             break
    #     for i in range(dst_img.size[1] - 1, 0, -1):
    #         if np.array(dst_img).sum(axis=1)[i] != dst_img.size[0] * 255:
    #             whitespace_lower = i
    #             break
    #     dst_img = dst_img.crop((whitespace_left, whitespace_upper, whitespace_right, whitespace_lower))
    #     dst_img.save(path + '/' + file)
    # print('CropIsFinished')



parser = argparse.ArgumentParser(description='Crop scanned images to character images')
parser.add_argument('--src_dir', dest='src_dir', required=True, help='directory to read scanned images')
parser.add_argument('--dst_dir', dest='dst_dir', required=True, help='directory to save character images')

args = parser.parse_args()

if __name__ == "__main__":
    rows = 12
    cols = 12
    header_ratio = 16.5/(16.5+42)
    crop_image_uniform(args.src_dir, args.dst_dir)
#    crop_image_frequency(args.src_dir, args.dst_dir)
