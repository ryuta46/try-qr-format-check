import sys
from typing import List

import cv2
import numpy as np


class QRCodeExtractor:
    format_mask = [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]

    def __init__(self, img, rect, size):
        self.img = img
        self.tl_x = rect[0][1]
        self.tl_y = rect[0][0]
        self.br_x = rect[2][1]
        self.br_y = rect[2][0]
        self.size = size
        self.scale = (self.br_x - self.tl_x + 1) // size
        print(self.scale)

        self.error_collect_level = 0
        self.mask_pattern = 0

    def extract(self, x, y) -> int:
        bgr = (self.img[self.tl_y + y * self.scale][self.tl_x + x * self.scale])
        bgr = np.array(bgr, dtype=np.int32)
        return 0 if (bgr[0] + bgr[1] + bgr[2]) > (127 * 3) else 1

    def extract_with_mask(self, x, y) -> int:
        raw = self.extract(x, y)
        if self.is_masked(x, y):
            return 0 if raw > 0 else 1
        else:
            return raw

    def assert_white(self, x, y) -> bool:
        if self.extract(x, y) == 1:
            raise Exception(f'({x}, {y}) is not white')
        return True

    def assert_black(self, x, y) -> bool:
        if self.extract(x, y) == 0:
            raise Exception(f'({x}, {y}) is not black')
        return True

    def validate_finder_pattern(self) -> bool:
        for x in range(7):
            # 左上、上
            self.assert_black(x, 0)
            # 左上、下
            self.assert_black(x, 6)
            # 右上、上
            self.assert_black(self.size - x - 1, 0)
            # 右上、下
            self.assert_black(self.size - x - 1, 6)
            # 左下、上
            self.assert_black(x, self.size - 1 - 6)
            # 左下、下
            self.assert_black(x, self.size - 1)

        for y in range(7):
            # 左上、左
            self.assert_black(0, y)
            # 左上、右
            self.assert_black(6, y)
            # 左上、左
            self.assert_black(self.size - 1 - 6, y)
            # 左上、右
            self.assert_black(self.size - 1, y)
            # 左下、左
            self.assert_black(0, self.size - y - 1)
            # 左下、右
            self.assert_black(6, self.size - y - 1)

        return True

    def validate_quiet_zone(self) -> bool:
        for x in range(8):
            self.assert_white(x, 7)
            self.assert_white(x, self.size - 1 - 7)
            self.assert_white(self.size - x - 1, 7)

        for y in range(8):
            self.assert_white(7, y)
            self.assert_white(7, self.size - 1 - y)
            self.assert_white(self.size - 1 - 7, y)

        return True

    def is_finder_pattern(self, x, y) -> bool:
        if x < 7 and y < 7:  # 左上
            return True
        elif x < 7 and y >= self.size - 7:  # 左下
            return True
        elif x >= self.size - 7 and y < 7:  # 右上
            return True
        return False

    def is_quiet_zone(self, x, y):
        if x < 8 and y == 7 or x == 7 and y < 8: # 左上
            return True
        elif x < 8 and y == self.size - 8 or x == 7 and y > self.size - 8:  # 左下
            return True
        elif x > self.size - 8 and y == 7 or x == self.size - 8 and y < 8:  # 右上
            return True

        return False

    def is_timing_pattern(self, x, y):
        if x < 8 and y == 7 or x == 7 and y < 8: # 左上
            return True
        elif x < 8 and y == self.size - 8 or x == 7 and y > self.size - 8:  # 左下
            return True
        elif x > self.size - 8 and y == 7 or x == self.size - 8 and y < 8:  # 右上
            return True

        return False

    def xor(self, bits, mask) -> List[int]:
        result = []
        for b, m in zip(bits, mask):
            result.append(b ^ m)

        return result

    def read_format(self):
        bits = []
        for x in range(8):
            if x == 6:
                continue
            bits.append(self.extract(x, 8))

        for y in range(8, -1, -1):
            if y == 6:
                continue
            bits.append(self.extract(8, y))

        bits = self.xor(bits, self.format_mask)

        error_correct_level = 0
        if bits[0] == 0 and bits[1] == 1:
            error_correct_level = 0
        elif bits[0] == 0 and bits[1] == 0:
            error_correct_level = 1
        elif bits[0] == 1 and bits[1] == 1:
            error_correct_level = 2
        elif bits[0] == 1 and bits[1] == 0:
            error_correct_level = 3

        print(f'Error correct {error_correct_level}')
        self.error_collect_level = error_correct_level

        mask_pattern = bits[2] * 4 + bits[3] * 2 + bits[4]
        print(f'Mask pattern {mask_pattern}')
        self.mask_pattern = mask_pattern

    def is_masked(self, x, y) -> bool:
        i = y
        j = x
        if self.mask_pattern == 0:
            return (i + j) % 2 == 0
        elif self.mask_pattern == 1:
            return i % 2 == 0
        elif self.mask_pattern == 2:
            return j % 3 == 0
        elif self.mask_pattern == 3:
            return (i + j) % 3 == 0
        elif self.mask_pattern == 4:
            return (i // 2 + j // 3) % 2 == 0
        elif self.mask_pattern == 5:
            return (i * j) % 2 + (i * j) % 3 == 0
        elif self.mask_pattern == 6:
            return ((i * j) % 2 + (i * j) % 3) % 2 == 0
        elif self.mask_pattern == 7:
            return ((i * j) % 3 + (i + j) % 2) % 2 == 0

        return False

    def read_contents(self):
        bits = [
            self.extract_with_mask(self.size - 1, self.size - 1),
            self.extract_with_mask(self.size - 2, self.size - 1),
            self.extract_with_mask(self.size - 1, self.size - 2),
            self.extract_with_mask(self.size - 2, self.size - 2),
            # self.extract(self.size - 1, self.size - 1),
            # self.extract(self.size - 2, self.size - 1),
            # self.extract(self.size - 1, self.size - 2),
            # self.extract(self.size - 2, self.size - 2),
        ]

    def read_data_mode(self):
        bits = [
            self.extract_with_mask(self.size - 1, self.size - 1),
            self.extract_with_mask(self.size - 2, self.size - 1),
            self.extract_with_mask(self.size - 1, self.size - 2),
            self.extract_with_mask(self.size - 2, self.size - 2),
            # self.extract(self.size - 1, self.size - 1),
            # self.extract(self.size - 2, self.size - 1),
            # self.extract(self.size - 1, self.size - 2),
            # self.extract(self.size - 2, self.size - 2),
        ]

        print(bits)

    def read_data_size(self):
        bits = [
            self.extract_with_mask(self.size - 1, self.size - 3),
            self.extract_with_mask(self.size - 2, self.size - 3),
            self.extract_with_mask(self.size - 1, self.size - 4),
            self.extract_with_mask(self.size - 2, self.size - 4),
            self.extract_with_mask(self.size - 1, self.size - 5),
            self.extract_with_mask(self.size - 2, self.size - 5),
            self.extract_with_mask(self.size - 1, self.size - 6),
            self.extract_with_mask(self.size - 2, self.size - 6),
            # self.extract(self.size - 1, self.size - 1),
            # self.extract(self.size - 2, self.size - 1),
            # self.extract(self.size - 1, self.size - 2),
            # self.extract(self.size - 2, self.size - 2),
        ]

        print(bits)

    def print_pattern(self, pattern_function):
        pattern = ''
        for y in range(self.size):
            for x in range(self.size):
                pattern += 'o' if pattern_function(x, y) else '.'
            pattern += '\n'
        print(pattern)

    def print_finder_pattern(self):
        self.print_pattern(self.is_finder_pattern)

    def print_quiet_zone(self):
        self.print_pattern(self.is_quiet_zone)


def check(img, qr_type):
    # qr_type ごとのサイズ
    qr_size = qr_type * 4 + 17

    qrd = cv2.QRCodeDetector()

    retval, points = qrd.detect(img)
    if retval:
        points = points.astype(np.int)

        for point in points:
            print(point)

            qr = QRCodeExtractor(img, point, qr_size)

            print("Check Finder Pattern")
            print(qr.validate_finder_pattern())

            qr.print_finder_pattern()

            print("Check Quiet")
            print(qr.validate_quiet_zone())

            qr.print_quiet_zone()

            print("Data Format")
            print(qr.read_format())

            print("Data Mode")
            print(qr.read_data_mode())

            print("Data Size")
            print(qr.read_data_size())


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'Usage: python {sys.argv[0]} image_file_path qr_type')
        exit(1)
    img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    check(img, int(sys.argv[2]))
