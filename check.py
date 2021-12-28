import itertools
import sys
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Rect:
    left: int = 0
    top: int = 0
    width: int = 0
    height: int = 0

    @property
    def right(self) -> int:
        # inclusive
        return self.left + self.width - 1

    @property
    def bottom(self) -> int:
        # inclusive
        return self.top + self.height - 1

    def contains(self, x, y) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom

    def inner(self, size: int) -> 'Rect':
        return Rect(self.left+size, self.top+size, self.width-size*2, self.height-size*2)


class QRCodeExtractor:
    format_mask = [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]

    size_bits = {
        0b0001: 10,  # number
        0b0010: 9,  # alphanumeric
        0b0100: 8,  # bytes
        0b1000: 8,  # Kanji
    }

    def __init__(self, img, rect, size):
        self.img = img
        self.tl_x = rect[0][1]
        self.tl_y = rect[0][0]
        self.br_x = rect[2][1]
        self.br_y = rect[2][0]
        self.size = size
        self.scale = (self.br_x - self.tl_x + 1) // size
        self.contents = []

        self.error_collect_level = 0
        self.mask_pattern = 0
        self.data_mode = 0

    def read_all(self):
        self.read_format()
        self.read_contents()

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

    @property
    def finder_rects(self) -> List[Rect]:
        return [
            Rect(0, 0, 7, 7),
            Rect(0, self.size - 7, 7, 7),
            Rect(self.size - 7, 0, 7, 7)
        ]

    @property
    def format_points(self) -> (List[Tuple[int, int]], List[Tuple[int, int]]):
        format_points_1 = []
        for x in range(8):
            format_points_1.append((x, 8))

        for y in range(8, -1, -1):
            format_points_1.append((8, y))

        format_points_2 = []

        for y in range(self.size - 1, self.size - 1 - 7, -1):
            format_points_2.append((8, y))

        for x in range(self.size - 1 - 7, self.size):
            format_points_2.append((x, 8))

        format_points_1 = list(filter(lambda p: not self.is_timing_pattern(p[0], p[1]), format_points_1))
        format_points_2 = list(filter(lambda p: not self.is_timing_pattern(p[0], p[1]), format_points_2))

        return format_points_1, format_points_2

    def validate_finder_pattern(self) -> bool:
        rects = self.finder_rects

        for x, y in itertools.product(range(self.size), range(self.size)):
            for rect in rects:
                inner_rect = rect.inner(1)
                more_inner_rect = inner_rect.inner(1)

                if rect.contains(x, y):
                    if inner_rect.contains(x, y) and not more_inner_rect.contains(x, y):
                        self.assert_white(x, y)
                    else:
                        self.assert_black(x, y)
                    break

        return True

    def validate_quiet_zone(self) -> bool:
        for x in range(self.size):
            for y in range(self.size):
                if self.is_quiet_zone(x, y):
                    self.assert_white(x, y)

        return True

    def validate_format_zone(self) -> bool:
        format_points = self.format_points
        for i in range(len(format_points[0])):
            if self.extract(*format_points[0][i]) != self.extract(*format_points[1][i]):
                raise Exception(f'Format point is not same. ({format_points[0][i]} != {format_points[1][i]}')
        return True

    def validate_timing_pattern(self) -> bool:
        timing_pattern = []
        for x in range(self.size):
            for y in range(self.size):
                if self.is_timing_pattern(x, y):
                    timing_pattern.append((x, y))

        for i in range(len(timing_pattern) // 2):
            if i % 2 == 0:
                self.assert_black(*timing_pattern[i])
                self.assert_black(*timing_pattern[i + len(timing_pattern) // 2])
            else:
                self.assert_white(*timing_pattern[i])
                self.assert_white(*timing_pattern[i + len(timing_pattern) // 2])
        return True

    def is_contents_zone(self, x, y) -> bool:
        return not (self.is_finder_pattern(x, y) or self.is_quiet_zone(x, y) or self.is_format_zone(x, y) or self.is_timing_pattern(x, y))

    def is_finder_pattern(self, x, y) -> bool:
        rects = self.finder_rects

        for rect in rects:
            if rect.contains(x, y):
                return True

        return False

    def is_quiet_zone(self, x, y):
        if x < 8 and y == 7 or x == 7 and y < 8:  # 左上
            return True
        elif x < 8 and y == self.size - 8 or x == 7 and y > self.size - 8:  # 左下
            return True
        elif x > self.size - 8 and y == 7 or x == self.size - 8 and y < 8:  # 右上
            return True

        return False

    def is_format_zone(self, x, y):
        for format_points in self.format_points:
            if (x, y) in format_points:
                return True

        return False

    def is_timing_pattern(self, x, y):
        if x == 6 and 7 < y < self.size - 8:
            return True
        if 7 < x < self.size - 8 and y == 6:
            return True

        return False

    @staticmethod
    def xor(bits, mask) -> List[int]:
        result = []
        for b, m in zip(bits, mask):
            result.append(b ^ m)

        return result

    def read_format(self):
        format_points = self.format_points

        bits = []

        for x, y in format_points[0]:
            bits.append(self.extract(x, y))

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
        print(f'Mask pattern {bin(mask_pattern)}')
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

    def get_next_read_vector_x(self, x, y):
        if x <= 6:  # left timing pattern
            if x == 6:
                if y == 0:
                    return -1
                else:
                    return 0
            x = x + 1

        if x % 2 == 0:
            return -1
        else:
            if y == 0 and x % 4 != 1 or y == self.size - 1 and x % 4 == 1:
                return -1
            return 1

    def get_next_read_vector_y(self, x, y):
        if x <= 6:  # left timing pattern
            if x == 6:
                return 0
            x = x + 1

        if y == 0:
            if x % 4 != 1:
                return 0
            else:
                return 1
        elif y == self.size - 1:
            if x % 4 != 3:
                return 0
            else:
                return -1
        elif x % 2 == 0:
            return 0
        elif x % 4 in [1, 2]:
            return 1
        else:
            return -1

    def read_contents(self):
        x = self.size - 1
        y = self.size - 1
        contents = []
        while x >= 0 and y >= 0:
            if self.is_contents_zone(x, y):
                contents.append(self.extract_with_mask(x, y))

            vector_x = self.get_next_read_vector_x(x, y)
            vector_y = self.get_next_read_vector_y(x, y)

            x += vector_x
            y += vector_y

        self.contents = contents
        print(f'Contents {contents}')

        data_pointer = self.contents.copy()
        # read data mode
        data_mode = 0
        for i in range(4):
            data_mode = data_mode << 1
            data_mode += data_pointer.pop(0)

        print(f'Data Mode {bin(data_mode)}')
        # read data size
        if data_mode not in self.size_bits:
            print(f'Unknown data type: {bin(data_mode)}')
            data_mode = 0b0100
        data_size = 0
        for i in range(self.size_bits[data_mode]):
            data_size = data_size << 1
            data_size += data_pointer.pop(0)

        print(f'Data Size {data_size}')

        # 8 bits only. TODO: Other data type
        data_bytes = []
        for i in range(data_size):
            byte = 0
            for j in range(8):
                byte = byte << 1
                byte += data_pointer.pop(0)

            data_bytes.append(byte)

        print(f'Actual Data {list(map(lambda b: hex(b), data_bytes))}')

        # termination
        termination = []
        for i in range(4):
            termination.append(data_pointer.pop(0))

        print(f'Termination {termination}')

    def print_pattern(self, pattern_function):
        pattern = ''
        for y in range(self.size):
            for x in range(self.size):
                pattern += 'o ' if pattern_function(x, y) else '. '
            pattern += '\n'
        print(pattern)

    def print_raw_pattern(self):
        self.print_pattern(lambda x, y: self.extract(x, y) > 0)

    def print_finder_pattern(self):
        self.print_pattern(self.is_finder_pattern)

    def print_quiet_zone(self):
        self.print_pattern(self.is_quiet_zone)

    def print_format_zone(self):
        self.print_pattern(self.is_format_zone)

    def print_timing_pattern(self):
        self.print_pattern(self.is_timing_pattern)

    def print_mask_pattern(self):
        self.print_pattern(self.is_masked)

    def print_read_vector(self):
        pattern = ''
        for y in range(self.size):
            for x in range(self.size):
                if self.is_finder_pattern(x, y) or self.is_quiet_zone(x, y) or self.is_format_zone(x, y) or self.is_timing_pattern(x, y):
                    # pattern += '.'
                    # continue
                    pass

                vector_x = self.get_next_read_vector_x(x, y)
                vector_y = self.get_next_read_vector_y(x, y)
                if vector_y > 0:
                    if vector_x > 0:
                        pattern += '┘'
                    elif vector_x < 0:
                        pattern += '└'
                    else:
                        pattern += 'v'
                elif vector_y < 0:
                    if vector_x > 0:
                        pattern += '┐'
                    elif vector_x < 0:
                        pattern += '┌'
                    else:
                        pattern += '^'
                else:
                    if vector_x > 0:
                        pattern += '>'
                    elif vector_x < 0:
                        pattern += '<'
                    else:
                        pattern += '.'
            pattern += '\n'
        print(pattern)


def check(img, qr_type):
    # qr_type ごとのサイズ
    qr_size = qr_type * 4 + 17

    qrd = cv2.QRCodeDetector()

    retval, points = qrd.detect(img)
    if retval:
        points = points.astype(int)

        for point in points:
            print(point)

            qr = QRCodeExtractor(img, point, qr_size)

            # print("Raw Pattern")
            # qr.print_raw_pattern()
            #
            # print("Check Finder Pattern")
            # qr.validate_finder_pattern()
            # qr.print_finder_pattern()
            #
            # print("Check Quiet")
            # qr.validate_quiet_zone()
            # qr.print_quiet_zone()
            #
            # print("Check Format Zone")
            # qr.validate_format_zone()
            # qr.print_format_zone()
            #
            # print("Check Timing Pattern")
            # qr.validate_timing_pattern()
            # qr.print_timing_pattern()
            #
            # print("Read Vector")
            # print(qr.print_read_vector())
            qr.read_all()

            # qr.print_mask_pattern()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'Usage: python {sys.argv[0]} image_file_path qr_type')
        exit(1)
    image = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    check(image, int(sys.argv[2]))
