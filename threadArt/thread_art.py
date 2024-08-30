import numpy as np
import cv2
import math
import time

"""
CREDIT: https://github.com/halfmonty/StringArtGenerator/blob/master/main.go
"""


# Constants
PINS = 300
MIN_DISTANCE = 30
MAX_LINES = 4000
LINE_WEIGHT = 8
IMG_SIZE = 500

# Globals
Pin_coords = []
SourceImage = []
Line_cache_y = []
Line_cache_x = []


def main():
    global SourceImage
    SourceImage = import_picture_and_get_pixel_array("../dog.jpg")
    print("Hello, world.")

    start_time = time.time()
    calculate_pin_coords()
    precalculate_all_potential_lines()
    calculate_lines()
    end_time = time.time()
    diff = end_time - start_time
    print(f"precalculateAllPotentialLines Taken: {diff:.6f} seconds")

    print("End")


def import_picture_and_get_pixel_array(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.flatten().astype(np.float64)


def calculate_pin_coords():
    global Pin_coords
    center = IMG_SIZE / 2
    radius = IMG_SIZE / 2 - 1

    Pin_coords = [
        (math.floor(center + radius * math.cos(2 * math.pi * i / PINS)),
         math.floor(center + radius * math.sin(2 * math.pi * i / PINS)))
        for i in range(PINS)
    ]


def precalculate_all_potential_lines():
    global Line_cache_y, Line_cache_x
    Line_cache_y = [[] for _ in range(PINS * PINS)]
    Line_cache_x = [[] for _ in range(PINS * PINS)]

    for i in range(PINS):
        for j in range(i + MIN_DISTANCE, PINS):
            x0, y0 = Pin_coords[i]
            x1, y1 = Pin_coords[j]

            d = math.floor(math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))
            xs = np.round(np.linspace(x0, x1, int(d))).astype(int)
            ys = np.round(np.linspace(y0, y1, int(d))).astype(int)

            Line_cache_y[j * PINS + i] = ys
            Line_cache_y[i * PINS + j] = ys
            Line_cache_x[j * PINS + i] = xs
            Line_cache_x[i * PINS + j] = xs


def calculate_lines():
    print("Drawing Lines....")
    error = 255 - SourceImage

    line_sequence = []
    current_pin = 0
    last_pins = [-1] * 20
    max_err = 0

    for _ in range(MAX_LINES):
        best_pin = -1
        line_err = 0
        max_err = 0
        index = 0

        for offset in range(MIN_DISTANCE, PINS - MIN_DISTANCE):
            test_pin = (current_pin + offset) % PINS
            if test_pin in last_pins:
                continue
            else:
                inner_index = test_pin * PINS + current_pin

                line_err = get_line_err(error, Line_cache_y[inner_index], Line_cache_x[inner_index])
                if line_err > max_err:
                    max_err = line_err
                    best_pin = test_pin
                    index = inner_index

        line_sequence.append(best_pin)

        for i in range(len(Line_cache_y[index])):
            v = int(Line_cache_y[index][i] * IMG_SIZE + Line_cache_x[index][i])
            error[v] -= LINE_WEIGHT

        last_pins = last_pins[1:] + [best_pin]
        current_pin = best_pin

    print(line_sequence)


def get_line_err(err, coords1, coords2):
    sum_err = 0
    for i in range(len(coords1)):
        sum_err += err[int(coords1[i] * IMG_SIZE + coords2[i])]
    return sum_err


if __name__ == "__main__":
    main()
