from visualizer import StringVisualizer

import numpy as np
import cv2
import math
import time
import logging

logger = logging.getLogger("threadArt")

"""
CREDIT: https://github.com/halfmonty/StringArtGenerator/blob/master/main.go
"""


class HalfmontyAlgorithm:
    def __init__(self, im: str, pin_count: int, min_distance: int, max_lines: int, line_weight: int,
                 img_size: int, use_visualizer):
        self.im = im

        self.pins = pin_count
        self.min_distance = min_distance
        self.max_lines = max_lines
        self.line_weight = line_weight
        self.img_size = img_size

        self.pin_coords = []
        self.source_im = []
        self.line_cache_y = []
        self.line_cache_x = []

        self.sequence = []

        self.visualizer = None
        if use_visualizer:
            self.visualizer = StringVisualizer(nail_count=self.pins, diameter=1024*2)
            self.visualizer.generate_nails(live_delay=10)

    def import_picture_and_get_pixel_array(self) -> np.ndarray:
        img = cv2.resize(self.im, (self.img_size, self.img_size))
        return img.flatten().astype(np.float64)

    def calculate_pin_coords(self):
        center = self.img_size / 2
        radius = self.img_size / 2 - 1

        self.pin_coords = [
            (math.floor(center + radius * math.cos(2 * math.pi * i / self.pins)),
             math.floor(center + radius * math.sin(2 * math.pi * i / self.pins)))
            for i in range(self.pins)
        ]
        logger.info("Calculated pin coords.")

    def precalculate_all_potential_lines(self):
        self.line_cache_x: list = [[] for _ in range(self.pins * self.pins)]
        self.line_cache_y: list = [[] for _ in range(self.pins * self.pins)]

        for i in range(self.pins):
            for j in range(i + self.min_distance, self.pins):
                x0, y0 = self.pin_coords[i]
                x1, y1 = self.pin_coords[j]

                d = math.floor(math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))
                xs = np.round(np.linspace(x0, x1, int(d))).astype(int)
                ys = np.round(np.linspace(y0, y1, int(d))).astype(int)

                self.line_cache_y[j * self.pins + i] = ys
                self.line_cache_y[i * self.pins + j] = ys
                self.line_cache_x[j * self.pins + i] = xs
                self.line_cache_x[i * self.pins + j] = xs

    def calculate_line(self):
        logger.info("Drawing lines.")
        error = np.subtract(255, self.source_im)

        current_pin = 0
        last_pins = [-1] * 20

        for _ in range(self.max_lines):
            best_pin = -1
            max_err = 0
            index = 0

            for offset in range(self.min_distance, self.pins - self.min_distance):
                test_pin = (current_pin + offset) % self.pins
                if test_pin in last_pins:
                    continue
                else:
                    inner_index = test_pin * self.pins + current_pin

                    line_err = self.get_line_err(error, self.line_cache_y[inner_index], self.line_cache_x[inner_index])
                    if line_err > max_err:
                        max_err = line_err
                        best_pin = test_pin
                        index = inner_index

            self.sequence.append(best_pin)
            if self.visualizer is not None:
                self.visualizer.add_line(current_pin, best_pin)
                self.visualizer.update(1)

            for i in range(len(self.line_cache_y[index])):
                v = int(self.line_cache_y[index][i] * self.img_size + self.line_cache_x[index][i])
                error[v] -= self.line_weight

            last_pins = last_pins[1:] + [best_pin]
            current_pin = best_pin

        return self.sequence

    def get_line_err(self, err, coords1, coords2):
        sum_err = 0
        for i in range(len(coords1)):
            sum_err += err[int(coords1[i] * self.img_size + coords2[i])]
        return sum_err

    def run(self) -> list[int]:
        self.source_im = self.import_picture_and_get_pixel_array()

        start_time = time.time()
        self.calculate_pin_coords()
        self.precalculate_all_potential_lines()
        sequence = self.calculate_line()

        end_time = time.time()

        logger.info(f"Pre-calculation {end_time-start_time:.6f}s")

        logger.info("Done!")
        return sequence


if __name__ == "__main__":
    PINS = 300
    MIN_DISTANCE = 30
    MAX_LINES = 4000
    LINE_WEIGHT = 8
    IMG_SIZE = 500

    img = cv2.imread("../horse.jpg")

    art = HalfmontyAlgorithm(
        im=img,
        pin_count=PINS,
        min_distance=MIN_DISTANCE,
        max_lines=MAX_LINES,
        line_weight=LINE_WEIGHT,
        img_size=IMG_SIZE,
        use_visualizer=False
    )
    art.run()

    with open("../results.txt", 'w') as f:
        f.write(','.join(map(str, art.sequence)))
