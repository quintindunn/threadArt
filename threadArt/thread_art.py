import collections
import sys

try:
    from .visualizer import StringVisualizer
except ImportError:
    from visualizer import StringVisualizer

import numpy as np
import cv2 as cv
import logging
import math

logger = logging.getLogger("threadArt")

"""
CREDIT: https://gist.github.com/kaspermeerts/781f0137b361b51224dcab722ae387b4
Additional credit for assisting in implementation to: https://github.com/zekemccrary
"""


class KasperMeertsAlgorithm:
    def __init__(self, im: np.ndarray, max_lines: int, n_pins: int, min_loop: int, min_distance: int, line_weight: int,
                 scale: int = 25, hoop_diameter: int = 0.625, use_visualizer: bool = False):
        self.im: np.ndarray = im
        self.max_lines: int = max_lines
        self.n_pins: int = n_pins
        self.min_loop: int = min_loop
        self.min_distance: int = min_distance
        self.line_weight: int = line_weight
        self.scale: int = scale
        self.hoop_diameter: int = hoop_diameter

        self.visualizer = None
        if use_visualizer:
            self.visualizer = StringVisualizer(nail_count=n_pins, diameter=2048)
            self.visualizer.generate_nails(1)

        self.pin_coords = []
        self.center: float | None = None
        self.radius: float | None = None

        self.sequence: list[int] = []

        # Generate the caches
        self.line_cache_y = [None] * n_pins * n_pins
        self.line_cache_x = [None] * n_pins * n_pins
        self.line_cache_weight = [1] * n_pins * n_pins  # Turned out to be unnecessary, unused
        self.line_cache_length = [0] * n_pins * n_pins

        self.optimal = None

    def preprocess_im(self):
        height, width, _ = self.im.shape

        square_size = min(width, height)

        left = (width - square_size) // 2
        top = (height - square_size) // 2
        right = left + square_size
        bottom = top + square_size

        self.im = self.im[top:bottom, left:right]

        self.im = cv.cvtColor(self.im, cv.COLOR_BGR2GRAY)

        assert self.im.shape[0] == self.im.shape[1]

        length = self.im.shape[0]

        x, y = np.ogrid[0:length, 0:length]
        circle_mask = (x - length / 2) ** 2 + (y - length / 2) ** 2 > length / 2 * length / 2
        self.im[circle_mask] = 0xFF

    def calculate_pin_coords(self):
        self.center = self.im.shape[0] / 2
        self.radius = self.im.shape[0] / 2 - .5

        for i in range(self.n_pins):
            angle = 2 * math.pi * i / self.n_pins
            self.pin_coords.append((math.floor(self.center + self.radius * math.cos(angle)),
                                    math.floor(self.center + self.radius * math.sin(angle))))

    def precalculate_lines(self):
        for a in range(self.n_pins):
            for b in range(a + self.min_distance, self.n_pins):
                x0 = self.pin_coords[a][0]
                y0 = self.pin_coords[a][1]

                x1 = self.pin_coords[b][0]
                y1 = self.pin_coords[b][1]

                d = int(math.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)))

                xs = np.linspace(x0, x1, d, dtype=int)
                ys = np.linspace(y0, y1, d, dtype=int)

                self.line_cache_y: np.ndarray
                self.line_cache_x: np.ndarray
                self.line_cache_y[b * self.n_pins + a] = ys
                self.line_cache_y[a * self.n_pins + b] = ys
                self.line_cache_x[b * self.n_pins + a] = xs
                self.line_cache_x[a * self.n_pins + b] = xs
                self.line_cache_length[b * self.n_pins + a] = d
                self.line_cache_length[a * self.n_pins + b] = d

    def calculate_lines(self):
        error = np.ones(self.im.shape) * 0xFF - self.im.copy()

        length = self.im.shape[0]

        result = np.ones((self.im.shape[0] * self.scale, self.im.shape[1] * self.scale), np.uint8) * 0xFF
        line_mask = np.zeros(self.im.shape, np.float64)

        pin = 0
        self.sequence.append(pin)

        thread_length = 0

        last_pins = collections.deque(maxlen=self.min_loop)

        for line in range(self.max_lines):
            if line % 100 == 0:

                img_result = cv.resize(result, self.im.shape, interpolation=cv.INTER_AREA)

                diff = img_result - self.im
                mul = np.uint8(img_result < self.im) * 254 + 1
                abs_diff = diff * mul
                err = abs_diff.sum() / length*length

                if self.optimal is not None:
                    self.optimal = err, line if err < self.optimal[0] else self.optimal
                else:
                    self.optimal = err, line

                logger.debug(f"Line: {line} - {err}")

            max_err = -math.inf
            best_pin = -1

            # Find the line which will lower the error the most
            for offset in range(self.min_distance, self.n_pins - self.min_distance):
                test_pin = (pin + offset) % self.n_pins
                if test_pin in last_pins:
                    continue

                xs = self.line_cache_x[test_pin * self.n_pins + pin]
                ys = self.line_cache_y[test_pin * self.n_pins + pin]

                # Error defined as the sum of the brightness of each pixel in the original
                # The idea being that a wire can only darken pixels in the result
                line_err = np.sum(error[ys, xs]) * self.line_cache_weight[test_pin * self.n_pins + pin]

                if line_err > max_err:
                    max_err = line_err
                    best_pin = test_pin

            self.sequence.append(best_pin)

            if self.visualizer is not None:
                self.visualizer.next_line(best_pin)
                self.visualizer.update(1)

            xs = self.line_cache_x[best_pin * self.n_pins + pin]
            ys = self.line_cache_y[best_pin * self.n_pins + pin]
            weight = self.line_weight * self.line_cache_weight[best_pin * self.n_pins + pin]

            # Subtract the line from the error
            line_mask.fill(0)
            line_mask[ys, xs] = weight
            error = error - line_mask
            error.clip(0, 255)

            # Draw the line in the result
            self.pin_coords[pin][0]: np.ndarray
            self.pin_coords[pin][1]: np.ndarray
            self.pin_coords[best_pin][0]: np.ndarray
            self.pin_coords[best_pin][1]: np.ndarray

            pt1 = self.pin_coords[pin][0] * self.scale, self.pin_coords[pin][1] * self.scale
            pt2 = self.pin_coords[best_pin][0] * self.scale, self.pin_coords[best_pin][1] * self.scale
            cv.line(result, pt1, pt2, color=(0, 0, 0), thickness=4, lineType=8)

            x0 = self.pin_coords[pin][0]
            y0 = self.pin_coords[pin][1]

            x1 = self.pin_coords[best_pin][0]
            y1 = self.pin_coords[best_pin][1]

            # Calculate physical distance
            dist = math.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))
            thread_length += self.hoop_diameter / self.im.shape[0] * dist

            last_pins.append(best_pin)
            pin = best_pin

    def run(self) -> list[int]:
        self.preprocess_im()
        self.calculate_pin_coords()
        self.precalculate_lines()
        self.calculate_lines()
        return self.sequence


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    PINS = 300
    MIN_DISTANCE = 30
    MAX_LINES = 4000
    LINE_WEIGHT = 8
    MIN_LOOP = 20

    img = cv.imread("../horse.jpg")

    art = KasperMeertsAlgorithm(
        im=img,
        n_pins=PINS,
        min_distance=MIN_DISTANCE,
        min_loop=MIN_LOOP,
        max_lines=MAX_LINES,
        line_weight=LINE_WEIGHT,
        use_visualizer=True
    )
    art.run()

    with open("../results.txt", 'w') as f:
        f.write(','.join(map(str, art.sequence)))
