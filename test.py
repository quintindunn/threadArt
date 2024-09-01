import logging
import sys

import cv2 as cv

from threadArt import KasperMeertsAlgorithm

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    test_file_dir = "./test/"
    test_files = [
        "cat.jpg",
        "dog.jpg",
        "goat.jpg",
        "horse.jpg",
        "pig.jpg",
        "stella.jpg"
    ]

    for file in test_files:
        file = test_file_dir + file
        im = cv.imread(file)

        art = KasperMeertsAlgorithm(
            im=im,
            n_pins=160,
            max_lines=4000,
            line_weight=20,
            min_distance=20,
            min_loop=20,
            use_visualizer=True
        )
        sequence = art.run()
        im = art.visualizer.im

        *prev, _ = file.split(".")
        final = f"{'.'.join(prev)}-sequence.txt"

        with open(final, 'w') as f:
            f.write(','.join(map(str, sequence)))

        *prev, ending = file.split(".")
        final = f"{'.'.join(prev)}-string.{ending}"
        cv.imwrite(final, im)
