import os

from PIL import Image, ImageOps, ImageDraw
import numpy as np

from skimage.draw import line

from visualizer import StringVisualizer

BOARD_WIDTH = 4000
PIXEL_WIDTH = 1
LINE_TRANSPARENCY = .125
NUM_NAILS = 288
MAX_ITERATIONS = 4000
NAILS_SKIP = 20


def preprocess_image(img: str | os.PathLike | Image.Image, size: tuple[int, int]) -> Image.Image:
    """
    Converts image to grayscale, crops image to a circle. Makes it compatible with the string art algorithm.
    :param img: Filepath or an Image object.
    :param size: The size of the output.
    :return: Image
    """

    if isinstance(img, (str, os.PathLike)):
        img = Image.open(img)

    img = img.convert("L").resize(size)
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + size, fill=255)
    mask = mask.resize(img.size, Image.Resampling.LANCZOS)
    img.putalpha(mask)

    output = ImageOps.fit(img, mask.size, centering=(0.5, 0.5))
    output.putalpha(mask)
    return output


def main():
    pixels = int(BOARD_WIDTH / PIXEL_WIDTH)
    size = (pixels + 1, pixels + 1)

    visualizer = StringVisualizer(nail_count=NUM_NAILS, diameter=BOARD_WIDTH)
    visualizer.generate_nails(live_delay=10)

    base = Image.new('L', size, color=255)

    ref = preprocess_image("./target/horse.jpg", size)

    angles = np.linspace(0, 2 * np.pi, NUM_NAILS)  # angles to the dots
    center_x, center_y = (BOARD_WIDTH / 2 / PIXEL_WIDTH, BOARD_WIDTH / 2 / PIXEL_WIDTH)  # center of circle

    xs = center_x + BOARD_WIDTH * 0.5 * np.cos(angles) / PIXEL_WIDTH
    ys = center_y + BOARD_WIDTH * 0.5 * np.sin(angles) / PIXEL_WIDTH

    nails = list(map(lambda x, y: (int(x), int(y)), xs, ys))

    current_nail = 0
    ref_arr = np.transpose(np.array(ref)[:, :, 0])

    sequence = []
    for i in range(MAX_ITERATIONS):
        best_line = None
        new_nail = None
        min_avg_value = 10**20
        for n in range(current_nail + 1 + NAILS_SKIP, current_nail + len(nails) - NAILS_SKIP):
            n = n % NUM_NAILS
            new_line = line(nails[current_nail][0], nails[current_nail][1], nails[n][0], nails[n][1])
            num_pts = len(new_line[0])

            tmp_value = np.sum(ref_arr[new_line])

            if tmp_value / num_pts < min_avg_value:
                best_line = new_line
                new_nail = n
                min_avg_value = tmp_value / num_pts

        ref_arr[best_line] = 255

        new_line = ImageDraw.Draw(base)
        new_line.line((nails[current_nail][0], nails[current_nail][1], nails[new_nail][0], nails[new_nail][1]),
                      fill=0)

        visualizer.add_line(current_nail, new_nail)
        visualizer.update(1)

        sequence.append(new_nail)
        current_nail = new_nail

    with open("results.txt", 'w') as f:
        f.write(",".join(map(str, sequence)))

    base.save("result.png")


if __name__ == '__main__':
    main()
