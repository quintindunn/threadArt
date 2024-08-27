import os
import sys

from PIL import Image, ImageOps, ImageDraw
import numpy as np

from skimage.draw import line

try:
    from .visualizer import StringVisualizer
except ImportError:
    from visualizer import StringVisualizer


import logging

logger = logging.getLogger("threadArt")


def _preprocess_image(img: str | os.PathLike | Image.Image, size: tuple[int, int]) -> Image.Image:
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
    logger.debug("Converted and cropped image to a grayscale circle.")
    return output


def process_image(im: Image.Image, board_width: int, pixel_width: int, nail_count: int, max_strings: int,
                  nails_skip: int, visualize: bool = False, progress: int = 200) -> tuple[list[int], Image.Image]:
    """
    Generates the string art representation of the image.
    :param im: Pillow Image to convert to string art.
    :param board_width: The width of the board.
    :param pixel_width: The line width.
    :param nail_count: Number of points around the board to lock onto.
    :param max_strings: The number of strings to draw.
    :param nails_skip: How many neighbors to skip when finding the next nail.
    :param visualize: Whether to show the visualizer.
    :param progress: Whether to have progress reports on, (Shown on logger level logging.INFO).
    :return:
    """
    pixels = int(board_width / pixel_width)
    size = (pixels + 1, pixels + 1)

    visualizer = None

    if visualize:
        visualizer = StringVisualizer(nail_count=nail_count, diameter=1024*2)
        visualizer.generate_nails(live_delay=10)

    base = Image.new('L', size, color=255)

    ref = _preprocess_image(im, size)

    angles = np.linspace(0, 2 * np.pi, nail_count)  # angles to the dots
    center_x, center_y = (board_width / 2 / pixel_width, board_width / 2 / pixel_width)  # center of circle

    xs = center_x + board_width * 0.5 * np.cos(angles) / pixel_width
    ys = center_y + board_width * 0.5 * np.sin(angles) / pixel_width

    nails = list(map(lambda x, y: (int(x), int(y)), xs, ys))

    current_nail = 0
    ref_arr = np.transpose(np.array(ref)[:, :, 0])

    sequence = []
    for i in range(max_strings):
        best_line = None
        new_nail = None
        min_avg_value = 10**20
        for n in range(current_nail + 1 + nails_skip, current_nail + len(nails) - nails_skip):
            n = n % nail_count
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

        if visualize and visualizer is not None:
            visualizer.add_line(current_nail, new_nail)
            visualizer.update(1)

        seq_len = len(sequence)
        if new_nail:
            logger.debug(f"String #{seq_len+1} completed {current_nail} -> {new_nail}")
            sequence.append(new_nail)

        if progress != -1 and seq_len % progress == 0 and seq_len != nail_count:
            logger.info(f"Processing: {seq_len/max_strings*100:.2f}%")

        current_nail = new_nail

    if progress != -1:
        logger.info("Processing: 100%")

    return sequence, base


def _main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    image_path = "../horse.jpg"
    width = 4000
    pixel_size = 1
    number_of_nails = 288
    max_strings = 4000
    nail_skip = 20

    result_file = "../results.txt"
    result_img = "result.png"

    progress_reports = 200  # After how many lines to print a progress report, 0 to never.
    use_visualizer = False

    settings = {
        "Width": width,
        "Pixel Size": pixel_size,
        "Number of Nails": number_of_nails,
        "Max Strings": max_strings,
        "Nail Skip": nail_skip
    }

    for k, v in settings.items():
        print(f"{k}: {v}")

    im = Image.open(image_path)
    seq, im = process_image(im=im, board_width=width, pixel_width=pixel_size, nail_count=number_of_nails,
                            max_strings=max_strings, nails_skip=nail_skip, visualize=use_visualizer,
                            progress=progress_reports)

    with open(result_file, 'w') as f:
        f.write(','.join(map(str, seq)))
    im.save(result_img)

    if progress_reports != -1:
        print(f"Saved result image as: \"{result_img}\", Sequence saved as \"{result_file}\"!")


if __name__ == '__main__':
    _main()
