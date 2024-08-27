import os

from PIL import Image, ImageOps, ImageDraw
import numpy as np

from skimage.draw import line

from visualizer import StringVisualizer


RESULT_FILE = "results.txt"
RESULT_IMG = "result.png"

PROGRESS_REPORTS = 200  # After how many lines to print a progress report, 0 to never.
USE_VISUALIZER = False


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


def process_image(board_width: int, pixel_width: int, nail_count: int, max_strings: int, nails_skip: int):
    pixels = int(board_width / pixel_width)
    size = (pixels + 1, pixels + 1)

    if USE_VISUALIZER:
        visualizer = StringVisualizer(nail_count=nail_count, diameter=1024*2)
        visualizer.generate_nails(live_delay=10)

    base = Image.new('L', size, color=255)

    ref = preprocess_image("./target/stella.jpg", size)

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

        if USE_VISUALIZER:
            visualizer.add_line(current_nail, new_nail)
            visualizer.update(1)

        sequence.append(new_nail)

        seq_len = len(sequence)
        if PROGRESS_REPORTS != -1 and seq_len % PROGRESS_REPORTS == 0 and seq_len != nail_count:
            print(f"Processing: {seq_len/max_strings*100:.2f}%")

        current_nail = new_nail

    if PROGRESS_REPORTS != -1:
        print("Processing: 100%")

    with open(RESULT_FILE, 'w') as f:
        f.write(",".join(map(str, sequence)))

    base.save(RESULT_IMG)

    if PROGRESS_REPORTS != -1:
        print(f"Saved result image as: \"{RESULT_IMG}\", Sequence saved as \"{RESULT_FILE}\"!")


def main():
    width = 4000
    pixel_size = 1
    number_of_nails = 288
    max_strings = 4000
    nail_skip = 20

    settings = {
        "Width": width,
        "Pixel Size": pixel_size,
        "Number of Nails": number_of_nails,
        "Max Strings": max_strings,
        "Nail Skip": nail_skip
    }

    for k, v in settings.items():
        print(f"{k}: {v}")

    process_image(board_width=width, pixel_width=pixel_size, nail_count=number_of_nails, max_strings=max_strings, nails_skip=nail_skip)


if __name__ == '__main__':
    main()
