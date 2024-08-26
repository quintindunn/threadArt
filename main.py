from PIL import Image, ImageOps, ImageDraw
from skimage.draw import line
import numpy as np
import sys

from visualizer import StringVisualizer

args = sys.argv
BOARD_WIDTH = 4000
PIXEL_WIDTH = 1
LINE_TRANSPARENCY = .125
NUM_NAILS = 288
MAX_ITERATIONS = 4000
NAILS_SKIP = 20
OUTPUT_TITLE = "output"

pixels = int(BOARD_WIDTH / PIXEL_WIDTH)
size = (pixels + 1, pixels + 1)


visualizer = StringVisualizer(nail_count=NUM_NAILS, diameter=BOARD_WIDTH)
visualizer.generate_nails(live_delay=10)


def crop_circle(path):
    img = Image.open(path).convert("L")
    img = img.resize(size)
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + size, fill=255)
    mask = mask.resize(img.size, Image.Resampling.LANCZOS)
    img.putalpha(mask)

    output = ImageOps.fit(img, mask.size, centering=(0.5, 0.5))
    output.putalpha(mask)
    return output


base = Image.new('L', size, color=255)

ref = crop_circle("../target/horse.jpg")


angles = np.linspace(0, 2 * np.pi, NUM_NAILS)  # angles to the dots
cx, cy = (BOARD_WIDTH / 2 / PIXEL_WIDTH, BOARD_WIDTH / 2 / PIXEL_WIDTH)  # center of circle
xs = cx + BOARD_WIDTH * 0.5 * np.cos(angles) / PIXEL_WIDTH
ys = cy + BOARD_WIDTH * 0.5 * np.sin(angles) / PIXEL_WIDTH
nails = list(map(lambda x, y: (int(x), int(y)), xs, ys))

cur_nail = 1
ref_arr = np.transpose(np.array(ref)[:, :, 0])
base_arr = base.load()


sequence = []
for i in range(MAX_ITERATIONS):
    best_line = None
    new_nail = None
    min_avg_value = 10000
    for n in range(cur_nail + 1 + NAILS_SKIP, cur_nail + len(nails) - NAILS_SKIP):
        n = n % NUM_NAILS
        new_line = line(nails[cur_nail][0], nails[cur_nail][1], nails[n][0], nails[n][1])
        num_pts = len(new_line[0])

        tmp_value = np.sum(ref_arr[new_line])

        if tmp_value / num_pts < min_avg_value:
            best_line = new_line
            new_nail = n
            min_avg_value = tmp_value / num_pts

    ref_arr[best_line] = 255

    addLine = ImageDraw.Draw(base)
    addLine.line((nails[cur_nail][0], nails[cur_nail][1], nails[new_nail][0], nails[new_nail][1]), fill=0)

    visualizer.add_line(cur_nail, new_nail)
    visualizer.update(1)

    sequence.append(new_nail)
    cur_nail = new_nail

with open("results.txt", 'w') as f:
    f.write(",".join(map(str, sequence)))


title = OUTPUT_TITLE + str(BOARD_WIDTH) + 'W-' + str(PIXEL_WIDTH) + "P-" + str(NUM_NAILS) + 'N-' + str(
    MAX_ITERATIONS) + '-' + str(LINE_TRANSPARENCY) + '.png'
base.save(title)
