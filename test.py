if __name__ == "__main__":
    import logging
    import sys
    from PIL import Image
    from threadArt import process_image

    image_path = "./horse.jpg"
    width = 4000
    pixel_size = 1
    number_of_nails = 288
    max_strings = 4000
    nail_skip = 20

    progress_reports = 200  # After how many lines to print a progress report, 0 to never.
    use_visualizer = True

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

    print(*seq, sep=", ")
    im.show()
