import blur_detector
from argparse import ArgumentParser
from PIL import Image
import os
from tqdm import tqdm

import numpy as np
from cv2 import threshold, THRESH_BINARY

IMG_EXT = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff','.tif', '.webp']

def process_file(input_folder, file, output_folder, mask_bg):

    if file.lower().endswith(tuple(IMG_EXT)):
        image_pil = Image.open(os.path.join(input_folder, file))
        file, ext = os.path.splitext(file)

        # Convert to grayscale
        image_pil = image_pil.convert("L")
        image = np.array(image_pil)

        # Detect blur
        blur = blur_detector.detectBlur(image, downsampling_factor=4, num_scales=4, scale_start=2, num_iterations_RF_filter=3, show_progress=False)
        _, th1 = threshold(blur, 0.1, 1, THRESH_BINARY)

        # save numpy array as an image
        th1 = Image.fromarray(th1.astype(np.uint8) * 255)
        th1 = th1.convert("L")
        th1.save(os.path.join(output_folder, 'blur', file + '.png'))
    else:
        print(f"Unsupported file type: {file}")

if __name__ == '__main__':
    parser = ArgumentParser(description='Remove Image Blurred Area')
    parser.add_argument('-i','--input', help='Input image file or folder')
    parser.add_argument('-o','--output', help='Output folder')
    parser.add_argument('--alpha-matting', action='store_true', help='Use alpha matting')
    parser.add_argument('--mask-bg', action='store_true', help='Mask background')
    parser.add_argument('--post-process-mask', action='store_true', help='Post process mask')
    args = parser.parse_args()

    assert(args.input != args.output)
    assert(os.path.exists(args.input))

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'blur'), exist_ok=True)

    if os.path.isfile(args.input): # Process file
        parent_folder = os.path.dirname(args.input)
        process_file(parent_folder, args.input, args.output, args.mask_bg)
    elif os.path.isdir(args.input): # Process folder
        for file in tqdm(sorted(os.listdir(args.input))):
            process_file(args.input, file, args.output, args.mask_bg)
    else:
        print(f"Unsupported input type: {args.input}")


