from rembg import remove
from argparse import ArgumentParser
from PIL import Image
import os
from tqdm import tqdm

import numpy as np

IMG_EXT = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff','.tif', '.webp']

def process_file(input_folder, file, output_folder, alpha_matting, post_process_mask):

    if file.lower().endswith(tuple(IMG_EXT)):
        image_pil = Image.open(os.path.join(input_folder, file))
        file, ext = os.path.splitext(file)

        # Remove background
        if alpha_matting:
            front = remove(image_pil, alpha_matting=True, post_process_mask=post_process_mask, bgcolor=(0,0,0,0))
        else:
            front = remove(image_pil, alpha_matting=False, post_process_mask=post_process_mask, bgcolor=(0,0,0,0))

        front.save(os.path.join(output_folder, "alpha", file + '.png'))

        # pillow if one of three channel > 0, make it 255
        front = front.convert("RGBA")
        front_data = np.array(front)
        alpha = front_data[:,:,3]
        front_data[alpha > 0] = [255,255,255,255]
        front_data[alpha <= 0] = [0,0,0,255]

        front_mask = Image.fromarray(front_data)
        front_mask = front_mask.convert("L")
        front_mask.save(os.path.join(output_folder, "mask", file + '.png'))
    else:
        print(f"Unsupported file type: {file}")

if __name__ == '__main__':
    parser = ArgumentParser(description='Remove Image Background')
    parser.add_argument('-i','--input', help='Input image file or folder')
    parser.add_argument('-o','--output', help='Output folder')
    parser.add_argument('--alpha-matting', action='store_true', help='Use alpha matting')
    parser.add_argument('--post-process-mask', action='store_true', help='Post process mask')
    args = parser.parse_args()

    assert(args.input != args.output)
    assert(os.path.exists(args.input))
    
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "mask"), exist_ok=True) 
    os.makedirs(os.path.join(args.output, "alpha"), exist_ok=True)

    if os.path.isfile(args.input): # Process file
        parent_folder = os.path.dirname(args.input)
        process_file(parent_folder, args.input, args.output, args.alpha_matting, args.mask_bg, args.post_process_mask)
    elif os.path.isdir(args.input): # Process folder
        for file in tqdm(sorted(os.listdir(args.input))):
            process_file(args.input, file, args.output, args.alpha_matting, args.post_process_mask)
    else:
        print(f"Unsupported input type: {args.input}")


