import cv2
import argparse
import os
from glob import glob


def converter():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_dir', help='input dataset directory')
    parser.add_argument('output_dir', default='dataset', help='output dataset directory')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print('Output directory created')

    file_template = '*.tif'
    for i, img in enumerate(glob(os.path.join(args.input_dir, file_template))):
        # read as colored imaged to ensure the correct conversion
        tiff = cv2.imread(img, cv2.IMREAD_COLOR)
        # convert tif to jpg
        save_name = img.split('/')[-1].replace('tif', 'jpg')
        save_dir = os.path.join(args.output_dir, save_name)
        # make image brighter by a factor of 10
        tiff = tiff * 10
        cv2.imwrite(save_dir, tiff)

        print('converting..' + str(i))


if __name__ == '__main__':
    converter()

