import argparse
import glob
import json
import os
import numpy as np
import PIL.Image

import labelme


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_dir', help='input annotated directory')
    parser.add_argument('output_dir', help='output dataset directory')
    parser.add_argument('--labels', help='labels file', required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(os.path.join(args.output_dir, 'SegmentationClassPNG')):
        os.makedirs(os.path.join(args.output_dir, 'SegmentationClassPNG'))

    if not os.path.exists(os.path.join(args.output_dir, 'SegmentationClassVisualization')):
        os.makedirs(os.path.join(args.output_dir, 'SegmentationClassVisualization'))

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_name = line.strip()
        class_name_to_id[class_name] = i
        if i == 0:
            assert class_name == 'background'
        class_names.append(class_name)
    class_names = tuple(class_names)
    print('class_names:', class_names)

    colormap = labelme.utils.label_colormap(255)

    for label_file in glob.glob(os.path.join(args.input_dir, '*.json')):
        print('Generating dataset from:', label_file)
        with open(label_file) as f:
            base = os.path.splitext(os.path.basename(label_file))[0]
            out_png_file = os.path.join(
                args.output_dir, 'SegmentationClassPNG', base + '.png')
            out_viz_file = os.path.join(
                args.output_dir,
                'SegmentationClassVisualization',
                base + '.jpg')

            data = json.load(f)

            img_file = os.path.join(os.path.dirname(label_file), data['imagePath'])
            img = np.asarray(PIL.Image.open(img_file))

            lbl = labelme.utils.shapes_to_label(
                img_shape=img.shape,
                shapes=data['shapes'],
                label_name_to_value=class_name_to_id,
            )
            labelme.utils.lblsave(out_png_file, lbl)

            viz = labelme.utils.draw_label(
                lbl, img, class_names, colormap=colormap)
            PIL.Image.fromarray(viz).save(out_viz_file)


if __name__ == '__main__':
    main()
