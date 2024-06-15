import pandas as pd
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
import glob
import argparse
import os



def tiler(imnames, newpath, falsepath, slice_size, ext):
    for imname in imnames:
        im = Image.open(imname)
        imr = np.array(im, dtype=np.uint8)
        height = imr.shape[0]
        width = imr.shape[1]
        labname = imname.replace("images\\", "labels\\").replace(ext, ".txt")
        labels = pd.read_csv(labname, sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])

        labels[['x1', 'w']] = labels[['x1', 'w']] * width
        labels[['y1', 'h']] = labels[['y1', 'h']] * height

        boxes = []

        for row in labels.iterrows():
            x1 = row[1]['x1'] - row[1]['w'] / 2
            y1 = (height - row[1]['y1']) - row[1]['h'] / 2
            x2 = row[1]['x1'] + row[1]['w'] / 2
            y2 = (height - row[1]['y1']) + row[1]['h'] / 2

            boxes.append((int(row[1]['class']), Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])))

        counter = 0
        print('Image:', imname)

        for i in range((-1*height // slice_size*-1)):
            for j in range((-1*width // slice_size*-1)):
                x1 = j * slice_size
                y1 = height - (i * slice_size)
                x2 = ((j + 1) * slice_size) - 1
                y2 = (height - (i + 1) * slice_size) + 1

                pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                imsaved = False
                slice_labels = []

                for box in boxes:
                    if pol.intersects(box[1]):
                        inter = pol.intersection(box[1])

                        if not imsaved:
                            sliced = imr[i * slice_size:(i + 1) * slice_size, j * slice_size:(j + 1) * slice_size]
                            sliced_im = Image.fromarray(sliced)
                            filename = imname.split('\\')[-1]
                            slice_path = newpath + "/images/" + filename.replace(ext, f'_{i}_{j}{ext}')
                            slice_labels_path = newpath + "/labels/" + filename.replace(ext, f'_{i}_{j}.txt')
                            sliced_im.save(slice_path)
                            imsaved = True

                        new_box = inter.envelope

                        centre = new_box.centroid

                        x, y = new_box.exterior.coords.xy

                        new_width = (max(x) - min(x)) / slice_size
                        new_height = (max(y) - min(y)) / slice_size

                        new_x = (centre.coords.xy[0][0] - x1) / slice_size
                        new_y = (y1 - centre.coords.xy[1][0]) / slice_size

                        counter += 1

                        slice_labels.append([box[0], new_x, new_y, new_width, new_height])

                if len(slice_labels) > 0:
                    slice_df = pd.DataFrame(slice_labels, columns=['class', 'x1', 'y1', 'w', 'h'])
                    slice_df.to_csv(slice_labels_path, sep=' ', index=False, header=False, float_format='%.6f')

                if not imsaved and falsepath:
                    sliced = imr[i * slice_size:(i + 1) * slice_size, j * slice_size:(j + 1) * slice_size]
                    sliced_im = Image.fromarray(sliced)
                    filename = imname.split('\\')[-1]
                    slice_path = falsepath + "/images/" + filename.replace(ext, f'_{i}_{j}{ext}')
                    labels_sl_name = falsepath + "/labels/" + filename.replace(ext, f'_{i}_{j}.txt')
                    file = open(labels_sl_name, 'w+')
                    file.close


                    sliced_im.save(slice_path)
                    imsaved = True


                if not imsaved:
                    print('Error, '+filename.replace(ext, f'_{i}_{j}{ext}')+' not saved')

    print('Tiled '+str(len(imnames))+' images')



if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()

    parser.add_argument("-source", default="data",
                        help="Source folder with images and labels needed to be tiled")
    parser.add_argument("-target", default="tmp", help="Target folder for a new sliced dataset")
    parser.add_argument("-ext", default=".jpg", help="Image extension in a dataset. Default: .JPG")
    parser.add_argument("-falsefolder", default='tmp/empt', help="Folder for tiles without bounding boxes")
    parser.add_argument("-size", type=int, default=640, help="Size of a tile. Dafault: 640")


    args = parser.parse_args()

    imnames = glob.glob(f'{args.source}/images/*{args.ext}')
    labnames = glob.glob(f'{args.source}/labels/*.txt')

    if len(imnames) == 0:
        raise Exception("Source folder should contain some images")

    upfolder = os.path.join(args.source, '..')
    target_upfolder = os.path.join(args.target, '..')

    if args.falsefolder:
        if not os.path.exists(args.falsefolder):
            os.makedirs(args.falsefolder)

    if not os.path.exists(os.path.join(args.target, "images")):
        os.makedirs(os.path.join(args.target, "images"))

    if not os.path.exists(os.path.join(args.falsefolder, "images")):
        os.makedirs(os.path.join(args.falsefolder, "images"))

    if not os.path.exists(os.path.join(args.target, "labels")):
        os.makedirs(os.path.join(args.target, "labels"))

    if not os.path.exists(os.path.join(args.falsefolder, "labels")):
        os.makedirs(os.path.join(args.falsefolder, "labels"))

    tiler(imnames, args.target, args.falsefolder, args.size, args.ext)