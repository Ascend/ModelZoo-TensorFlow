import os
import argparse
import numpy as np
import cv2



def error_out(s):
    print("error: " + s)


def to_image(filepath):
    bin_img = np.fromfile(filepath, dtype=np.float32).reshape((64, 128, 3))
    img = cv2.resize((bin_img + 1) * 127.5, (128, 64))
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", dest="source")
    parser.add_argument("-d", "--destination", dest="destination")
    args = parser.parse_args()
    destination = args.destination
    source = args.source

    if not os.path.isdir(destination):
        error_out("-d/--destination should be a folder.")
    file_list = []
    if os.path.isdir(source):
        for file_name in os.listdir(source):
            file_list.append(os.path.join(source, file_name))
    else:
        file_list.append(source)

    for file in file_list:
        img = to_image(file)
        cv2.imwrite(os.path.join(destination, os.path.basename(file)[:-4] + ".png"), img)


if __name__ == "__main__":
    main()