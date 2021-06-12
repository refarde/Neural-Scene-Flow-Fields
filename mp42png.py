import cv2
import os
import errno
import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument('--input_path', type=str, help='Video file path')
parser.add_argument('--output_dir', type=str,
                    default="./nsff_data/dense/images", help='output images\' directory path')
args = parser.parse_args()
input = args.input_path
output = args.output_dir

if os.path.isfile(input) == False:
    raise Exception("Wrong Parameter: No file(input_path)") 

try:
    os.makedirs(output)
except OSError as exception:
    if exception.errno != errno.EEXIST:
        raise

vidcap = cv2.VideoCapture(input)
success, image = vidcap.read()
count = 0
index = 0

print(success)

while success:
    if count % 3 == 0:
        path = output + "/%05d.png" % index
        cv2.imwrite(path, image)     # save frame as PNG file
        print('Write a frame: ', path)
        index += 1

        # max 30 frame
        if index >= 30:
            break

    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
