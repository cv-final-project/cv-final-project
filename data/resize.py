import os
from string import digits
import cv2


for i in range(21):
    dir = f'raw/{str(i).zfill(2)}000'
    for file in os.listdir(dir):
        image = cv2.imread(f'{dir}/{file}')
        resized = cv2.resize(image, (256, 256))

        cv2.imwrite(f'small/no_mask/{file}', resized)

    print(f'Done resizing for directory {dir}')
