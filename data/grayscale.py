import os
import cv2

for dir in os.listdir('small'):
    if not os.path.isdir(f'small/{dir}'):
        continue
        
    train_path = f'small/{dir}/train'
    test_path = f'small/{dir}/test'

    for file in os.listdir(train_path):
        image = cv2.imread(f'{train_path}/{file}')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'{train_path}/{file}', gray)

    for file in os.listdir(test_path):
        image = cv2.imread(f'{test_path}/{file}')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'{test_path}/{file}', gray)

    print('Done with directory ' + dir)
