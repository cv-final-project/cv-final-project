import os
import random
import shutil

from sklearn.model_selection import train_test_split

for dir in os.listdir('small'):
    if not os.path.isdir(f'small/{dir}'):
        continue


    files = os.listdir(f'small/{dir}')
    random.shuffle(files)
    os.mkdir(f'small/{dir}/train')
    os.mkdir(f'small/{dir}/test')

    train, test = train_test_split(files, test_size=0.2)
    for file in train:
        shutil.move(f'small/{dir}/{file}', f'small/{dir}/train/')

    for file in test:
        shutil.move(f'small/{dir}/{file}', f'small/{dir}/test/')

    print(f'Done with {dir}')
