import os
import collections
import torch
import cv2

from torchvision.transforms import ToTensor

NOSE_CLASSIFIER = ''
MOUTH_CLASSIFIER = ''
MASK_CLASSIFIER = ''

nose = torch.load(NOSE_CLASSIFIER)
mouth = torch.load(MOUTH_CLASSIFIER)
mask = torch.load(MASK_CLASSIFIER)

NO_MASK, INCORRECT, CORRECT = 0, 1, 2
labels = {}
dir_labels = {
    '_Mask_Chin': 1,
    '_Mask_Mouth_Chin': 1,
    '_Mask_Nose_Mouth': 2,
    'mask': 2,
    'no_mask': 0,
}

for dir in os.listdir('data/small'):
    if not os.path.isdir(f'data/small/{dir}'):
        continue

    files = os.listdir('data/small/{dir}/test')
    for file in files:
        labels[file] = dir_labels[dir]


nose.eval()
mouth.eval()
mask.eval()

correct = 0
total = len(labels)
with torch.no_grad():
    while labels:
        file, label = labels.popitem()
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (100, 100))
        image_tensor = ToTensor(image)

        nose_pred = torch.argmax(nose(image_tensor))
        mouth_pred = torch.argmax(mouth(image_tensor))
        mask_pred = torch.argmax(mask(image_tensor))

        pred_label = None
        if mask_pred:
            if nose_pred or mouth_pred:
                pred_label = INCORRECT
            else:
                pred_label = CORRECT

        else:
            pred_label = NO_MASK

        correct += int(label == pred_label)

print("Accuracy: " + (correct / total))
