import os
import collections
import torch
import cv2

from model.python_files.gpunose_classifier import Nose_Network
from model.python_files.gpumouth_classifier import Mouth_Network
from model.python_files.gpumask_classifier import Mask_Network

from torchvision.transforms import ToTensor

NOSE_CLASSIFIER = 'model/model_archive/nose.pt'
MOUTH_CLASSIFIER = 'model/model_archive/mouth.pt'
MASK_CLASSIFIER = 'model/model_archive/mask.pt'

nose = Nose_Network()
mouth = Mouth_Network()
mask = Mask_Network()

nose.load_state_dict(torch.load(NOSE_CLASSIFIER, map_location=torch.device('cpu')))
mouth.load_state_dict(torch.load(MOUTH_CLASSIFIER, map_location=torch.device('cpu')))
mask.load_state_dict(torch.load(MASK_CLASSIFIER, map_location=torch.device('cpu')))

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

    files = os.listdir(f'data/small/{dir}/test')
    if len(files) > 400:
        files = files[:400]

    for file in files:
        labels[f'data/small/{dir}/test/{file}'] = dir_labels[dir]


nose.eval()
mouth.eval()
mask.eval()

correct = 0
total = len(labels)
with torch.no_grad():
    while labels:
        file, label = labels.popitem()
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        transform = ToTensor()
        image_tensor = transform(image)

        nose_scores = nose(image_tensor[None, ...])
        mouth_scores = mouth(image_tensor[None, ...])
        mask_scores = mouth(image_tensor[None, ...])

        nose_pred = torch.argmax(nose_scores)
        mouth_pred = torch.argmin(mouth_scores)
        mask_pred = torch.argmax(mask_scores)

        pred_label = None
        if mask_pred:
            if nose_pred or mouth_pred:
                pred_label = INCORRECT
            else:
                pred_label = CORRECT

        else:
            pred_label = NO_MASK

        correct += int(label == pred_label)

        print('Done with ' + file)

print("Accuracy: " + str(correct / total))
