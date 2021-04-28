# -*- coding: utf-8 -*-
"""GPUnose_classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/cv-final-project/cv-final-project/blob/main/model/notebooks/GPUNoseclassifier.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar
import os
from pathlib import Path
import shutil
from pprint import pprint
import random
import itertools

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchsummary import summary
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from PIL import Image

if torch.cuda.is_available():
    print("Using the GPU. You are good to go!")
    device = torch.device('cuda:0')
else:
    raise Exception("WARNING: Could not find GPU! Using CPU only. \
To enable GPU, please to go Edit > Notebook Settings > Hardware \
Accelerator and select GPU.")

! rm -rf sample_data/
! git clone https://eecs442finalproject:eecs442isgreat@github.com/cv-final-project/data_translations.git

def get_min_num_files(class_d, tt):
  mins = []
  for class_, dirs in class_d.items():
    t = sum([len(os.listdir(f'data_translations/{x}/{tt}')) for x in dirs])
    mins.append(t)
  return min(mins)
def create_classes(class_d):
  new_dir = f'org_data'
  os.mkdir(f'{new_dir}/')
  # print(min)
  for tt in ['train', 'test']:
    os.mkdir(f'{new_dir}/{tt}')
    min = get_min_num_files(class_d, tt)
    #################################################
    if tt == 'train':
      min = 1500
    else:
      min = 5000
    #################################################
    for class_, dirs_ in class_d.items():
      # print(class_, tt)
      dirs = [f'data_translations/{x}/{tt}' for x in dirs_]
      os.mkdir(f'{new_dir}/{tt}/{class_}')
      files = []
      for dir in dirs:
        files += [f'{dir}/{file}' for file in os.listdir(dir)]
      random.shuffle(files)
      files = files[:min]
      for file in files:
        shutil.copy2(file, f'{new_dir}/{tt}/{class_}')
        os.remove(file)

########
class_d = {
    'nose':['_Mask_Chin','_Mask_Mouth_Chin','no_mask'],
    'no_nose':['_Mask_Nose_Mouth','mask']
}
########
! rm -rf org_data/
create_classes(class_d)

! rm -rf data_translations/

###########HYPERPARAMETERS###########
BATCH_SIZE = 64
#####################################
def get_loader(tt, BATCH_SIZE):
  resize = transforms.Compose([
      transforms.Grayscale(num_output_channels=1),
      transforms.ToTensor()
  ])
  train_data = datasets.ImageFolder(f'org_data/{tt}', transform=resize)
  return DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
train_loader = get_loader('train', BATCH_SIZE)
val_loader = get_loader('test', BATCH_SIZE)

print(len(train_loader.dataset))
print(len(os.listdir('org_data/train/nose')))
print(len(os.listdir('org_data/train/no_nose')))
print(len(val_loader.dataset))
print(len(os.listdir('org_data/test/nose')))
print(len(os.listdir('org_data/test/no_nose')))

class Nose_Network(nn.Module):
  def __init__(self):
    super().__init__()
    ##############################################################################
    # TODO: Design your own network, define layers here.                         #
    # Here We provide a sample of two-layer fc network from HW4 Part3.           #
    # Your solution, however, should contain convolutional layers.               #
    # Refer to PyTorch documentations of torch.nn to pick your layers.           #
    # (https://pytorch.org/docs/stable/nn.html)                                  #
    # Some common choices: Linear, Conv2d, ReLU, MaxPool2d, AvgPool2d, Dropout   #
    # If you have many layers, use nn.Sequential() to simplify your code         #
    ##############################################################################
    self.in_dim = 1
    self.mid_layer_params = 32
    self.num_classes = 2
    self.cnn_layers_max = nn.Sequential(
        # Defining a 2D convolution layer
        nn.Conv2d(self.in_dim, self.mid_layer_params, kernel_size=3),
        nn.BatchNorm2d(self.mid_layer_params),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.cnn_layers_avg = nn.Sequential(
        nn.Conv2d(self.mid_layer_params, self.mid_layer_params, kernel_size=3),
        nn.BatchNorm2d(self.mid_layer_params),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )
    self.cnn_layers_max_2 = nn.Sequential(
        nn.Conv2d(self.mid_layer_params, self.mid_layer_params, kernel_size=3),
        nn.BatchNorm2d(self.mid_layer_params),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.linear_layers = nn.Sequential(
        nn.Linear(self.mid_layer_params*30**2, self.num_classes)
    )
    
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################

  def forward(self,x):
    ##############################################################################
    # TODO: Design your own network, implement forward pass here                 #
    ##############################################################################
    x = self.cnn_layers_max(x)
    x = self.cnn_layers_avg(x)
    x = self.cnn_layers_max_2(x)
    x = x.view(x.size(0), -1)
    x = self.linear_layers(x)
    return x
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################

model = Nose_Network().to(device)
criterion = nn.CrossEntropyLoss() # Specify the loss layer
print('Your network:')
print(summary(model, (1,256,256))) # visualize your model

##############################################################################
# TODO: Modify the lines below to experiment with different optimizers,      #
# parameters (such as learning rate) and number of epochs.                   #
##############################################################################
# Set up optimization hyperparameters
learning_rate = 1e-2
weight_decay = 1e-4
num_epoch = 10  # TODO: Choose an appropriate number of training epochs
optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                       weight_decay=weight_decay)
##############################################################################
#                             END OF YOUR CODE                               #
##############################################################################

#External References used
## https://discuss.pytorch.org/t/iterating-through-imagefolder-for-sample-target/82291/2
## ^how to iterate through ImageFolder

# Code to train the neural net
def train(model, trainloader, valloader, num_epoch = 10): # Train the model
  print("Start training...")
  trn_loss_hist = []
  trn_acc_hist = []
  val_acc_hist = []
  model.train() # Set the model to training mode
  for i in range(num_epoch):
    running_loss = []
    print('-----------------Epoch = %d-----------------' % (i+1))
    for batch, label in tqdm(trainloader):
      batch = batch.to(device)
      label = label.to(device)
      optimizer.zero_grad() # Clear gradients from the previous iteration
      pred = model(batch) # This will call Network.forward() that you implement
      loss = criterion(pred, label) # Calculate the loss
      running_loss.append(loss.item())
      loss.backward() # Backprop gradients to all tensors in the network
      optimizer.step() # Update trainable weights
    print("\n Epoch {} loss:{}".format(i+1,np.mean(running_loss)))

    # Keep track of training loss, accuracy, and validation loss
    trn_loss_hist.append(np.mean(running_loss))
    trn_acc_hist.append(evaluate(model, trainloader))
    print("\n Evaluate on validation set...")
    val_acc_hist.append(evaluate(model, valloader))
  print("Done!")
  return trn_loss_hist, trn_acc_hist, val_acc_hist

def evaluate(model, loader): # Evaluate accuracy on validation / test set
  model.eval() # Set the model to evaluation mode
  correct = 0
  with torch.no_grad(): # Do not calculate grident to speed up computation
    for batch, label in tqdm(loader):
      batch = batch.to(device)
      label = label.to(device)
      pred = model(batch)
      correct += (torch.argmax(pred,dim=1)==label).sum().item()
    acc = correct/len(loader.dataset)
    print("\n Evaluation accuracy: {}".format(acc))
    return acc
    
trn_loss_hist, trn_acc_hist, val_acc_hist = train(model, train_loader, 
                                                  val_loader, num_epoch)

##############################################################################
# TODO: Note down the evaluation accuracy on test set                        #
##############################################################################
#print("\n Evaluate on test set")
#evaluate(model, test_loader);

##############################################################################
# TODO: Submit the accuracy plot                                             #
##############################################################################
# visualize the training / validation accuracies
x = np.arange(num_epoch)
# train/val accuracies for MiniVGG
plt.figure()
plt.plot(x, trn_acc_hist)
plt.plot(x, val_acc_hist)
plt.legend(['Training', 'Validation'])
plt.xticks(x)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Nose Classification')
plt.gcf().set_size_inches(10, 5)
plt.savefig("nose_classifier_translation_performace.png")

torch.save(model.state_dict(), 'nose.pt')

model.eval()

img_dir = "org_data/test/no_nose/00043_Mask_T4.jpg"
img = Image.open(img_dir)
display(img)
img = transforms.Grayscale(num_output_channels=1)(img)
img = transforms.ToTensor()(img)
img = img.to(device)
pprint(img.shape)
res = model(img[None,...])
res = res.cpu()
nns = res.detach().numpy()[0,0]
print("no_nose score: " + str(nns))
ns = res.detach().numpy()[0,1]
print("nose score: " + str(ns))
if nns > ns:
  print("nose covered")
else:
  print("nose detected")

