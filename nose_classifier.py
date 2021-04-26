import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Displays a progress bar
import os
from pathlib import Path
import shutil
from pprint import pprint

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchsummary import summary
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler

 ########
class_d = {
    'nose':['chin/','mouth_chin/','no_mask/'],
    'no_nose':['mouth_nose/','mouth_nose_chin/']
}
########
#os.mkdir('org_data')
for class_ in class_d:
  class_d[class_] = ['data/train/'+x for x in class_d[class_]]
  if not os.path.isdir('org_data/'+class_):
    os.mkdir('org_data/'+class_)
  for folder_ in class_d[class_]:
    folder = os.listdir(folder_)
    for file in folder:
      shutil.copy2(folder_+file, 'org_data/'+class_)

      
len(os.listdir('org_data/no_nose')), len(os.listdir('org_data/nose'))

def get_class_weights(classes):
  weights = [len(os.listdir('org_data/'+class_)) for class_ in classes]
  weights = [x/min(weights) for x in weights]
  return weights
def get_loader(data, classes, BATCH_SIZE):
  weights = get_class_weights(classes)
  weights_per_sample = [0 for i in range(len(data))]
  for idx, (sample, target) in enumerate(data):
    weights_per_sample[idx] = weights[target]
  wrs = WeightedRandomSampler(weights=weights_per_sample, 
                              num_samples=len(weights_per_sample),
                              replacement=True)
  return DataLoader(data, batch_size=BATCH_SIZE, sampler=wrs)
def load_data_balanced(classes, BATCH_SIZE):
  grayscale_resize = transforms.Compose([
      transforms.Grayscale(num_output_channels=1),
      transforms.Resize((100, 100)),
      transforms.ToTensor()
  ])
  train_data = datasets.ImageFolder('org_data', transform=grayscale_resize)
  train_size = int(0.8*len(train_data))
  val_size = len(train_data) - train_size
  train_data, val_data = random_split(train_data, 
                                      [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))
  print("Getting train loader...")
  train_loader = get_loader(train_data, classes, BATCH_SIZE)
  print("Getting val loader...")
  val_loader = get_loader(val_data, classes, BATCH_SIZE)
  return train_loader, val_loader

###########HYPERPARAMETERS###########
BATCH_SIZE = 8
#####################################
train_loader, val_loader = load_data_balanced(class_d.keys(), BATCH_SIZE)
#print(train_loader, val_loader)

class Network(nn.Module):
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
        nn.Conv2d(1, self.mid_layer_params, kernel_size=3),
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
        nn.Linear(self.mid_layer_params*10**2, self.num_classes)
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

model = Network()
criterion = nn.CrossEntropyLoss() # Specify the loss layer
print('Your network:')
print(summary(model, (1,100,100))) # visualize your model

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

# Code to train the neural net
%%time
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
    
trn_loss_hist, trn_acc_hist, val_acc_hist = train(model, trainloader, 
                                                  valloader, num_epoch)

##############################################################################
# TODO: Note down the evaluation accuracy on test set                        #
##############################################################################
print("\n Evaluate on test set")
evaluate(model, testloader);

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
plt.title('fashion MNIST Classification')
plt.gcf().set_size_inches(10, 5)
plt.show()