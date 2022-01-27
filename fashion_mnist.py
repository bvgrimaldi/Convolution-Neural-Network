############################################################
# CIS 521: Neural Network for Fashion MNIST Dataset
############################################################

student_name = "Type your full name here."

############################################################
# Imports
############################################################

from PIL.Image import new
from numpy.core.numeric import full
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools

# Include your imports here, if any are used.



############################################################
# Neural Networks
############################################################

def load_data(file_path, reshape_images):
    file_name = open(file_path,'r')
    x_arr = np.zeros((20000,784))
    file_name.readline()
    y_arr = np.zeros(20000)
    last_ctr = 0
    lin_ctr = 0
    for line in file_name:

        ctr = 0
        reg_ctr = 0
        var = ''
        for char in line:
            
            if ctr == 0:
                y_arr[lin_ctr] = char
                ctr += 1

            if char != ',':
                var += char
                
            elif len(var) != 0:
                x_arr[lin_ctr,reg_ctr] = var
                reg_ctr += 1
                var = ''

            last_ctr = reg_ctr

        lin_ctr += 1
    if len(var)!= 0:
        x_arr[lin_ctr-1,last_ctr-1] = var
        
    if reshape_images == True:
        x_arr = x_arr.reshape(20000,1,28,28)


    return x_arr,y_arr




# PART 2.2
class EasyModel(torch.nn.Module):
    def __init__(self):
        super(EasyModel, self).__init__()
        self.fc = torch.nn.Linear(784,10)

    def forward(self, x):
        x = self.fc(x)
        return x


# PART 2.3
class MediumModel(torch.nn.Module):
    def __init__(self):
        super(MediumModel, self).__init__()
        hid_size = 500
        self.fc1 = torch.nn.Linear(784,hid_size)
        self.fc2 = torch.nn.Linear(hid_size,hid_size)
        self.fc3 = torch.nn.Linear(hid_size,10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.nn.functional.log_softmax(x)


# PART 2.4
class AdvancedModel(torch.nn.Module):
    def __init__(self):
        super(AdvancedModel, self).__init__()
        self.conv_layer = torch.nn.Sequential(
                  torch.nn.Conv2d(1, 16, kernel_size=5, padding=2),
                  torch.nn.Conv2d(16, 12, kernel_size=5, padding=2),
                  torch.nn.BatchNorm2d(12),
                  torch.nn.ReLU(),
                  torch.nn.MaxPool2d(2)
                  )

        self.forward_layer = torch.nn.Linear(2352, 10)

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1) 
        x = self.forward_layer(x)
        return x


############################################################
# Fashion MNIST dataset
############################################################

class FashionMNISTDataset(Dataset):
    def __init__(self, file_path, reshape_images):
        self.X, self.Y = load_data(file_path, reshape_images)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

############################################################
# Reference Code
############################################################

def train(model, data_loader, num_epochs, learning_rate):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(data_loader):
            images = torch.autograd.Variable(images.float())
            images = images.type(torch.LongTensor)
            labels = torch.autograd.Variable(labels)
            labels = labels.type(torch.LongTensor)

            optimizer.zero_grad()
            outputs = model(images.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                y_true, y_predicted = evaluate(model, data_loader)
                print(f'Epoch : {epoch}/{num_epochs}, '
                      f'Iteration : {i}/{len(data_loader)},  '
                      f'Loss: {loss.item():.4f},',
                      f'Train Accuracy: {100.* accuracy_score(y_true, y_predicted):.4f},',
                      f'Train F1 Score: {100.* f1_score(y_true, y_predicted, average="weighted"):.4f}')


def evaluate(model, data_loader):
    model.eval()
    y_true = []
    y_predicted = []
    for images, labels in data_loader:
        images = torch.autograd.Variable(images.float())
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels)
        y_predicted.extend(predicted)
    return y_true, y_predicted


def plot_confusion_matrix(cm, class_names, title=None):
    plt.figure()
    if title:
        plt.title(title)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def main():
    class_names = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    num_epochs = 2
    batch_size = 100
    learning_rate = 0.001
    file_path = 'dataset.csv'

    data_loader = torch.utils.data.DataLoader(dataset=FashionMNISTDataset(file_path, False),
                                              batch_size=batch_size,
                                              shuffle=True)
    data_loader_reshaped = torch.utils.data.DataLoader(dataset=FashionMNISTDataset(file_path, True),
                                                       batch_size=batch_size,
                                                       shuffle=True)

    # EASY MODEL
    easy_model = EasyModel()
    train(easy_model, data_loader, num_epochs, learning_rate)
    y_true_easy, y_pred_easy = evaluate(easy_model, data_loader)
    print(f'Easy Model: '
          f'Final Train Accuracy: {100.* accuracy_score(y_true_easy, y_pred_easy):.4f},',
          f'Final Train F1 Score: {100.* f1_score(y_true_easy, y_pred_easy, average="weighted"):.4f}')
    plot_confusion_matrix(confusion_matrix(y_true_easy, y_pred_easy), class_names, 'Easy Model')

    # MEDIUM MODEL
    medium_model = MediumModel()
    train(medium_model, data_loader, num_epochs, learning_rate)
    y_true_medium, y_pred_medium = evaluate(medium_model, data_loader)
    print(f'Medium Model: '
          f'Final Train Accuracy: {100.* accuracy_score(y_true_medium, y_pred_medium):.4f},',
          f'Final F1 Score: {100.* f1_score(y_true_medium, y_pred_medium, average="weighted"):.4f}')
    plot_confusion_matrix(confusion_matrix(y_true_medium, y_pred_medium), class_names, 'Medium Model')

    # ADVANCED MODEL
    advanced_model = AdvancedModel()
    train(advanced_model, data_loader_reshaped, num_epochs, learning_rate)
    y_true_advanced, y_pred_advanced = evaluate(advanced_model, data_loader_reshaped)
    print(f'Advanced Model: '
          f'Final Train Accuracy: {100.* accuracy_score(y_true_advanced, y_pred_advanced):.4f},',
          f'Final F1 Score: {100.* f1_score(y_true_advanced, y_pred_advanced, average="weighted"):.4f}')
    plot_confusion_matrix(confusion_matrix(y_true_advanced, y_pred_advanced), class_names, 'Advanced Model')

############################################################
# Feedback
############################################################

feedback_question_1 = """
Coat and Shirt would be confused the most since they are very similar in appearance
"""

feedback_question_2 = """
Used a very similar architecture to that of the example but tweaked the linear function at the end and added another convolution layer
"""

feedback_question_3 = 8

feedback_question_4 = """
Trying to debug the nerual network.  Running into type errors.
"""

feedback_question_5 = """
I enjoyed the cnn functions the most.  The neural network seems cool but the difficulties with
type errors is very frustrating.
"""

if __name__ == '__main__':
    main()

# def main():

#     # X,Y = load_data('dataset.csv', True)
#     # print(X.shape)
#     # print(Y.shape)
#     class_names = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
#     X, Y = load_data('dataset.csv', False)
#     plt.imshow(X[0].reshape(28, 28), cmap='gray')
#     plt.title(class_names[Y[0]])
#     plt.show()

# main()