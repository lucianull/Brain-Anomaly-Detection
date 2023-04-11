import torch
import torch.nn as nn
from math import ceil
from torchvision import transforms, datasets
import os
import cv2
import torch.optim as optim
import datetime
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from DataPreprocessing import Data

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None) -> None:
        super(ResNetBlock, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.BatchNorm1 = nn.BatchNorm2d(out_channels)
        self.Conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.BatchNorm2 = nn.BatchNorm2d(out_channels)
        self.ReLU_activation = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identityTensor = x.clone()      # retinem o clona a input-ului pentru a face convolutia de la final
        x = self.Conv1(x)
        x = self.BatchNorm1(x)
        x = self.ReLU_activation(x)
        x = self.Conv2(x)
        x = self.BatchNorm2(x)
        if self.downsample is not None:
            identityTensor = self.downsample(identityTensor)        # in cazul in care output-ul are dimensiuni mai mici, redimensionam si inputul de la inceputul functiei
        x += identityTensor     # aplicam convolutia, care in cazul nostru reprezinta adunarea fiecarui element cu corespondentul sau
        x = self.ReLU_activation(x)
        return x
    
class ResNet18(nn.Module):
    def __init__(self) -> None:
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.BatchNorm1 = nn.BatchNorm2d(self.in_channels)
        self.ReLU_activation = nn.ReLU(inplace=True)
        self.MaxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.ResidualLayers = nn.Sequential(
            self.makeResNetLayer(64),
            self.makeResNetLayer(128, stride=2),
            self.makeResNetLayer(256, stride=2),
            self.makeResNetLayer(512, stride=2),
        )
        self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.Classifier = nn.Sequential(
            nn.Linear(512, 2),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def makeResNetLayer(self, out_channels, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(ResNetBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        layers.append(ResNetBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.Conv1(x)
        x = self.BatchNorm1(x)
        x = self.ReLU_activation(x)
        x = self.MaxPool(x)

        x = self.ResidualLayers(x)

        x = self.AvgPool(x)
        x = torch.flatten(x, 1)
        x = self.Classifier(x)
        return x
    

predicted_image_index = 17001
outFile = open('submissions.csv', 'w')
outFile.write('id,class\n')

def print_predictions(predicted_labels):
    global predicted_image_index
    for x in predicted_labels:
        outFile.write(f'{predicted_image_index:06},{int(x[0])}\n')
        predicted_image_index += 1


data_dir = 'data/data_for_cnn/train_data'
predict_dir = 'data/data_for_cnn/test_data'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 25
NAME = 'ResNet18-Architecture-' + str(epochs) + '-'+ datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if __name__ == "__main__":
    transform = transforms.Compose([        # aplicam diferite augmentari pe datele de antrenare/validare
        transforms.RandomCrop(224),
        transforms.RandomRotation(degrees=35),
        transforms.RandomVerticalFlip(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    transformPredict = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transformPredict)
    train_size = int(0.88 * len(dataset))   #proportia aproximativa intre datele de antrenare si cele de validare
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])   #despartim setul de date conform proportiei de mai sus
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
    predict_dataset = datasets.ImageFolder(root=predict_dir, transform=transformPredict)
    data_to_predict = torch.utils.data.DataLoader(predict_dataset, batch_size=64, shuffle=False)
    model = ResNet18().to(device)       # cream modelul si il adaugam pe GPU
    best_f1_score = 0.0
    class_weights = torch.tensor([3.38]).to(device)    # folosim weight-ul obtinut anterior si il hard-codam pentru a face procesul mai rapid, reprezinta weight-ul clasei 1
    loss_function = nn.BCEWithLogitsLoss(pos_weight=class_weights)      # initializam ca loss function Binary Cross Entropy si setam weight-ul clasei 1
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
    
    train_loss_values = []
    train_f1_scores = []
    validation_f1_scores = []
    
    for epoch in range(1, epochs + 1):
        outputs = None
        labels = None
        total = 0
        correct = 0
        tp = 0
        fp = 0
        fn = 0
        avg_loss = 0.0
        f1_score_running = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(1)
            labels = labels.float()
            outputs = model(inputs)
            mask = outputs >= 0.5       # cream o masca prin care sa retinem valorile de 1, intrucat functia de activare de la ultimul strat este sigmoida, putem obtine orice valoare intre [0, 1]
            new_outputs = torch.zeros_like(outputs)
            new_outputs[mask] = 1.0     # retinem valorile prezise folosind masca
            total += labels.size(0)
            correct += (new_outputs == labels).sum().item()
            tp += ((new_outputs == 1) & (labels == 1)).sum().item()
            fp += ((new_outputs == 1) & (labels == 0)).sum().item()
            fn += ((new_outputs == 0) & (labels == 1)).sum().item()
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            try:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1_score_running = 2 * (precision * recall) / (precision + recall)
            except ZeroDivisionError:
                f1_score_running = 0
            avg_loss += loss.item()
            print(f' Progress Training: epoch: {epoch}/{epochs} batch: {i + 1}/{len(train_loader)} accuracy: {(correct/total):.4f} f1_score: {(f1_score_running):.4f} loss_value: {(avg_loss/(i+1)):.4f} learning rate: {optimizer.param_groups[0]["lr"]:.6f}', end='\r')
        scheduler.step(loss)
        print()
        correct = 0
        total = 0
        train_f1_scores.append(f1_score_running)
        train_loss_values.append(avg_loss)
        with torch.no_grad():
            outputs = None
            labels = None
            tp = 0
            fp = 0
            fn = 0
            for i, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.unsqueeze(1)
                labels = labels.float()
                outputs = model(inputs)
                mask = outputs >= 0.5
                new_outputs = torch.zeros_like(outputs)
                new_outputs[mask] = 1.0
                tp += ((new_outputs == 1) & (labels == 1)).sum().item()
                fp += ((new_outputs == 1) & (labels == 0)).sum().item()
                fn += ((new_outputs == 0) & (labels == 1)).sum().item()
                total += labels.size(0)
                correct += (new_outputs == labels).sum().item()
                try:
                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    f1_score_running = 2 * (precision * recall) / (precision + recall)
                    if f1_score_running > best_f1_score:
                        best_f1_score = f1_score_running
                        torch.save(model.state_dict(), 'Models/ModelParameters/' + NAME + '.pt')
                except ZeroDivisionError:
                    f1_score_running = 0
                print(f' Progress Testing: epoch: {epoch}/{epochs} batch: {i + 1}/{len(test_loader)} accuracy: {(correct/total):.4f} f1_score: {(f1_score_running):.4f}', end='\r')
            print()
        validation_f1_scores.append(f1_score_running)
        
            
    model.load_state_dict(torch.load('Models/ModelParameters/' + NAME + '.pt'))     #incarcam cel mai bun model pentru a face predictiile pe datele de testare cu el
    data_loader = Data()
    data_loader.OpenFile()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_to_predict):
            inputs = inputs.to(device)
            outputs = model(inputs)
            mask = outputs >= 0.5
            new_outputs = torch.zeros_like(outputs)
            new_outputs[mask] = 1
            new_outputs[~mask] = 0
            data_loader.PrintData(new_outputs)
            print(f' Progress Predicting: batch: {i + 1}/{len(data_to_predict)}', end='\r')
        print()
    data_loader.CloseFile()