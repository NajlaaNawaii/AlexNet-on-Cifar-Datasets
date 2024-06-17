import torch # type: ignore
import numpy as np # type: ignore
import torch # type: ignore
from torch.utils.data.sampler import SubsetRandomSampler # type: ignore
from torchvision import datasets, transforms # type: ignore
import torch.nn as nn # type: ignore

class CIFAR10Trainer:
    def __init__(self, batch_size, augment, mixup_alpha=0.4, num_epochs=30):
        self.batch_size = batch_size
        self.augment = augment
        self.mixup_alpha = mixup_alpha
        self.num_epochs = num_epochs
        
        self.device = torch.device("mps")

    def mixup_augmentation(self, x, y, alpha):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        idx = torch.randperm(batch_size).to(self.device)
        mixed_x = lam * x + (1 - lam) * x[idx, :]
        y_a, y_b = y, y[idx]

        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, y_a, y_b, lam):
        return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def train(self, model, train_loader, optimizer, criterion, scheduler=None):
        model.to(self.device)
        model.train()
        
        ema_train_loss = None
        train_loss = []
        print("------Training--------")
        for epoch in range(self.num_epochs):
            
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                inputs, targets_a, targets_b, lam = self.mixup_augmentation(images, labels, self.mixup_alpha)
                outputs = model(inputs)
                loss_func = self.mixup_criterion(targets_a, targets_b, lam)
                loss = loss_func(criterion, outputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if ema_train_loss is None:
                    ema_train_loss = loss.item()
                else:
                    ema_train_loss = 0.9 * ema_train_loss + 0.1 * loss.item()
                    
                if i % 100 == 0:
                    train_loss.append(ema_train_loss)
                    print(f"Epoch: {epoch+1}/{self.num_epochs}, Step: {i+1}/{len(train_loader)}, Loss: {ema_train_loss:.4f}")
            
            if scheduler is not None:
                scheduler.step()
        
        return train_loss
    
    def validate(self, model, valid_loader, criterion):
        model.to(self.device)
        model.eval()
        
        valid_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        valid_loss /= len(valid_loader)
        
        print(f"Validation Loss: {valid_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return valid_loss, accuracy
    
    def test(self, model, test_loader, criterion):
        model.to(self.device)
        model.eval()
        
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        test_loss /= len(test_loader)
        
        print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return test_loss, accuracy