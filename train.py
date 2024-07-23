import torch

class Train:
    # Train the model
    @staticmethod
    def train(model, train_loader, optimizer, criterion, epoch, device):

        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1) 
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_correct / total
        return train_loss, train_acc