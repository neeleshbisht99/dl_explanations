import torch

class Validate:
    # Validation function
    @staticmethod
    def validate(model, val_loader, criterion, device):
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long) # change
                
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels) 
                val_loss += loss.item() * inputs.size(0) 
                _, preds = torch.max(outputs, 1) 
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        return val_loss, val_acc