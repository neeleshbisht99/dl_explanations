import torch
import torch.nn.functional as F 

class Inference:
    # Inference function
    @staticmethod
    def inference(model, test_loader, device):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)
                
                outputs, _ = model(inputs)

                softmaxed_prediction = F.softmax(outputs, dim=1) 
                scores = softmaxed_prediction.squeeze().cpu().numpy()
                
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        test_acc = correct / total
        return test_acc