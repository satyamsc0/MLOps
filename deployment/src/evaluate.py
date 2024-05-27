
import torch
import torch.nn as nn
from model import ResnetX
from preprocess import preprocess_data, to_device

def accuracy(logits, labels):
  pred, predClassId = torch.max(logits, dim=1) # BxN
  return torch.tensor(torch.sum(predClassId == labels).item() / len(logits))

# function for model evaluation
def evaluate(model, dl, loss_func):
  model.eval()
  batch_losses, batch_accs = [], []
  for images, labels in dl:
    with torch.no_grad():
      logits = model(images)
    batch_losses.append(loss_func(logits, labels))
    batch_accs.append(accuracy(logits, labels))
  epoch_avg_loss = torch.stack(batch_losses).mean().item()
  epoch_avg_acc = torch.stack(batch_accs).mean()
  return epoch_avg_loss, epoch_avg_acc

if __name__ == "__main__":
    model = ResnetX(3, 10)
    train_dl, val_dl, test_dl, device = preprocess_data()

    # move to device
    model = to_device(model, device)
    loss_func = nn.functional.cross_entropy
    
    _, test_acc = evaluate(model, test_dl, loss_func)
    print(test_acc)

    torch.save(model.state_dict(), "cifar10.ResnetX.pth")   # saving the model weights
    model2 = ResnetX(3, 10)
    model2 = to_device(model2, device)
    _, test_acc = evaluate(model2, test_dl, loss_func)
    print(test_acc)

    model2.load_state_dict(torch.load("cifar10.ResnetX.pth"))   # loading the model weights for test/inference
    _, test_acc = evaluate(model2, test_dl, loss_func)
    print(test_acc)