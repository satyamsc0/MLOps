import torch
import torch.nn as nn
from evaluate import evaluate
from model import ResnetX
from preprocess import preprocess_data, to_device
import matplotlib.pyplot as plt

# function to train the model
def train(model, train_dl, val_dl, epochs, max_lr, loss_func, optim):
  # initialize the optimizer
  optimizer = optim(model.parameters(), max_lr)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs*len(train_dl))

  results = []
  for epoch in range(epochs):
    model.train()
    train_losses = []
    lrs = []
    for images, labels in train_dl:
      logits = model(images)
      loss = loss_func(logits, labels)
      train_losses.append(loss)
      loss.backward() # delta_loss / delta_model_parameters
      optimizer.step()
      optimizer.zero_grad()
      lrs.append(optimizer.param_groups[0]["lr"])
      scheduler.step()
    epoch_train_loss = torch.stack(train_losses).mean().item()

    epoch_avg_loss, epoch_avg_acc = evaluate(model, val_dl, loss_func)

    results.append({'avg_valid_loss': epoch_avg_loss, 'avg_valid_acc': epoch_avg_acc, 'avg_train_loss': epoch_train_loss, 'lr':lrs})
  return results


# plot the results
def plot(results, pairs):
  fig, axes = plt.subplots(len(pairs), figsize=(10,10))
  for i, pair in enumerate(pairs):
    for title, graphs in pair.items():
      axes[i].set_title(title)
      for graph in graphs:
        axes[i].plot([result[graph] for result in results], label=graph)
      if i != len(pairs) - 1:  # For the last subplot only
        axes[i].legend()



if __name__ == "__main__":
    model = ResnetX(3, 10)
    
    train_dl, val_dl, test_dl, device = preprocess_data()
    # move to device
    model = to_device(model, device)
    epochs = 10
    max_lr = 1e-2
    loss_func = nn.functional.cross_entropy
    optim = torch.optim.Adam
    
    results = train(model, train_dl, val_dl, epochs, max_lr, loss_func, optim)
    
    # saving the model weights
    torch.save(model.state_dict(), "cifar10.ResnetX.pth")
    
    # results after each epoch
    for result in results:
        print(result["avg_valid_acc"])
    plot(results, [{"Accuracies vs epochs": ["avg_valid_acc"]}, {"Losses vs epochs": ["avg_valid_loss", "avg_train_loss"]}, {"Learning rates vs batches": ["lr"]}])
    plt.show()
    
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