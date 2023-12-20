from torch import no_grad, max
def train(model, dataloader, loss_fn, optimizer, device):
  size = len(dataloader.dataset)
  mini_batch_loss, total_loss, correct = 0, 0, 0

  model.train()
  for i, (X, y) in enumerate(dataloader, start=0):
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()

    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()

    mini_batch_loss += loss.item()
    total_loss += loss.item()
    _, pred_index = max(pred, dim=1)
    correct += (pred_index == y).sum().item()

    if i % 4000 == 3999:
      loss, current = loss.item(), (i+1)*len(X)
      print(f"Loss: {loss:7f} [{current:>5d}/{size:>5d}]")

  train_loss = total_loss / size
  train_accuracy = correct / size
  return train_loss, train_accuracy

def valid(model, dataloader, loss_fn, device):
  size = len(dataloader.dataset)
  mini_batch_loss, total_loss, correct = 0, 0, 0

  model.eval()
  with no_grad():
    for (X, y) in dataloader:
      X, y = X.to(device), y.to(device)

      pred = model(X)
      loss = loss_fn(pred, y)

      mini_batch_loss += loss.item()
      total_loss += loss.item()
      _, pred_index = max(pred, dim=1)
      correct += (pred_index == y).sum().item()

  valid_loss = total_loss / size
  valid_accuracy = correct / size

  print(f"Validation Error ----- Loss: {valid_loss:.8f} Accuracy: {valid_accuracy * 100:.2f}")
  return valid_loss, valid_accuracy