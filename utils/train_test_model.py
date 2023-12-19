from torch import no_grad, max
def train(model, dataloader, loss_fn, optimizer, epochs, device):
  for epoch in range(epochs):
    mini_batch_loss = 0
    for i, (X, y) in enumerate(dataloader, start=0):
      X, y = X.to(device), y.to(device)
      optimizer.zero_grad()

      pred = model(X)
      loss = loss_fn(pred, y)

      loss.backward()
      optimizer.step()

      mini_batch_loss += loss.item()
      if i % 4000 == 3999:
        #For every 2000 iterations, print out the 2000 iteration's average loss
        print(f"[{epoch+1}, {i+1:5d}] -------- Mini-batch Loss {mini_batch_loss/2000:>.4f}")
        mini_batch_loss = 0
  print("Training Finished")


def test(model, dataloader, device):
  correct, total = 0, len(dataloader.dataset)
  with no_grad():
    for (X, y) in dataloader:
      X, y = X.to(device), y.to(device)

      outputs = model(X)
      # outputs shares the same data with outputs.data, but the .data attribute skips autograd backpropagation computation graph
      # Since the larger logits would have the larger probability, you do not need to pass in softmax layer
      _, pred_index = max(outputs.data, dim=1) #reduce/collapse dimension 1 (4 rows of batch and 10 columns of classses)

      correct += (pred_index == y).sum().item() 

  print(f'Test Accuracy of size {total}: {(correct/total) * 100:.2f} %')