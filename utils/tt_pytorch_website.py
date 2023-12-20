from torch import no_grad, max
def train(model, dataloader, loss_fn, optimizer, epochs, device, print_logs=False):
  loss_list, accuracy_list = [], []
  model.train()
  for epoch in range(epochs):
    mini_batch_loss, total_loss, correct, size = 0, 0, 0, len(dataloader.dataset)
    for i, (X, y) in enumerate(dataloader, start=0):
      X, y = X.to(device), y.to(device)

      optimizer.zero_grad()

      outputs = model(X)
      loss = loss_fn(outputs, y)

      loss.backward()
      optimizer.step()

      _, pred_index = max(outputs.data, dim=1)

      correct += (pred_index == y).sum().item()
      total_loss += loss.item()
      mini_batch_loss += loss.item()

      if print_logs == True:
        print("y", y)
        print("outputs", outputs)
        print("pred index", pred_index)
      
      if i % 4000 == 3999:
        #For every 2000 iterations, print out the 2000 iteration's average loss
        print(f"[{epoch+1}, {i+1:5d}] -------- Mini-batch Loss {mini_batch_loss/2000:>.4f}")
        mini_batch_loss = 0

    loss_list.append(total_loss / size)
    accuracy_list.append(correct / size)

  print("Training Finished")
  return loss_list, accuracy_list


def test(model, dataloader, device, print_logs=False):
  correct, total = 0, len(dataloader.dataset)
  model.eval()
  with no_grad():
    for (X, y) in dataloader:
      X, y = X.to(device), y.to(device)

      outputs = model(X)
      """
          - outputs shares the same data with outputs.data, but the .data attribute skips autograd backpropagation computation graph
          - Since the larger logits would have the larger probability, you do not need to pass in softmax layer
          - reduce/collapse dimension 1 (4 rows of batch and 10 columns of classses)
      """
      
      _, pred_index = max(outputs.data, dim=1)
      if print_logs == True:
        print("y", y)
        print("outputs", outputs)
        print("pred index", pred_index)

      correct += (pred_index == y).sum().item() 

  print(f'Test Accuracy of size {total}: {(correct/total) * 100:.2f} %')
  