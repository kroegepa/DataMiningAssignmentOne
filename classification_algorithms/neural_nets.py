import torch
import torch.nn as nn
import torch.optim as optim


class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # gru layer (with dropout)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        # linear output layer
        self.fc = nn.Linear(hidden_size, num_classes)

        self.best_valid_loss = float('inf')

    def forward(self, x, h0=None):
        # Ensure the initial hidden state is on the same device as the inputs
        if h0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, hn = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

    def train_model(self, train_loader, num_epochs, learning_rate, device="cpu", test_loader=None):
        self.to(device)  # set model device
        criterion = nn.CrossEntropyLoss()  # good for multi-class-classification
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.train()  # set model to train mode
            total_loss = 0
            train_acc = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # set data device
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # calc acc
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels).sum().item()
                train_acc += correct / labels.size(0)

            avg_train_loss = total_loss / len(train_loader)
            avg_train_acc = train_acc / len(train_loader)

            if test_loader is not None:
                self.eval()  # set model to evaluation mode
                total_valid_loss = 0
                test_acc = 0
                with torch.no_grad():  # no updating the weights
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = self(inputs)
                        loss = criterion(outputs, labels)
                        total_valid_loss += loss.item()
                        # calc acc
                        _, predicted = torch.max(outputs, 1)
                        correct = (predicted == labels).sum().item()
                        test_acc += correct / labels.size(0)

                avg_valid_loss = total_valid_loss / len(test_loader)
                avg_test_acc = test_acc / len(test_loader)

                # TODO: save best model
                ''' 
                # Save the model if the validation loss has decreased
                best_valid_loss = float('inf')
                if avg_valid_loss < best_valid_loss:
                    best_valid_loss = avg_valid_loss
                    torch.save(self.state_dict(), 'best_model.pth')
                '''

                print(
                    f"Epoch {epoch + 1}/{num_epochs}:\n" +
                    f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}\n" +
                    f"Train Acc: {avg_train_acc * 100:.2f}%, Validation Acc: {avg_test_acc * 100:.2f}%")
            else:
                print(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}" +
                      f"Train Acc: {avg_train_acc * 100:.2f}%")
