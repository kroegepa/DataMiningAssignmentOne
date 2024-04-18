import torch
import torch.nn as nn
import torch.optim as optim


class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # gru layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # linear output layer
        self.fc = nn.Linear(hidden_size, num_classes)

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
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # set data device
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if test_loader is not None:
                self.eval()  # set model to evaluation mode
                total_valid_loss = 0
                with torch.no_grad():  # no updating the weights
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = self(inputs)
                        loss = criterion(outputs, labels)
                        total_valid_loss += loss.item()

                avg_train_loss = total_loss / len(train_loader)
                avg_valid_loss = total_valid_loss / len(test_loader)

                # TODO: save best model
                ''' 
                # Save the model if the validation loss has decreased
                best_valid_loss = float('inf')
                if avg_valid_loss < best_valid_loss:
                    best_valid_loss = avg_valid_loss
                    torch.save(self.state_dict(), 'best_model.pth')
                '''

                print(
                    f'Epoch {epoch + 1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}')
            else:
                avg_train_loss = total_loss / len(train_loader)
                print(f'Epoch {epoch + 1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}')
