import torch 
import torch.nn as nn

class Trainer:
    def __init__(self, model, 
                train_dataset, 
                valid_dataset, 
                test_dataset, 
                learning_rate,
                batch_size,
                n_epochs):
        # Param init
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        # Train config 
        self.batch_size = batch_size
        self.num_epochs = n_epochs
        self.optimizer = nn.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        

    def train(self):
        train_loader = nn.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = nn.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0

            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(self.train_dataset)

            self.model.eval()
            val_loss = 0.0
            val_corrects = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_corrects += torch.sum(preds == labels.data)

                val_loss /= len(self.val_dataset)
                val_acc = val_corrects.double() / len(self.val_dataset)

            print('Epoch {}/{}: Train Loss: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'.format(epoch+1, self.num_epochs, train_loss, val_loss, val_acc))