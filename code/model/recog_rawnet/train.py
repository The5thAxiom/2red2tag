# train.py
from model import RawNet
from dataset import VoiceDataset
import torch
import yaml
import os
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Load the model configuration
with open('model_config_RawNet.yaml') as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = RawNet(d_args=config['model'], device=device)
model.to(device)

# Load dataset paths
ai_folder = "processed_ai"  # Update with the correct path
human_folder = "processed_human"  # Update with the correct path


# Create dataset
full_dataset = VoiceDataset(ai_folder=ai_folder, human_folder=human_folder)

# Splitting the dataset into training, validation, and testing
train_size = int(0.7 * len(full_dataset))
val_test_size = len(full_dataset) - train_size
val_size = int(0.5 * val_test_size)
test_size = val_test_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training and optimization setup
optimizer = torch.optim.Adam(model.parameters(), amsgrad=bool(config['amsgrad']))
criterion = torch.nn.CrossEntropyLoss()

# Initialize variables for early stopping
best_loss = np.inf
patience = 5
patience_counter = 0

# Training loop
num_epochs = 100  # Start with a high number; early stopping will determine the actual epochs
for epoch in range(num_epochs):
    model.train()
    train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Training]')
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_bar.set_postfix(loss=loss.item())
        
    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')
    
    # Early stopping logic
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

# Load the best model for evaluation/testing
model.load_state_dict(torch.load('best_model.pth'))

# Evaluation
model.eval()  # Set model to evaluation mode
correct_predictions = 0
total_predictions = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs, _ = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

accuracy = correct_predictions / total_predictions
print(f'Evaluation Accuracy: {accuracy:.4f}')

# Calculate the confusion matrix
conf_matrix = confusion_matrix(all_labels, all_predictions)

# Visualization of the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['AI', 'Human'], yticklabels=['AI', 'Human'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()




# # train.py
# from model import RawNet
# from dataset import VoiceDataset
# import torch
# import yaml
# import os
# import numpy as np
# from torch.utils.data import DataLoader, random_split
# from tqdm import tqdm
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns  # Optional for a nicer confusion matrix visualization


# best_loss = np.inf
# patience_counter = 0
# patience = 5  


# # Load the model configuration
# with open('model_config_RawNet.yaml') as f:
#     config = yaml.safe_load(f)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("cuda")
# print(torch.cuda.is_available())


# # Initialize the model
# model = RawNet(d_args=config['model'], device=device)
# model.to(device)

# # Load dataset paths
# ai_folder = "processed_ai"  # Update with the correct path
# human_folder = "processed_human"  # Update with the correct path

# # Create dataset
# full_dataset = VoiceDataset(ai_folder=ai_folder, human_folder=human_folder)

# # Splitting the dataset into training and testing
# train_size = int(0.8 * len(full_dataset))
# test_size = len(full_dataset) - train_size
# train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # Training and optimization setup
# optimizer = torch.optim.Adam(model.parameters(), amsgrad=bool(config['amsgrad']))
# criterion = torch.nn.CrossEntropyLoss()


# # Training loop
# num_epochs = 6  # Specify the number of epochs
# for epoch in range(num_epochs):
#     model.train()  # Set model to training mode
#     train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Training]')

#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         #train_bar.set_postfix(loss=loss.item())
        
#         # Forward pass
#         outputs, _ = model(inputs)
#         loss = criterion(outputs, labels)
        
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         train_bar.set_postfix(loss=loss.item())    
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")



# # Evaluation
# model.eval()  # Set model to evaluation mode
# correct_predictions = 0
# total_predictions = 0
# all_predictions = []
# all_labels = []
# test_loss = 0

# with torch.no_grad():
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
        
#         # Forward pass
#         outputs, _ = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
        
#         test_loss += criterion(outputs, labels).item()
#         # Update correct predictions count
#         correct_predictions += (predicted == labels).sum().item()
#         total_predictions += labels.size(0)

#         # Collect all labels and predictions
#         all_labels.extend(labels.cpu().numpy())
#         all_predictions.extend(predicted.cpu().numpy())

#         test_loss /= total_predictions
#         print(f'Test Loss: {test_loss:.4f}')

#         if test_loss < best_loss:
#             best_loss = test_loss
#             patience_counter = 0
#         else:
#             patience_counter += 1

#         if patience_counter >= patience:
#             print("Early stopping triggered.")
#             break

# accuracy = correct_predictions / total_predictions
# print(f'Evaluation Accuracy: {accuracy:.4f}')


# # Calculate the confusion matrix
# conf_matrix = confusion_matrix(all_labels, all_predictions)

# # Visualization of the confusion matrix
# plt.figure(figsize=(10, 7))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['AI', 'Human'], yticklabels=['AI', 'Human'])
# plt.title('Confusion Matrix')
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')
# plt.show()


