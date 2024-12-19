import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tinydb import TinyDB

# Load data
db = TinyDB(
    '/home/chen/Thesis_Project/app/database/restaurant_values_with_timestamp_db.json'
)
data = db.all()
df = pd.DataFrame(data)

# Convert 'timestamp' to datetime format and set it as the index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df = df.sort_index()
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek

# Resample data to a 30-minute interval and forward fill missing values
df_resampled = df.resample("30T").mean()
df_resampled["count"] = df_resampled["count"].fillna(method="ffill")

# Scale features
scaler_count = MinMaxScaler()
scaler_hour = MinMaxScaler()
scaler_dayofweek = MinMaxScaler()

df_resampled["count_scaled"] = scaler_count.fit_transform(
    df_resampled[["count"]])
df_resampled["hour_scaled"] = scaler_hour.fit_transform(
    df_resampled[["hour"]].values.reshape(-1, 1))
df_resampled["dayofweek_scaled"] = scaler_dayofweek.fit_transform(
    df_resampled[["dayofweek"]].values.reshape(-1, 1))


# Create sequences
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequence = data[i:i + seq_length]
        label = data[i + seq_length,
                     0]  # We use only the 'count' value as the label
        sequences.append(sequence)
        labels.append(label)
    return np.array(sequences), np.array(labels)


data = df_resampled[["count_scaled", "hour_scaled", "dayofweek_scaled"]].values
seq_length = 48
sequences, labels = create_sequences(data, seq_length)

# Split data into training and testing
train_size = int(len(sequences) * 0.8)
train_sequences = sequences[:train_size]
train_labels = labels[:train_size]
test_sequences = sequences[train_size:]
test_labels = labels[train_size:]


# Define the LSTM model
class PeopleFlowLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(PeopleFlowLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(
            out[:, -1, :])  # Decode the hidden state of the last time step
        return out


# Hyperparameters
input_size = 3
hidden_size = 32
num_layers = 2
output_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PeopleFlowLSTM(input_size, hidden_size, num_layers,
                       output_size).to(device)

# Convert to tensors and prepare DataLoader
train_sequences_tensor = torch.FloatTensor(train_sequences).to(device)
train_labels_tensor = torch.FloatTensor(train_labels).to(device).view(
    -1, 1)  # Ensure labels are (batch_size, 1)

train_dataset = TensorDataset(train_sequences_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

test_sequences_tensor = torch.FloatTensor(test_sequences).to(device)
test_labels_tensor = torch.FloatTensor(test_labels).to(device).view(-1, 1)

# Training loop
num_epochs = 500
loss_list = []
for epoch in range(num_epochs):
    model.train()
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)

        outputs = model(sequences)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        outputs = model(test_sequences_tensor)

        test_loss = criterion(outputs,
                              test_labels_tensor)  # Calculate the test loss
        # print(f"Test Loss: {test_loss.item()}")
    if (epoch + 1) % 10 == 0:
        loss_list.append(loss.item())
        print(
            f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, test_loss: {test_loss.item()}'
        )

# Preparing test data
test_sequences_tensor = torch.FloatTensor(test_sequences).to(device)
test_labels_tensor = torch.FloatTensor(test_labels).to(device).view(-1, 1)
# Making predictions
model.eval()
with torch.no_grad():
    outputs = model(test_sequences_tensor)
    predictions = outputs.cpu().numpy(
    )  # Convert to numpy array for inverse transformation
    test_loss = criterion(outputs,
                          test_labels_tensor)  # Calculate the test loss
    print(f"Test Loss: {test_loss.item()}")

# Inverse transform predictions and test labels to original scale
predictions = scaler_count.inverse_transform(predictions)
test_labels_original_scale = scaler_count.inverse_transform(
    test_labels.reshape(-1, 1))

# Plotting results
plt.plot(test_labels_original_scale, label="Actual")
plt.plot(predictions, label="Predicted")
plt.xlabel("Time Step")
plt.ylabel("People Count")
plt.legend()
plt.show()

# Plotting loss
plt.plot(loss_list, label="Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Load data
db = TinyDB('/home/chen/Thesis_Project/app/database/test.json')
data = db.all()
df = pd.DataFrame(data)

# Convert 'timestamp' to datetime format and set it as the index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df = df.sort_index()
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek

# Resample data to a 30-minute interval and forward fill missing values
df_resampled = df.resample("30T").mean()
df_resampled["count"] = df_resampled["count"].fillna(method="ffill")

# Scale features
scaler_count = MinMaxScaler()
scaler_hour = MinMaxScaler()
scaler_dayofweek = MinMaxScaler()

df_resampled["count_scaled"] = scaler_count.fit_transform(
    df_resampled[["count"]])
df_resampled["hour_scaled"] = scaler_hour.fit_transform(
    df_resampled[["hour"]].values.reshape(-1, 1))
df_resampled["dayofweek_scaled"] = scaler_dayofweek.fit_transform(
    df_resampled[["dayofweek"]].values.reshape(-1, 1))

data = df_resampled[["count_scaled", "hour_scaled", "dayofweek_scaled"]].values
seq_length = 48
sequences, labels = create_sequences(data, seq_length)

# Number of predictions needed for the next two days (48 intervals per day)
forecast_horizon = 96
sequences_tensor = torch.FloatTensor(sequences).to(device)

# Start with the last sequence for predictions
current_sequence = sequences_tensor[-1].cpu().tolist(
)  # Convert last test sequence to list for manipulation
future_predictions = []

model.eval()
with torch.no_grad():
    for _ in range(forecast_horizon):
        # Convert the current sequence to a tensor and move to device
        input_sequence = torch.FloatTensor(current_sequence).unsqueeze(0).to(
            device)

        # Make the next prediction
        next_prediction = model(input_sequence)
        next_prediction_value = next_prediction.cpu().item(
        )  # Convert tensor to scalar

        # Store the prediction
        future_predictions.append(next_prediction_value)

        # Update the current sequence: remove the oldest entry and add the new prediction
        # Here we add placeholders (e.g., 0 or 1) for `hour` and `dayofweek`
        current_sequence = current_sequence[1:] + [[
            next_prediction_value, 0, 0
        ]]  # Adjust placeholders as needed
# Inverse transform predictions and test labels to original scale
future_predictions = scaler_count.inverse_transform(
    np.array(future_predictions).reshape(-1, 1))
test_labels_original_scale = scaler_count.inverse_transform(
    labels.reshape(-1, 1))
print(future_predictions)
print(len(future_predictions))
print(test_labels_original_scale)
# Create an x-axis for future predictions that starts right after the actual test data
x_actual = range(len(test_labels_original_scale))
x_future = range(len(test_labels_original_scale),
                 len(test_labels_original_scale) + forecast_horizon)

# Plot the actual and predicted values with appropriate x-axis alignment
plt.plot(x_actual, test_labels_original_scale, label="Actual")
plt.plot(x_future, future_predictions, label="Predicted", color="orange")
plt.xlabel("Time Step")
plt.ylabel("People Count")
plt.legend()
plt.show()
