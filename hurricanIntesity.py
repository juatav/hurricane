import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Load Atlantic hurricane dataset
atlantic_url = "https://raw.githubusercontent.com/juatav/hurricane/main/atlantic.csv?token=GHSAT0AAAAAACK6Y2RIOJGM4LTSGCS3NWWKZLHSSOQ"
atlantic_data = pd.read_csv(atlantic_url)

# Debug to check columns and data after loading Atlantic dataset
print("Columns in Atlantic Data:")
print(atlantic_data.columns)
print("\nFirst few rows of Atlantic Data:")
print(atlantic_data.head())

# Load Pacific hurricane dataset
pacific_url = "https://raw.githubusercontent.com/juatav/hurricane/main/pacific.csv?token=GHSAT0AAAAAACK6Y2RIIYF2E6B45JV5LJGQZLHSTAQ"
pacific_data = pd.read_csv(pacific_url)

# Debug to check columns and data after loading Pacific dataset
print("\nColumns in Pacific Data:")
print(pacific_data.columns)
print("\nFirst few rows of Pacific Data:")
print(pacific_data.head())

# Concatenate the datasets (assuming they have similar structures)
combined_data = pd.concat([atlantic_data, pacific_data], ignore_index=True)


# Drop non-numeric columns
non_numeric_columns = ['ID', 'Name', 'Date', 'Time', 'Event', 'Latitude', 'Longitude']
combined_data = combined_data.drop(columns=non_numeric_columns)

# Convert categorical features to numerical values
le = LabelEncoder()
combined_data['Status'] = le.fit_transform(combined_data['Status'])

# Replace missing values (assuming -999 represents missing values)
combined_data.replace(-999, np.nan, inplace=True)

# Handle missing values
combined_data = combined_data.dropna()

# Separate features (X) and target variable (y)
y = combined_data['Maximum Wind']
X = combined_data.drop('Maximum Wind', axis=1)

# Standardize numeric features
numeric_columns = X.select_dtypes(include=[np.number]).columns
X[numeric_columns] = StandardScaler().fit_transform(X[numeric_columns])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# RNN Model Integration
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        # Forward pass of the RNN
        self.inputs = inputs
        self.h_prev = np.zeros((self.Wxh.shape[0], 1))

        self.h_next = np.tanh(np.dot(self.Wxh, inputs) + np.dot(self.Whh, self.h_prev) + self.bh)
        self.output = np.dot(self.Why, self.h_next) + self.by

        return self.output, self.h_next

    def backward(self, d_output, learning_rate=0.01):
        # Backward pass of the RNN
        d_Why = np.dot(d_output, self.h_next.T)
        d_by = d_output
        d_h_next = np.dot(self.Why.T, d_output)

        d_tanh = (1 - self.h_next ** 2) * d_h_next

        d_Wxh = np.dot(d_tanh, self.inputs.T)
        d_Whh = np.dot(d_tanh, self.h_prev.T)
        d_bh = d_tanh

        # Update weights and biases
        self.Wxh -= learning_rate * d_Wxh
        self.Whh -= learning_rate * d_Whh
        self.Why -= learning_rate * d_Why
        self.bh -= learning_rate * d_bh
        self.by -= learning_rate * d_by

    def train(self, X_train, y_train, epochs=10, learning_rate=0.01):
        for epoch in range(epochs):
            for i in range(len(X_train)):
                inputs = X_train.iloc[i].values.reshape(-1, 1)
                target = y_train.iloc[i].reshape(-1, 1)

                # Forward pass
                output, h_next = self.forward(inputs)

                # Compute loss (MSE)
                loss = 0.5 * np.sum((output - target) ** 2)

                # Backward pass
                d_output = output - target
                self.backward(d_output, learning_rate)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

        return loss  # Return the final loss

# Experiment details
experiment_number = 1
parameters = {
    "Model Type": "SimpleRNN",
    "Input Size": X_train.shape[1],
    "Hidden Size": 8,
    "Output Size": 1,
    "Epochs": 10,
    "Learning Rate": 0.01,
}

# Instantiate the RNN model
rnn_model = SimpleRNN(parameters["Input Size"], parameters["Hidden Size"], parameters["Output Size"])

# Train the RNN model and get the final loss
final_loss = rnn_model.train(X_train, y_train, epochs=parameters["Epochs"], learning_rate=parameters["Learning Rate"])

# User input for a specific month and ocean
user_ocean = input("Enter the ocean (Atlantic or Pacific): ")

# Filter data based on the user's selected ocean
ocean_data = atlantic_data if user_ocean.lower() == 'atlantic' else pacific_data

# Ask user for the month
user_month = input("Enter the month (e.g., 06 for June): ")

# Filter data based on user input and ocean
filtered_data = ocean_data[(ocean_data['Date'].astype(str).str[4:6] == user_month)]

# Print the filtered data
print("\nFiltered Data:")
print(filtered_data.head())

# Extract features and standardize
user_X = filtered_data.drop('Maximum Wind', axis=1)
user_X[numeric_columns] = StandardScaler().fit_transform(user_X[numeric_columns])

# Print the intermediate steps for debugging
print("\nFiltered Data (After Standardization):")
print(user_X.head())

# Make predictions using the trained RNN model
user_predictions = []
for i in range(len(filtered_data)):
    user_input_array = user_X.iloc[i].values.reshape(-1, 1)
    predicted_wind_speed, _ = rnn_model.forward(user_input_array)
    user_predictions.append(predicted_wind_speed[0][0])

# Display the predicted maximum wind speed in a graph
plt.figure(figsize=(10, 6))
plt.plot(filtered_data['ID'], user_predictions, label='Predicted Wind Speed')
plt.xlabel('Hurricane ID')
plt.ylabel('Maximum Wind Speed (Predicted)')
plt.title(f'Predicted Maximum Wind Speed for {user_month} in the {user_ocean} Ocean')
plt.legend()
plt.show()
