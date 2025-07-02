from google.colab import files
uploaded = files.upload()

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

#Load the dataset
train_data = pd.read_csv("/content/machine-1-1.txt")
# Preview the dataset
print(train_data.head())

# Normalize the data
scaler = MinMaxScaler()
train_data_normalized = scaler.fit_transform(train_data)

# Create sequences for training (sequence_length = 50)
sequence_length = 50
X_train, y_train = [], []

for i in range(sequence_length, len(train_data_normalized)):
    X_train.append(train_data_normalized[i-sequence_length:i])
    y_train.append(train_data_normalized[i])

X_train, y_train = np.array(X_train), np.array(y_train)

# Load the testing data
test_data = pd.read_csv("/content/test-machine-1-1.txt")
test_data_normalized = scaler.transform(test_data)

# Create sequences for testing
X_test, y_test = [], []

for i in range(sequence_length, len(test_data_normalized)):
    X_test.append(test_data_normalized[i-sequence_length:i])
    y_test.append(test_data_normalized[i])

X_test, y_test = np.array(X_test), np.array(y_test)

# Build the LSTM model
model = Sequential([
    LSTM(64, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, activation='tanh', return_sequences=False),
    Dropout(0.2),
    Dense(X_train.shape[2])  # Output layer matching the number of features
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, shuffle=True)

# Plot the training loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Predict on the test data
y_pred = model.predict(X_test)

# Calculate reconstruction error
reconstruction_error=np.mean(np.abs(y_pred - y_test), axis=1)

# Plot reconstruction error
plt.figure(figsize=(15, 5))
plt.plot(reconstruction_error, label="Prediction Error")
plt.title("Prediction Error Over Time")
plt.legend()
plt.show()

print(reconstruction_error)

df =np.abs(y_pred - y_test)
plt.figure(figsize=(15, 5))
plt.plot(df[:,1])
plt.ylim(0,4)

# Generating error matrix
error=np.zeros((len(train_data_normalized), 38))
for i in range (50,len(train_data_normalized)):
  for j in range(38):
    error[i][j]=np.abs(y_pred[i-50][j] - y_test[i-50][j])

#Load the dataset
test_label = pd.read_csv("/content/test_label.txt")
# Preview the dataset
print(test_label.head(5))

flattened_list = test_label.values.flatten().tolist()
print(len(flattened_list))

# Function to extract index ranges where value is 1
def find_index_ranges(data):
    ranges = []
    start = None  # Start of a range

    for i in range(1, len(data)):
        # Detect the start of a range
        if data[i] == 1 and data[i - 1] == 0:
            start = i
        # Detect the end of a range
        elif data[i] == 0 and data[i - 1] == 1:
            if start == i - 1:



                # Single index range
                ranges.append((start, start))
            else:
                # Multi-index range (make the end inclusive by subtracting 1)
                ranges.append((start, i ))
            start = None

    # Handle the case where the dataset ends with a range of 1's
    if data[-1] == 1:
        if start == len(data) - 1:
            ranges.append((start, start))
        else:
            ranges.append((start, len(data) ))  # Make the end inclusive

    return ranges


actual_ranges = find_index_ranges(flattened_list)

actual_ranges

def calculate_detailed_metrics(predicted_ranges, actual_ranges):
    tp_count = 0
    fp_count = 0
    fn_count = 0

    # Convert ranges into sets of indices for easy comparison
    predicted_sets = [set(range(start, end + 1)) for start, end in predicted_ranges]
    actual_sets = [set(range(start, end + 1)) for start, end in actual_ranges]

    matched_predicted = set()  # Keep track of matched predicted ranges
    matched_actual = set()     # Keep track of matched actual ranges

    # Calculate True Positives and False Negatives
    for i, actual in enumerate(actual_sets):
        tp_found = False
        for j, predicted in enumerate(predicted_sets):
            if actual & predicted:  # Overlap exists
                tp_found = True
                matched_predicted.add(j)  # Mark predicted range as matched
                matched_actual.add(i)     # Mark actual range as matched
        if tp_found:
            tp_count += 1
        else:
            fn_count += 1  # No overlap found for this actual range

    # Calculate False Positives
    for j, predicted in enumerate(predicted_sets):
        if j not in matched_predicted:
            fp_count += 1

    return tp_count, fp_count, fn_count

act=np.zeros((len(flattened_list), 38)).astype(int)

feature_list = []  # Stores lists of feature indices for each range
set1 = []  # Stores extracted index ranges

# Read the dataset from a text file
interpretation_label_file = "/content/interpretation_label.txt"
with open(interpretation_label_file, "r") as file:
    for line in file:
        range_part, indices_part = line.strip().split(":")  # Split into range and indices
        start, end = map(int, range_part.split("-"))  # Extract range values
        indices = sorted(map(int, indices_part.split(",")))  # Extract and sort feature indices

        set1.append((start, end))  # Store the range (start, end)
        feature_list.append(indices)  # Store indices as a list

# Output the extracted data
for i in range(len(set1)):
    print(f"Range {set1[i]} -> Features: {feature_list[i]}")

len(set1)

# Function to find overlapping predicted ranges for an actual range
def find_overlapping_ranges(predicted, actual_range, max_index):
    start_actual, end_actual = actual_range
    overlaps = []
    for start_pred, end_pred in predicted:
        # Ensure end_pred doesn't exceed max_index
        end_pred = min(end_pred, max_index)
        if max(start_pred, start_actual) <= min(end_pred, end_actual):
            overlaps.append((start_pred, end_pred))
    return overlaps

# Function to compute mean for overlapping ranges
def compute_means(predicted_ranges, actual_ranges, data):
    means = {}
    max_index = data.shape[0] - 1  # Get the maximum valid index
    for actual_range in actual_ranges:
        overlapping_ranges = find_overlapping_ranges(predicted_ranges, actual_range, max_index) #To get the overlapping ranges

        # Get all row indices from overlapping predicted ranges
        row_indices = []
        for start, end in overlapping_ranges:
            # Ensure end doesn't exceed max_index
            end = min(end, max_index)
            row_indices.extend(range(start, end + 1)) #row indices are all indices that overlap actual ranges

        # Compute column-wise mean for the rows in `row_indices`
        if row_indices:
            rows_data = data[row_indices, :] # Extract rows from data
            means[actual_range] = np.mean(rows_data, axis=0) # Column-wise mean
        else:
            means[actual_range] = np.zeros(38) # No overlap found
    return means

best_threshold_timesteps = 0
best_precision_timesteps = 0
best_recall_timesteps = 0
best_f1_timesteps = 0
best_threshold = 0
best_precision = 0
best_recall = 0
best_f1 = 0
threshold = 0.1

while threshold < 5.1:
  threshold = round(threshold,2)

  #Write code
  anomalies = reconstruction_error > threshold
  anomalous_timesteps = 50 + np.where(anomalies)[0]   # Staring 50 sequence length is unpredicted

  # Store anomalous timesteps as an array
  ad = np.array(anomalous_timesteps)

  #print(f"Anomalous Timesteps: {ad}")

  #Bit Matrix generation based on threshold
  predicted_array =np.zeros((len(train_data_normalized), 38))
  for i in range(len(error)):
    for j in range(38):
      if error[i][j] > threshold :
        predicted_array[i][j] = 1

  Predicted=[]
  flag=0
  for i in range(len(predicted_array)):
    for j in ad:
      if i==j:
        flag=1
        break
    if flag==1:
      Predicted.append(1)
    else:
      Predicted.append(0)
    flag=0

  predicted_ranges = find_index_ranges(Predicted)

  # Calculating performance metrices for finding anomalous timesteps

  true_positives, false_positives, false_negatives = calculate_detailed_metrics(predicted_ranges, actual_ranges)

  Precision = (true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else 0
  Recall = (true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 0
  F1_score = (2 * Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0

  if F1_score > best_f1_timesteps:
    best_f1_timesteps = F1_score
    best_precision_timesteps = Precision
    best_recall_timesteps = Recall
    best_threshold_timesteps = threshold

  # Compute column-wise means

  updated_predicted_range = [(start - 50, end - 50) for start, end in predicted_ranges]

  updated_actual_range = [(start - 50, end - 50) for start, end in set1]

  error_matrix = error[50:]

  # Compute column-wise means


  means = compute_means(updated_predicted_range, updated_actual_range, error_matrix)

  dom = [value for value in means.values()]
  dom = np.array(dom)

  predicted_feature = [[i+1 for i, val in enumerate(row) if val > threshold] for row in dom]

  total_TP = 0
  total_FP = 0
  total_FN = 0

  # Compute TP, FP, FN for all rows and sum up
  for predicted, actual in zip(predicted_feature, feature_list):
    predicted_set = set(predicted)
    actual_set = set(actual)

    TP = len(predicted_set & actual_set)  # True Positives
    FP = len(predicted_set - actual_set)  # False Positives
    FN = len(actual_set - predicted_set)  # False Negatives

    total_TP += TP
    total_FP += FP
    total_FN += FN

  # Compute precision, recall, and F1-score
  precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
  recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
  f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


  if f1 > best_f1:
    best_f1 = f1
    best_precision = precision
    best_recall = recall
    best_threshold = threshold

  threshold += 0.1

print(f"Best Threshold:", best_threshold_timesteps)
print(f"Precision: {best_precision_timesteps}")
print(f"Recall: {best_recall_timesteps}")
print(f"F1 Score: {best_f1_timesteps}")

# Display metrics
print(f"Best Threshold:", best_threshold)
print(f"Precision: {best_precision}")
print(f"Recall: {best_recall}")
print(f"F1 Score: {best_f1}")

best_threshold_timesteps = 0
best_precision_timesteps = 0
best_recall_timesteps = 0
best_f1_timesteps = 0
best_threshold = 0
best_precision = 0
best_recall = 0
best_f1 = 0
threshold = 0.1

#50% match logic
while threshold < 5.1:
  threshold = round(threshold,2)

  anomalies = reconstruction_error > threshold
  anomalous_timesteps = 50 + np.where(anomalies)[0]   # Staring 50 sequence length is unpredicted

  # Store anomalous timesteps as an array
  ad = np.array(anomalous_timesteps)

  #print(f"Anomalous Timesteps: {ad}")

  #Bit Matrix generation based on threshold
  predicted_array =np.zeros((len(train_data_normalized), 38))
  for i in range(len(error)):
    for j in range(38):
      if error[i][j] > threshold :
        predicted_array[i][j] = 1

  Predicted=[]
  flag=0
  for i in range(len(predicted_array)):
    for j in ad:
      if i==j:
        flag=1
        break
    if flag==1:
      Predicted.append(1)
    else:
      Predicted.append(0)
    flag=0

  predicted_ranges = find_index_ranges(Predicted)

  # Calculating performance metrices for finding anomalous timesteps

  true_positives, false_positives, false_negatives = calculate_detailed_metrics(predicted_ranges, actual_ranges)

  Precision = (true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else 0
  Recall = (true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 0
  F1_score = (2 * Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0

  if F1_score > best_f1_timesteps:
    best_f1_timesteps = F1_score
    best_precision_timesteps = Precision
    best_recall_timesteps = Recall
    best_threshold_timesteps = threshold

  # Compute column-wise means

  updated_predicted_range = [(start - 50, end - 50) for start, end in predicted_ranges]

  updated_actual_range = [(start - 50, end - 50) for start, end in set1]

  error_matrix = error[50:]

  # Compute column-wise means


  means = compute_means(updated_predicted_range, updated_actual_range, error_matrix)

  dom = [value for value in means.values()]
  dom = np.array(dom)

  predicted_feature = [[i+1 for i, val in enumerate(row) if val > threshold] for row in dom]

  total_TP = 0
  total_FP = 0
  total_FN = 0

  # Compute TP, FP, FN using threshold-based logic
  for predicted, actual in zip(predicted_feature, feature_list):
    predicted_set = set(predicted)
    actual_set = set(actual)

    matched = len(predicted_set & actual_set)

    # Get the length of binary vector after padding
    max_index = max(predicted_set.union(actual_set))

    # Create binary vectors of length = max_index
    binary_predicted = [0] * max_index
    binary_actual = [0] * max_index

    for idx in predicted_set:
        binary_predicted[idx - 1] = 1  # 1-based to 0-based index

    for idx in actual_set:
        binary_actual[idx - 1] = 1

    if matched >= len(actual_set)/2:
      for i in range(len(binary_actual)):
        if binary_actual[i]==1:
          binary_predicted[i]=1

    TP, FP, FN = 0, 0, 0
    for a, p in zip(binary_actual, binary_predicted):
      if a == 1 and p == 1:
        TP += 1
      elif a == 0 and p == 1:
        FP += 1
      elif a == 1 and p == 0:
        FN += 1

    total_TP += TP
    total_FP += FP
    total_FN += FN

  # Compute precision, recall, and F1-score
  precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
  recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
  f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


  if f1 > best_f1:
    best_f1 = f1
    best_precision = precision
    best_recall = recall
    best_threshold = threshold

  threshold += 0.1

print(f"Best Threshold:", best_threshold_timesteps)
print(f"Precision: {best_precision_timesteps}")
print(f"Recall: {best_recall_timesteps}")
print(f"F1 Score: {best_f1_timesteps}")

# Display metrics
print(f"Best Threshold:", best_threshold)
print(f"Precision: {best_precision}")
print(f"Recall: {best_recall}")
print(f"F1 Score: {best_f1}")

best_threshold_timesteps = 0
best_precision_timesteps = 0
best_recall_timesteps = 0
best_f1_timesteps = 0
best_threshold = 0
best_precision = 0
best_recall = 0
best_f1 = 0
threshold = 0.1

#1 match logic
while threshold < 5.1:
  threshold = round(threshold,2)

  anomalies = reconstruction_error > threshold
  anomalous_timesteps = 50 + np.where(anomalies)[0]   # Staring 50 sequence length is unpredicted

  # Store anomalous timesteps as an array
  ad = np.array(anomalous_timesteps)

  #print(f"Anomalous Timesteps: {ad}")

  #Bit Matrix generation based on threshold
  predicted_array =np.zeros((len(train_data_normalized), 38))
  for i in range(len(error)):
    for j in range(38):
      if error[i][j] > threshold :
        predicted_array[i][j] = 1

  Predicted=[]
  flag=0
  for i in range(len(predicted_array)):
    for j in ad:
      if i==j:
        flag=1
        break
    if flag==1:
      Predicted.append(1)
    else:
      Predicted.append(0)
    flag=0

  predicted_ranges = find_index_ranges(Predicted)

  # Calculating performance metrices for finding anomalous timesteps

  true_positives, false_positives, false_negatives = calculate_detailed_metrics(predicted_ranges, actual_ranges)

  Precision = (true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else 0
  Recall = (true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 0
  F1_score = (2 * Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0

  if F1_score > best_f1_timesteps:
    best_f1_timesteps = F1_score
    best_precision_timesteps = Precision
    best_recall_timesteps = Recall
    best_threshold_timesteps = threshold

  # Compute column-wise means

  updated_predicted_range = [(start - 50, end - 50) for start, end in predicted_ranges]

  updated_actual_range = [(start - 50, end - 50) for start, end in set1]

  error_matrix = error[50:]

  # Compute column-wise means


  means = compute_means(updated_predicted_range, updated_actual_range, error_matrix)

  dom = [value for value in means.values()]
  dom = np.array(dom)

  predicted_feature = [[i+1 for i, val in enumerate(row) if val > threshold] for row in dom]

  total_TP = 0
  total_FP = 0
  total_FN = 0

  # Compute TP, FP, FN using threshold-based logic
  for predicted, actual in zip(predicted_feature, feature_list):
    predicted_set = set(predicted)
    actual_set = set(actual)

    matched = len(predicted_set & actual_set)

    # Get the length of binary vector after padding
    max_index = max(predicted_set.union(actual_set))

    # Create binary vectors of length = max_index
    binary_predicted = [0] * max_index
    binary_actual = [0] * max_index

    for idx in predicted_set:
        binary_predicted[idx - 1] = 1  # 1-based to 0-based index

    for idx in actual_set:
        binary_actual[idx - 1] = 1

    if matched >= 1:
      for i in range(len(binary_actual)):
        if binary_actual[i]==1:
          binary_predicted[i]=1

    TP, FP, FN = 0, 0, 0
    for a, p in zip(binary_actual, binary_predicted):
      if a == 1 and p == 1:
        TP += 1
      elif a == 0 and p == 1:
        FP += 1
      elif a == 1 and p == 0:
        FN += 1

    total_TP += TP
    total_FP += FP
    total_FN += FN

  # Compute precision, recall, and F1-score
  precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
  recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
  f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


  if f1 > best_f1:
    best_f1 = f1
    best_precision = precision
    best_recall = recall
    best_threshold = threshold

  threshold += 0.1

print(f"Best Threshold:", best_threshold_timesteps)
print(f"Precision: {best_precision_timesteps}")
print(f"Recall: {best_recall_timesteps}")
print(f"F1 Score: {best_f1_timesteps}")

# Display metrics
print(f"Best Threshold:", best_threshold)
print(f"Precision: {best_precision}")
print(f"Recall: {best_recall}")
print(f"F1 Score: {best_f1}")