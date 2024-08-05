import pandas as pd
import json
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
from sklearn.linear_model import LogisticRegression

# def get_song_probability(df):
    
#     # Aggregate snippet features per song
#     features = data.groupby('song_name')['snip_probability'].agg(['mean', 'std', 'max', 'min']).reset_index()

#     # Map song_class to each song (assuming the class is the same for all snippets of a song)
#     features['song_class'] = data.groupby('song_name')['song_class'].first().values

#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(features[['mean', 'std', 'max', 'min']], features['song_class'], test_size=0.2, random_state=42)

#     # Initialize and train logistic regression model
#     model = LogisticRegression()
#     model.fit(X_train, y_train)

#     # Predict on test set
#     predictions = model.predict(X_test)

#     # Calculate accuracy
#     accuracy = accuracy_score(y_test, predictions)
#     print(f'Accuracy: {accuracy}')

#     # Optionally, use the model to predict class probabilities for new song data
#     probabilities = model.predict_proba(X_test)[:, 1]  # Probability of being in class 1
#     print(probabilities)
def generate_and_save_confusion_matrix(predicted_labels, true_labels, save_path, epoch,method):
    """
    Generate and save a confusion matrix, ROC curve, and calculate AUC based on predicted and true labels.

    Args:
    predicted_labels (list): List of predicted labels.
    true_labels (list): List of actual true labels.
    save_path (str): Directory to save the confusion matrix plot, ROC curve, and AUC.
    epoch (int): Epoch number to tag the output files for tracking.

    Returns:
    None
    """
    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)

    # Compute the confusion matrix
    matrix = confusion_matrix(true_labels, predicted_labels)
    matrix_df = pd.DataFrame(matrix, index=[i for i in range(matrix.shape[0])],
                             columns=[i for i in range(matrix.shape[1])])

    # # Save the confusion matrix to CSV
    # csv_file_path = os.path.join(save_path, f'{epoch}_confusion_matrix.csv')
    # matrix_df.to_csv(csv_file_path)
    # print(f"Confusion matrix saved to {csv_file_path}.")

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=True, yticklabels=True)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Save the plot
    plot_file_path = os.path.join(save_path, f'{method}_{epoch}_confusion_matrix.png')
    plt.savefig(plot_file_path)
    plt.close()
    print(f"Plot of confusion matrix saved as {plot_file_path}.")

    # Generate ROC curve and AUC
    fpr, tpr, _ = roc_curve(true_labels, predicted_labels, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Save the ROC plot
    roc_plot_file_path = os.path.join(save_path, f'{method}_{epoch}_roc_curve.png')
    plt.savefig(roc_plot_file_path)
    plt.close()
    print(f"ROC curve plot saved as {roc_plot_file_path}. AUC = {roc_auc:.2f}")

    correct_predictions = (true_labels == predicted_labels).sum()
    total_predictions = len(true_labels)
    accuracy = correct_predictions / total_predictions

    print(f'Song Accuracy : {accuracy*100}%')
    return roc_auc  # Optionally return AUC if you want to use it programmatically later

# Path to your JSON file
json_file_path = '/blue/srampazzi/bhattr/music_genre_classification/classification_test_set_140.json'
# Define a path to save the CSV files and plots
save_path = 'test_set_results/'  # Adjust this to your preferred save directory

# Read the JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Collect all unique songs and epochs
all_songs = set()
epochs = set()
batch_loss_per_epoch = {}  # Dictionary to store batch losses per epoch

for d in data:
    batch_data = d['data']
    epochs.add(d['epoch'])
    for item in batch_data:
        all_songs.add(item['song_name'])
    # Collect batch losses per epoch
    if 'batch_loss' in d:
        batch_loss_per_epoch[d['epoch']] = d['batch_loss']

# Convert sets to sorted lists to maintain order
all_songs = sorted(list(all_songs))
epochs = sorted(list(epochs))

# Calculate the average batch loss per epoch
average_loss_per_epoch = {}
for epoch, losses in batch_loss_per_epoch.items():
    if losses:
        average_loss_per_epoch[epoch] = sum(losses) / len(losses)

# Process each epoch
# Majority Vote
print("======================Majority Vote==============================")
for epoch in epochs:
    epoch_data = next((item for item in data if item['epoch'] == epoch), None)
    if epoch_data:
        df = pd.DataFrame(epoch_data['data'])
    else:
        df = pd.DataFrame(columns=['song_name', 'snip_probability', 'snip_predict_label', 'snip_true_label'])

    df_majority = df.groupby('song_name').agg({
        'snip_predict_label': lambda x: x.mode().values[0] if len(x.mode().values) > 0 else 0,
        'snip_true_label': lambda x: x.mode().values[0] if len(x.mode().values) > 0 else 0
    }).reindex(all_songs).fillna({
        'snip_predict_label': 0,
        'snip_true_label': 0
    }).reset_index()

    df_majority['Epoch'] = epoch
    df_majority.columns = ['Song Name', 'Predicted Label', 'True Label', 'Epoch']

    generate_and_save_confusion_matrix(df_majority['Predicted Label'], df_majority['True Label'], save_path, epoch, method='majority_vote')
    file_name_majority = f'majority_vote_results_epoch_{epoch}.csv'
    df_majority.to_csv(save_path + file_name_majority, index=False)
    print(f"Majority Vote results for epoch {epoch} saved to {file_name_majority}.")


#  Average Probability
print("======================Avg Probability==============================")
for epoch in epochs:
    epoch_data = next((item for item in data if item['epoch'] == epoch), None)
    if epoch_data:
        df = pd.DataFrame(epoch_data['data'])
    else:
        df = pd.DataFrame(columns=['song_name', 'snip_probability', 'snip_predict_label', 'snip_true_label'])

    df = df.groupby('song_name').agg({
        'snip_probability': 'mean',
        'snip_predict_label': lambda x: x.mode().values[0] if len(x.mode().values) > 0 else 0,
        'snip_true_label': lambda x: x.mode().values[0] if len(x.mode().values) > 0 else 0
    }).reindex(all_songs).fillna({
        'snip_probability': 0,
        'snip_predict_label': 0,
        'snip_true_label': 0
    }).reset_index()

    df['Epoch'] = epoch
    # print(df.columns)
    df.columns = ['Song Name', 'Average Probability', 'Predicted Label', 'True Label', 'Epoch']
    # print(os.path.basename(df['Song Name'][0]))
    # df['Sort Key'] = df['Song Name'].apply(lambda x: int(x.split('/')[-1].split('__')[-1]))
    # df.sort_values('Sort Key', inplace=True)
    # df.drop(columns='Sort Key', inplace=True)

    generate_and_save_confusion_matrix(df['Predicted Label'],df['True Label'],save_path,epoch,method='avg_prob')
    file_name = f'avg_probability_results_epoch_{epoch}.csv'
    df.to_csv(save_path + file_name, index=False)
    print(f"Results for epoch {epoch} saved to {file_name}.")


# Logistic
print("======================Logistic Regression==============================")
for epoch in epochs:
    epoch_data = next((item for item in data if item['epoch'] == epoch), None)
    if epoch_data:
        df = pd.DataFrame(epoch_data['data'])
    else:
        df = pd.DataFrame(columns=['song_name', 'snip_probability', 'snip_predict_label', 'snip_true_label'])

    # Aggregate probabilities and labels per song
    song_data = df.groupby('song_name').agg({
        'snip_probability': ['mean', 'std', 'max', 'min'],
        'snip_true_label': lambda x: x.mode().values[0] if len(x.mode().values) > 0 else 0
    }).reset_index()
    
    song_data.columns = ['Song Name', 'Mean Probability', 'STD Probability', 'Max Probability', 'Min Probability', 'True Label']

    # Split the data into features and target
    X = song_data[['Mean Probability', 'STD Probability', 'Max Probability', 'Min Probability']]
    y = song_data['True Label']

    # Train logistic regression model
    if not X.empty and not y.empty:
        model = LogisticRegression()
        model.fit(X, y)

        # Predict using the trained model
        song_data['Predicted Label'] = model.predict(X)

        # Generate and save confusion matrix
        generate_and_save_confusion_matrix(song_data['Predicted Label'], song_data['True Label'], save_path, epoch,method="Logistic")

        # Save the results
        file_name = f'Logistic_results_epoch_{epoch}.csv'
        song_data.to_csv(save_path + file_name, index=False)
        print(f"Results for epoch {epoch} saved to {file_name}.")
    else:
        print(f"No data available for epoch {epoch}.")
        
# Plotting the average batch loss per epoch
plt.figure(figsize=(10, 6))
sorted_epochs = sorted(average_loss_per_epoch.keys(), key=lambda x: int(x))
plt.plot(sorted_epochs, [average_loss_per_epoch[epoch] for epoch in sorted_epochs], marker='o', linestyle='-')
plt.title('Average Batch Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Batch Loss')
plt.grid(True)

# Save the plot in the results folder
plot_filename = save_path + 'average_batch_loss_per_epoch.png'
plt.savefig(plot_filename)
plt.close()

print(f"Plot of average batch loss per epoch saved as {plot_filename}.")
