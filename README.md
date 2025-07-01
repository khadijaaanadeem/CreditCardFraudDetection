
This project builds a machine learning model to detect fraudulent credit card transactions using historical transaction data. By analyzing transaction patterns, the model helps identify suspicious activities early and reduce risk for financial institutions.

 Features:
✅ Data Exploration & Visualization

Analyzed transaction distributions, amount patterns, and correlations.

Visualized the correlation matrix to understand relationships between features.

✅ Data Preprocessing

Handled highly imbalanced datasets (only ~0.02% fraudulent cases).

Separated input features and target labels.

✅ Modeling

Trained a Random Forest Classifier to distinguish between normal and fraudulent transactions.

Used train-test split to evaluate model performance.

✅ Evaluation

Assessed metrics including accuracy, precision, recall, F1-Score, and Matthews Correlation Coefficient.

Created a confusion matrix heatmap for clarity.

 Technologies Used:
Python

pandas & NumPy

scikit-learn

RandomForestClassifier

train_test_split

classification metrics

Matplotlib & Seaborn

Results:
The trained model achieved:

Accuracy: 99.96%

Precision: 98.73% (few false alarms)

Recall: 79.59% (caught most frauds)

F1-Score: 88.14%

Matthews Correlation Coefficient: 0.8863

Note: Due to class imbalance, accuracy alone is not sufficient; precision, recall, and F1 provide a clearer picture.

 How to Use:
Load the dataset

python
Copy
Edit
data = pd.read_csv("creditcard.csv")
Preprocess the data

Separate features and target labels

Convert to NumPy arrays

Split the dataset

80% training, 20% testing

Train the Random Forest model

python
Copy
Edit
model = RandomForestClassifier()
model.fit(xTrain, yTrain)
Evaluate

Calculate metrics and visualize the confusion matrix

 Future Improvements:
Apply resampling techniques (SMOTE, undersampling) to handle class imbalance more effectively

Experiment with ensemble models and XGBoost

Deploy the model as an API for real-time fraud detection



