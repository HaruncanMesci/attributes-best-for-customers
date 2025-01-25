import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data_path = 'processed.xlsx'
df = pd.read_excel(data_path, sheet_name='data')

# Drop unnecessary columns
unnecessary_columns = ['custid', 'birthmonth']
df = df.drop(columns=unnecessary_columns)

# Drop all columns that start with 'ln'
df = df.loc[:, ~df.columns.str.startswith('ln')]

# Drop columns with minimal variance or excessive missing values
low_variance_columns = [col for col in df.columns if df[col].nunique() < 2]
missing_value_threshold = 0.5  # 50% missing values
high_missing_columns = [col for col in df.columns if df[col].isnull().mean() > missing_value_threshold]

print("\n")

columns_to_drop = list(set(low_variance_columns + high_missing_columns))
df_reduced = df.drop(columns=columns_to_drop)
print("Columns dropped based on heuristics:", columns_to_drop)

print("\n")

# calculate correlation matrix for numeric attributes only
correlation_matrix = df_reduced.select_dtypes(include=['number']).corr()
print("Correlation matrix for numeric attributes only:",correlation_matrix)

print("\n")

# filter strong correlations
strong_corr = correlation_matrix[(correlation_matrix > 0.7) | (correlation_matrix < -0.7)]

# plot filtered heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(strong_corr, annot=False, cmap="coolwarm", cbar=True)

plt.title("Strong Correlations Heatmap")
plt.show()

print("\n")

# Identify features with high correlation (>0.5)
high_correlation_pairs = []
threshold = 0.5
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            high_correlation_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))

print("\n")

# Drop one feature from each highly correlated pair
correlated_features_to_drop = list(set(pair[0] for pair in high_correlation_pairs))
df = df.drop(columns=correlated_features_to_drop)
print("Highly correlated features dropped:", correlated_features_to_drop)

print("\n")

missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]  # Filter only columns with missing data

# Plot
missing_values.plot(kind='bar')
plt.title("Missing Data in Columns")
plt.xlabel("Columns")
plt.ylabel("Count of Missing Values")
plt.show()

print("\n")

def impute_missing(data):
    for column in data.columns:
        if data[column].dtype in ['float64', 'int64']:
            data[column].fillna(data[column].median(), inplace=True)
        else:
            data[column].fillna(data[column].mode()[0], inplace=True)

impute_missing(df)

print("\n")

# Confirm no missing values
print("\nMissing Values After Imputation:\n", df.isnull().sum().sum())

df = df.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
df = df.apply(pd.to_numeric, errors='coerce')

# Initial inspection
print("Dataset Shape:", df.shape)
print("\n")
print("\nColumn Names:\n", df.columns)
print("\n")
print("\nFirst 5 Rows:\n", df.head())
print("\n")

# Summary statistics
print("\nSummary Statistics:\n", df.describe())

print("\n")

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:\n", missing_values[missing_values > 0])

print("\n")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, classification_report

# Define predictors and target
target = 'response_01'  # Example target column
X = df.drop(columns=[target])
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)


y_pred_rf = rf.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", rf_acc)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix:")
print(conf_matrix)

# Key metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# Display all rows without truncation
pd.set_option('display.max_rows', None)

# Create and display the feature importance DataFrame
rf_feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nRandom Forest Feature Importance:")
print(rf_feature_importance)

# Plot Random Forest Feature Importance
rf_feature_importance[:10].plot(kind='barh', x='Feature', y='Importance', legend=False, title='Top 10 Random Forest Feature Importances')
plt.gca().invert_yaxis()
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

