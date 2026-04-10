import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download NLTK stopwords
nltk.download('stopwords')

# 1. Load the dataset 
# (Note: Kaggle ki file aksar 'latin-1' encoding mein hoti hai)
df = pd.read_csv('spam.csv', encoding='latin-1')

# Cleaning the dataframe (dropping unnecessary columns)
df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
df.columns = ['label', 'message']

# 2. Basic Exploratory Data Analysis (EDA)
print("--- Dataset Info ---")
print(df.info())
print("\n--- Class Distribution ---")
print(df['label'].value_counts())

# Visualize distribution
sns.countplot(x='label', data=df, palette='viridis')
plt.title('Distribution of Ham vs Spam')
plt.show()

# 3. Text Preprocessing
def preprocess_text(text):
    # Remove punctuation and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and remove stopwords
    text = text.split()
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

print("\nPreprocessing text... please wait.")
df['clean_message'] = df['message'].apply(preprocess_text)

# 4. Feature Extraction & Data Splitting
# Converting text to numbers using Bag of Words (CountVectorizer)
cv = CountVectorizer()
X = cv.fit_transform(df['clean_message']).toarray()
y = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Simple Classifier (Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)

print(f"\nAccuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Visualize Results with Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Spam Classification')
plt.show()