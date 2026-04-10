# SMS Spam Classification

## 📄 Project Description
This project was developed as part of my **Machine Learning Internship at Codmetric**. The objective is to build a classification model that can accurately distinguish between **Spam** (unwanted/scam) and **Ham** (legitimate) SMS messages using Natural Language Processing (NLP) techniques.

## 📊 Dataset Overview
The project uses the **SMS Spam Collection Dataset**. 
- **Ham Samples:** 4,825
- **Spam Samples:** 747
- **Total Samples:** 5,572

## 🛠️ Methodology & Requirements
As per the internship requirements, the following steps were implemented:

1. **Exploratory Data Analysis (EDA):**
   - Performed data cleaning by removing unnecessary columns.
   - Visualized the distribution of Ham vs. Spam using `Seaborn`.

2. **Text Preprocessing:**
   - Removed special characters and digits using Regular Expressions (`re`).
   - Converted text to lowercase for uniformity.
   - Tokenized text and removed English stopwords using the `NLTK` library to focus on meaningful words.

3. **Feature Extraction:**
   - Used `CountVectorizer` (Bag of Words) to convert text data into a numerical matrix that the machine learning model can process.

4. **Model Training:**
   - Implemented the **Multinomial Naive Bayes** classifier, which is highly efficient for text classification tasks.

5. **Evaluation:**
   - Evaluated the model using a Confusion Matrix and Classification Report.

## 🚀 Results
The model performed exceptionally well on the test dataset:
- **Accuracy Score:** 97.13%
- **Precision (Ham):** 0.99
- **Precision (Spam):** 0.86

The **Confusion Matrix** confirmed that the model correctly identified 942 Ham messages and 141 Spam messages from the test set.

## 💻 Tech Stack
- **Language:** Python
- **Libraries:** Pandas, Scikit-learn, NLTK, Matplotlib, Seaborn
- **Tool:** VS Code / Jupyter Notebook
