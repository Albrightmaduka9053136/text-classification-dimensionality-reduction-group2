# Text Classification with Dimensionality Reduction
**Group 2**

### Overview

In this project, you will perform text classification using multiple machine learning techniques. You will learn how to transform text data into numerical features, reduce dimensionality, and evaluate model performance using various metrics and visualizations.

#### Dataset
`Data`: A text based dataset.

`Task`: Binary classification (such as: Positive vs Negative; Spam or Ham, etc).

`Labels`: this project will use supervised learning.

`Size`: ~2,000 documents/reviews/emails/ any text dataset that you will be able to use binary classification.

`Split`: 75% training, 25% testing.


**Learning Objectives**
By completing this project, you will understand and implement:

- TF-IDF - Convert text to numerical features
- Confusion Matrix - Evaluate classification performance with detailed error analysis
- SVD (Singular Value Decomposition) 
- PCA (Principal Component Analysis) 
- Model Evaluation - Compare different approaches

### Project Steps
**Step 1: Data Loading and Preprocessing**
- Load the dataset
- Split into training and test sets
- Explore the data distribution
- Apply text preprocessing (optional: remove stopwords, lowercase, etc.)

**Step 2: TF-IDF Feature Extraction**
- Convert text documents into numerical TF-IDF 

- Understand how TF-IDF weights terms by importance

- Examine the vocabulary size and sample features

- Visualize the TF-IDF matrix structure


**Step 3: Baseline Model - Naive Bayes with TF-IDF**

YOU WILL IMPLEMENT:

- Train a Naive Bayes classifier on the TF-IDF features

- Make predictions on the test set

- Create and visualize a confusion matrix to analyze:

    - True Positives (TP)

    - True Negatives (TN)

    - False Positives (FP)

    - False Negatives (FN)

- Calculate accuracy, precision, recall, and F1-score

- Interpret the results: What types of errors is the model making?


**Step 4: Dimensionality Reduction with SVD**

- Apply Truncated SVD (Latent Semantic Analysis) to reduce TF-IDF dimensions

- Reduce from ~500-5000 features to 50-100 components

- Visualize the explained variance ratio

- Explain in your markdowns how SVD captures semantic relationships in text, and what it means for your data.


**Step 5: Model Training - Logistic Regression with SVD**

YOU WILL IMPLEMENT:

- Train a Logistic Regression classifier on SVD-reduced features

- Make predictions on the test set

- Create and visualize a confusion matrix for the SVD model

- Compare performance with the baseline Naive Bayes model

- Analyze: Did dimensionality reduction help or hurt performance?

- Discuss on the markdowns: How does reducing dimensions affect model accuracy and speed?


**Step 6: Dimensionality Reduction with PCA**

- Apply Principal Component Analysis (PCA) to the TF-IDF features

- First standardize the data (required for PCA)

- Reduce to the same number of components as SVD for fair comparison

- Compare PCA vs SVD variance curves


**Step 7: Model Training - Logistic Regression with PCA**

YOU WILL IMPLEMENT:

- Train a Logistic Regression classifier on PCA-reduced features

- Make predictions on the test set

- Create and visualize a confusion matrix for the PCA model

- Compare performance with both previous models

- Analyze: Which dimensionality reduction technique works better for text data?


**Step 8: Visual Comparison**

- Visualize all three confusion matrices side-by-side