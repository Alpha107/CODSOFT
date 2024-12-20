# Data Science Internship Projects - CodSoft Virtual Internship

Welcome to the comprehensive repository of projects completed during the Data Science virtual internship at CodSoft! This repository showcases the work done during the 4-week internship from December 15, 2024, to January 15, 2025. As part of this internship, we worked on real-world data science problems, applying machine learning and deep learning techniques to solve them.

## Projects Overview

### 1. **Sales Prediction Using Regression Models**
This project focuses on predicting sales based on various advertising channels (TV, Radio, and Newspaper) using machine learning algorithms. The goal was to build a model that predicts sales values based on historical data, providing insights into the effectiveness of advertising spend across different media channels.

**Key Components:**
- **Data Cleaning:** The dataset is preprocessed to remove duplicates, handle missing values, and eliminate outliers.

- **Exploratory Data Analysis (EDA):** Correlation heatmaps, scatter plots, and pair plots were created to visualize relationships between features and the target variable (Sales).

- **Feature Engineering:** Polynomial features were added to improve the model's ability to capture non-linear relationships.

- **Modeling:** We used several regression models such as Random Forest Regressor, Gradient Boosting Regressor, and Linear Regression. Hyperparameter tuning was performed for the Random Forest model.

- **Evaluation:** The model's performance was evaluated using metrics like Mean Squared Error (MSE), R-squared (R²), and Mean Absolute Error (MAE).

- **Model Deployment:** The best-performing model was saved using `joblib` for future use.

**Technologies Used:**
- Python (pandas, numpy, scikit-learn, matplotlib, seaborn)

- Random Forest Regressor, Gradient Boosting Regressor, Linear Regression

### 2. **Titanic Survival Prediction**
This project involved predicting whether a passenger survived the Titanic disaster based on features such as age, sex, passenger class, fare, and embarkation port. The goal was to use classification models to predict survival and analyze feature importance.

**Key Components:**
- **Data Preprocessing:** Missing values were handled using imputation for age and mode imputation for embarkation port. Categorical variables like sex and embarkation port were encoded using Label Encoding.

- **Feature Engineering:** A new feature, "FamilySize," was created by combining SibSp and Parch columns.

- **Modeling:** We used the Random Forest Classifier and performed hyperparameter tuning to find the best model.

- **Evaluation:** The model was evaluated using metrics such as accuracy, confusion matrix, classification report, and ROC-AUC score.

- **Model Deployment:** The best model was saved for future predictions.

**Technologies Used:**
- Python (pandas, numpy, scikit-learn, matplotlib, seaborn)

- Random Forest Classifier

### 3. **Credit Card Fraud Detection Using Neural Networks**
This project aimed to detect fraudulent credit card transactions using a deep learning model. The dataset contained highly imbalanced classes (fraud vs. non-fraud), and techniques like SMOTE (Synthetic Minority Over-sampling Technique) were used to balance the data.

**Key Components:**
- **Data Preprocessing:** The dataset was scaled using StandardScaler, and class imbalance was handled using SMOTE.

- **Modeling:** A simple feed-forward neural network was built using PyTorch. The model architecture consisted of three fully connected layers with ReLU activations and a sigmoid output layer.

- **Training:** The model was trained using the Adam optimizer and Binary Cross-Entropy loss function.

- **Evaluation:** The model's performance was evaluated using metrics like confusion matrix, classification report, ROC-AUC score, and ROC curve.

- **Model Deployment:** The trained model was saved using PyTorch for future use.

**Technologies Used:**
- Python (pandas, numpy, scikit-learn, matplotlib, seaborn, imbalanced-learn)

- PyTorch (neural networks, training, evaluation)

---

## Internship Experience

During the 4-week virtual internship at **CodSoft**, I had the opportunity to work on real-world data science problems. The internship provided a platform to apply theoretical knowledge in machine learning and deep learning, gaining hands-on experience in:

- **Data Preprocessing:** Cleaning, transforming, and handling missing or imbalanced data.

- **Modeling:** Building and fine-tuning various machine learning and deep learning models.

- **Evaluation:** Evaluating model performance using appropriate metrics and visualizations.

- **Deployment:** Saving models for future predictions and use in real-world applications.

### Skills Developed
- **Machine Learning:** Regression, Classification, Model Evaluation, Hyperparameter Tuning

- **Deep Learning:** Neural Networks, PyTorch

- **Data Analysis & Visualization:** Exploratory Data Analysis (EDA), Data Preprocessing, Visualizations

- **Tools & Libraries:** Python, pandas, numpy, scikit-learn, matplotlib, seaborn, PyTorch, SMOTE, joblib

---

## Acknowledgements

I would like to extend my sincere gratitude to **CodSoft** for providing this valuable learning opportunity. The internship has been an incredible journey of growth and learning, allowing me to work on real-world projects and gain practical experience in data science and machine learning.

---

## Conclusion

This repository serves as a collection of my work during the **Data Science** virtual internship at **CodSoft**. It showcases the application of various machine learning and deep learning techniques to solve real-world problems, from predicting sales and survival outcomes to detecting credit card fraud. I hope this work serves as a useful resource for those interested in learning more about data science, machine learning, and deep learning.

Feel free to explore the code, run the models, and adapt them for your own projects. Happy coding!
