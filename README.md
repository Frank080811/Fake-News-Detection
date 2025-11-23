# 📰 Fake News Detection Using Machine Learning & NLP

A Complete End-to-End Pipeline: From Text Cleaning → Feature Engineering → TF-IDF → Model Training → Hyperparameter Tuning → Final Predictions

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📖 Table of Contents

- [About the Project](#about-the-project)
- [Built With](#built-with)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)

## 🎯 Project Overview

This project builds a machine learning classifier capable of distinguishing between real and fake news headlines.
You are provided with:

training_data.csv — Contains labeled headlines

0 → Fake

1 → Real

testing_data.csv — Contains unlabeled headlines (label = 2)
Your final task is to replace 2 with the predicted label (0 or 1).

This project demonstrates a full modern NLP workflow:

This project demonstrates a full modern NLP workflow:
✔ Data Cleaning
✔ Stopword Removal
✔ Feature Engineering
✔ TF-IDF Vectorization
✔ Model Training
✔ Hyperparameter Tuning
✔ Evaluation with Confusion Matrix & ROC
✔ Final Prediction Export


**📂 Project Structure**
│── data/
│   ├── training_data.csv
│   ├── testing_data.csv
│
│── notebooks/
│   ├── fake_news_model.ipynb
│
│── plots/
│   ├── final_predictions.csv
│ 
│
│── README.md


**🧹 Data Preprocessing**
- ✔ Cleaning Performed

* Remove HTML tags, scripts, and CSS 

* Remove punctuation and numbers

* Convert to lowercase

* Remove single characters & reduce multiple spaces

* Stopword removal using NLTK

**✔ Why?**

Cleaning text standardizes inputs, reduces noise, and improves signal captured by feature extraction and ML algorithms.

## ✨ Feature Engineering
In addition to TF-IDF, several hand-crafted NLP features were engineered to improve model performance:

🔎 Extra  Features
Feature	Description	Purpose
char_length	Number of characters	Detect exaggerated long/short headlines
word_count	Number of words	Captures headline complexity
avg_word_length	Character/word ratio	Fake news often uses short emotional words
capital_ratio	Ratio of capital letters	Clickbait uses CAPITALIZATION
exclamation_count	Count of !	Emotional exaggeration
suspicious_keywords	“shocking, breaking, exposed…”	Fake headlines often use sensational words
sentiment_polarity + subjectivity	From TextBlob	Fake news tends to be highly subjective
starts_with_number	Headlines beginning with a number	Clickbait detection
contains_listicle	“X reasons why…” patterns	Strong clickbait indicator

**✔ Why Add Extra Features?**

These features capture behavioral patterns often found in fake news beyond just words. They help classical ML models outperform simple TF-IDF alone.

## 🔠 Vectorization — TF-IDF

### TF-IDF used with:

max_features = 8000

n-gram range = (1, 2) (unigrams + bigrams)

min_df = 2 (ignore rare words)

max_df = 0.9 (ignore overly frequent words)

sublinear_tf = True

**✔ Why TF-IDF?**

It balances word frequency with importance.
Captures meaningful differences between real & fake tone.

## 🤖 Models Trained

A set of traditional ML models were evaluated:

### 🧪 Models

* Logistic Regression

* Linear SVM (Support Vector Machine)

* Random Forest

* XGBoost

### 📊 Baseline Results
Model	Accuracy	Precision	Recall	F1 Score
Linear SVM	0.9407	0.9318	0.9472	0.9394
Logistic Regression	0.9395	0.9265	0.9508	0.9385
Random Forest	0.9253	0.9110	0.9379	0.9242
XGBoost	0.9061	0.8792	0.9352	0.9063

### ✔ Best Model Selected: Linear SVM

* Highest overall accuracy

* Best F1 score

* Best balance between recall & precision

* Performs extremely well with TF-IDF

## 🔧 Hyperparameter Tuning

### Used GridSearchCV to optimize:

**Logistic Regression**

* C, solver, class_weight

* Linear SVM

* C, class_weight, loss

**Random Forest**

* n_estimators, max_depth, max_features

**XGBoost**

*n_estimators, learning_rate, max_depth, subsample

## 📈 Model Evaluation
### 🟦 Confusion Matrix

Shows strong diagonal dominance → correct predictions.

## 🧪 Final Testing Predictions

Using the best model, predictions were generated for testing_data.csv.

### Prediction flow:

* Apply same cleaning

* Apply same stopword removal

* Extract TF-IDF + engineered features

* Combine into final feature matrix

* Predict 0 or 1

Final output saved as:
### final_testing.csv

## 🧠 Key Insights

* Fake news exhibits clear linguistic patterns (sensational words, punctuation, subjective tone).

* Classical ML (SVM + TF-IDF + engineered features) can outperform deep learning on short text.

* Feature engineering significantly boosted model performance.
```

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

### How to Contribute

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- Additional architecture implementations
- Performance optimizations
- Enhanced visualizations
- Documentation improvements


## 📄 License

Distributed under the MIT License. See `LICENSE` file for more information.

## 👨‍💻 Authors

**Frank Sarfo** - https://github.com/FRANK080811

**Bosco** - https://github.com/bosco2024


### Tools
- Google Colab for GPU acceleration
- Matplotlib and Seaborn for visualization
- scikit-learn for evaluation metrics

---

<div align="center">

### ⭐ Don't forget to star this repository if you found it helpful!

</div>
