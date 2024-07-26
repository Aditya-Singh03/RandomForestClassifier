# Random Forest Classifier

This project implements a Random Forest Classifier model from scratch in Python. The implementation includes core components for decision trees (Node, Decision Tree class), information gain calculations, model training, prediction, and evaluation metrics (accuracy, precision, recall, F1-score). 

## Project Overview

This project demonstrates the development of a Random Forest Classifier model from the ground up using Python, without relying on external machine learning libraries. The model is tested on the Wine dataset, the Voting dataset, and additional datasets (Cancer and Contraceptive Method Choice). The project aims to showcase the inner workings of a Random Forest Classifier and provide insights into its performance across different datasets.

### Key Components

* **Node class:** Represents a node in the decision tree, storing information about the feature, threshold, information gain, and child nodes (left and right).
* **DT class (Decision Tree):** Implements the core functionality of a decision tree, including best split calculations, tree building, prediction, and accuracy evaluation.
* **RandomForest class:** Builds an ensemble of decision trees (Random Forest), each trained on a bootstrapped sample of the data. Includes methods for fitting the model, making predictions, and calculating evaluation metrics.
* **Data Loading:** Fetches the Wine, Voting, Cancer, and Contraceptive Method Choice datasets from CSV files and the UCI Machine Learning Repository.
* **Data Preprocessing:** Splits the data into training and testing sets, applies stratified k-fold cross-validation to ensure robust evaluation, and shuffles the data for randomization.
* **Model Training:** Fits the Random Forest model on the training data using both Gini impurity and entropy as split criteria.
* **Model Evaluation:** Evaluates the model's performance using metrics such as accuracy, precision, recall, and F1-score.
* **Result Visualization:** Generates plots to visualize the performance of the model across different numbers of trees and datasets.

## Usage

1. **Prerequisites:**
   * Python 3.x
   * NumPy
   * Pandas
   * matplotlib
   * sklearn
   * ucimlrepo

2. **Installation:**
    ```bash
    pip install numpy pandas matplotlib sklearn ucimlrepo
    ```



3. **Data:**
   * The Wine and Voting datasets are included as CSV files (`hw3_wine.csv`, `hw3_house_votes_84.csv`).
   * The Cancer and Contraceptive Method Choice datasets are fetched using `ucimlrepo`.

4. **Execution:**
   * **Just execute**
        >`python run.py`
    
        in the terminal and you will be able to run the model and create the its plots in one go. All the plots will also get saved in this very same folder. (Just make sure that when you try to run this file, you are in the same folder as the file's parent folder)

## Dataset Descriptions

* **Wine Dataset:** Contains 13 attributes describing various chemical properties of wines (e.g., alcohol content, acidity levels) and a class label indicating the wine type (1, 2, or 3).
* **House Votes Dataset:** Includes 16 attributes representing votes ("yes" or "no") on different political issues and a class label indicating the party affiliation ("democrat" or "republican").
* **Breast Cancer Wisconsin (Diagnostic) Dataset:**  Offers 30 numerical features extracted from breast cancer cell images, with the goal of classifying tumors as benign or malignant.
* **Contraceptive Method Choice Dataset:** Provides 9 socio-economic and demographic features (e.g., age, education, number of children) along with a target variable indicating the chosen contraceptive method.

## Key Features

* **From-Scratch Implementation:** The Random Forest model, including the decision trees, is built entirely from scratch, giving you a deeper understanding of the algorithm's inner workings.
* **Gini Impurity and Entropy:** The code supports both Gini impurity and information gain (entropy) as criteria for splitting nodes in the decision trees.
* **Multiple Datasets:** The model is evaluated on four datasets:
    * **Wine Dataset:** A classic dataset for classifying wine types based on various chemical properties. ([Source: UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/109/wine))
    * **House Votes Dataset:** A dataset related to U.S. Congressional voting records on different issues. ([Source: UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records))
    * **Breast Cancer Wisconsin (Diagnostic) Dataset:** A medical dataset for classifying breast cancer tumors as benign or malignant. ([Source: UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)))
    * **Contraceptive Method Choice Dataset:** A dataset exploring the factors influencing contraceptive method choice among women. ([Source: UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice))
* **Stratified K-Fold Cross-Validation:** Rigorous evaluation is conducted using stratified k-fold cross-validation to prevent overfitting and ensure robust performance estimates.
* **Performance Evaluation:** The model's effectiveness is measured using accuracy, precision, recall, and F1-score.
* **Result Visualization:** Clear and informative plots illustrate how the model's performance varies with the number of trees in the ensemble.

## Results

The code outputs plots showcasing the relationship between the number of trees and various evaluation metrics (accuracy, precision, recall, F1-score) for each dataset. Additionally, it prints the average values of these metrics for different numbers of trees.

**Key Observations:**
* Increasing the number of trees generally improves model performance up to a certain point, after which the gains become marginal.
* The choice between Gini impurity and entropy as the split criterion might lead to minor differences in performance, but the overall trend remains similar.



## Additional Notes

* The implementation includes stratified k-fold cross-validation to mitigate overfitting and provide more robust evaluation results.
* The code includes detailed comments and explanations to facilitate understanding.

## Future Improvements

* Explore different hyperparameter settings (e.g., maximum depth of trees, minimum samples per leaf) to further optimize model performance.
* Implement feature importance analysis to gain insights into the most influential features for each dataset. 
* Experiment with other datasets to assess the generalizability of the model.


>[!NOTE]
>This implementation is intended solely for educational purposes and should not be used for production or critical applications.

---

**Let me know if you'd like any modifications or have specific aspects you want to highlight more prominently.**






