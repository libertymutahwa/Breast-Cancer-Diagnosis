# Breast Cancer Diagnosis with Scikit-learn

## 📖 Overview
**Breast Cancer Diagnosis with Scikit-learn** is a machine learning project that classifies breast cancer tumors as malignant or benign using the Wisconsin Breast Cancer dataset. The project leverages Python's `scikit-learn` library to implement a comprehensive pipeline, including data preprocessing, feature selection, dimensionality reduction, model training, and evaluation. Key techniques include Principal Component Analysis (PCA), Random Forest feature importance, and ensemble methods like Voting Classifier, achieving high accuracy for medical diagnostics.

This project demonstrates a practical application of machine learning in medical diagnostics, focusing on breast cancer—a critical area in healthcare. It’s designed to be reproducible and accessible for researchers, data scientists, and bioinformatics professionals interested in exploring machine learning workflows for classification tasks.

## 🛠️ Installation
Follow these steps to set up the project on your local machine:

1. **Clone the Repository**:
   ```
   git clone https://github.com/libertymutahwa/Breast-Cancer-Diagnosis.git
   cd Breast-Cancer-Diagnosis
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.6+ installed. Install the required packages using the provided `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```
   Dependencies include:
   - `jupyter`
   - `numpy`
   - `pandas`
   - `scikit-learn`
   - `matplotlib` (optional, for visualization if extended)

3. **Launch Jupyter Notebook**:
   Start Jupyter Notebook to run the project:
   ```
   jupyter notebook
   ```
   Open `main.ipynb` in the browser interface.

## 📊 Dataset
The project uses the **Wisconsin Breast Cancer dataset**, which is directly loaded via `sklearn.datasets.load_breast_cancer()`. No external download is required. The dataset contains:
- **Samples**: 569
- **Features**: 30 (e.g., radius, texture, perimeter of cell nuclei)
- **Classes**: Malignant (0) and Benign (1)
- **Class Distribution**: 212 malignant, 357 benign

The dataset is split into training (80%) and testing (20%) sets with stratification to maintain class balance.

## 🚀 Usage
1. **Run the Notebook**:
   Open `main.ipynb` in Jupyter Notebook and execute all cells. The notebook:
   - Loads the dataset using `load_breast_cancer()`.
   - Performs exploratory analysis (e.g., PCA to visualize variance).
   - Identifies important features using Random Forest.
   - Trains and evaluates multiple models (Logistic Regression, SVM, Random Forest, Voting Classifier).
   - Outputs performance metrics (accuracy, confusion matrix, classification report, sensitivity, specificity).

2. **Expected Outputs**:
   - Dataset shape and class distribution.
   - Variance explained by PCA (first 2 components).
   - Top 10 features by importance.
   - Model accuracies (e.g., Logistic Regression: ~97%).
   - Confusion matrix and classification report for the best model.
   - Sensitivity and specificity metrics.

## 🔍 Methodology
The project follows a structured machine learning workflow:
1. **Data Preprocessing**:
   - Load the Wisconsin Breast Cancer dataset using `scikit-learn`.
   - Split into training (80%) and testing (20%) sets with stratification (`train_test_split`).

2. **Dimensionality Reduction**:
   - Apply PCA to reduce features to 2 dimensions and calculate the explained variance (e.g., ~63%).

3. **Feature Selection**:
   - Use Random Forest to identify the top 10 most important features (e.g., worst area, worst concave points).

4. **Model Training**:
   - Train four models:
     - Logistic Regression (`max_iter=10000`)
     - Support Vector Machine (SVM) with probability estimates
     - Random Forest
     - Voting Classifier (soft voting with the above models)

5. **Evaluation**:
   - Compute accuracy for all models.
   - For the best model (Logistic Regression), provide:
     - Confusion matrix (e.g., [[42 1], [2 69]])
     - Classification report (precision, recall, F1-score)
     - Sensitivity (~97%) and specificity (~98%)

## 📈 Results
The project achieves robust performance for breast cancer classification:
- **Best Model**: Logistic Regression
- **Accuracy**: 97.37%
- **Confusion Matrix**:
  ```
  [[42  1]
   [ 2 69]]
  ```
- **Classification Report**:
  ```
              precision    recall  f1-score   support
  malignant       0.95      0.98      0.97        43
  benign          0.99      0.97      0.98        71
  accuracy                            0.97       114
  macro avg       0.97      0.97      0.97       114
  weighted avg    0.97      0.97      0.97       114
  ```
- **Sensitivity**: 97.18% (ability to correctly identify benign cases)
- **Specificity**: 97.67% (ability to correctly identify malignant cases)

These results highlight the model’s effectiveness for medical diagnostics, balancing false positives and false negatives—a critical consideration in healthcare.

## 🤝 Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Open a Pull Request with a description of your changes.

Potential enhancements:
- Add visualizations (e.g., PCA scatter plot, feature importance bar chart).
- Implement cross-validation for more robust evaluation.
- Explore additional models or hyperparameter tuning.

## 📧 Contact
For questions, feedback, or collaboration opportunities, reach out to me:
- **Email**: mutahwaliberty@gmail.com
- **GitHub**: [libertymutahwa](https://github.com/libertymutahwa)
- **LinkedIn**: www.linkedin.com/in/mutahwa-liberty



## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

