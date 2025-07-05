# 🍷 Wine Quality Prediction App

A sophisticated machine learning application that predicts wine quality based on chemical properties. Built for Mr. Sanborn's boutique winery quality assurance team.

![Wine Quality Predictor](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 🚀 Live Demo

**[Try the App Live!](https://your-app-url-here.streamlit.app)**

## 📋 Features

- **Real-time Prediction**: Input wine chemical properties and get instant quality predictions
- **Quality Classification**: Wines are classified as "Good Quality" (7+ rating) or "Not Good Quality" (<7 rating)
- **Confidence Scoring**: See how confident the model is in its predictions
- **Feature Importance**: Understand which chemical properties most influence wine quality
- **Quality Improvement Suggestions**: Get actionable recommendations for improving wine quality
- **Professional UI**: Clean, intuitive interface designed for quality assurance teams

## 🧪 Chemical Properties Analyzed

The app analyzes 11 key chemical properties:

1. **Fixed Acidity** - Non-volatile acids that don't evaporate
2. **Volatile Acidity** - Amount of acetic acid in wine
3. **Citric Acid** - Adds freshness and flavor
4. **Residual Sugar** - Sugar remaining after fermentation
5. **Chlorides** - Amount of salt in wine
6. **Free Sulfur Dioxide** - Prevents microbial growth and oxidation
7. **Total Sulfur Dioxide** - Total amount of SO2
8. **Density** - Density of wine relative to water
9. **pH** - Acidity/alkalinity level
10. **Sulphates** - Wine additive contributing to SO2 levels
11. **Alcohol** - Alcohol percentage

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/LowAPMnoGFdev/LabExam1.git
   cd LabExam1
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app/app.py
   ```

5. **Open your browser** to `http://localhost:8501`

### Google Colab Deployment

Run the notebook `notebooks/wine_quality_debug.ipynb` in Google Colab for a cloud-based experience.

## 📊 Model Performance

- **Algorithm**: Random Forest Classifier
- **Training Data**: Red wine dataset with 1,599 samples
- **Features**: 11 chemical properties
- **Target**: Binary classification (Good/Not Good Quality)
- **Accuracy**: ~85% (varies based on train/test split)

## 📁 Project Structure

```
wine-quality-prediction/
├── app/
│   ├── app.py              # Main Streamlit application
│   ├── utils.py            # Utility functions
│   └── wine_model.pkl      # Trained model and scaler
├── data/
│   └── raw/
│       └── winequality-red.csv  # Raw dataset
├── notebooks/
│   └── wine_quality_debug.ipynb # Model training & debugging
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔬 Model Training

The model is trained using a Random Forest classifier with the following pipeline:

1. **Data Loading**: Load wine quality dataset
2. **Preprocessing**: Handle missing values, feature scaling
3. **Feature Engineering**: Binary classification (quality ≥7 = Good)
4. **Model Training**: Random Forest with optimized hyperparameters
5. **Evaluation**: Cross-validation and performance metrics
6. **Model Persistence**: Save trained model and scaler

## 🚀 Deployment Options

### Streamlit Cloud
1. Fork this repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy directly from GitHub

### Heroku
1. Create a `Procfile`:
   ```
   web: sh setup.sh && streamlit run app/app.py
   ```
2. Deploy to Heroku

### Docker
1. Build the container:
   ```bash
   docker build -t wine-quality-app .
   ```
2. Run the container:
   ```bash
   docker run -p 8501:8501 wine-quality-app
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [LowAPMnoGFdev](https://github.com/LowAPMnoGFdev)

## 🙏 Acknowledgments

- Wine quality dataset from UCI Machine Learning Repository
- Streamlit team for the amazing framework
- scikit-learn contributors

## 📞 Support

If you have any questions or issues, please:
1. Check the [Issues](https://github.com/LowAPMnoGFdev/LabExam1/issues) page
2. Create a new issue if needed
3. Contact the development team

---

**Made with ❤️ for wine quality assurance**