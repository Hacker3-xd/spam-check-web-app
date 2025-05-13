# Spam Classifier Web App

A web-based application that classifies text messages as Spam or Not Spam using Machine Learning. Built with Streamlit and scikit-learn.

## Features

- 📊 Data exploration and visualization
- 🤖 Machine learning model training (Naive Bayes)
- 🧪 Real-time spam classification
- 📈 Model performance metrics
- 🎨 Interactive and user-friendly interface

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd spam-classifier-app
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
- Place the `spam.csv` file in the project root directory
- You can download the dataset from [here](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

## Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

## Usage

1. **Home Page**: Overview of the application
2. **Data Exploration**: View dataset statistics and word clouds
3. **Model Training**: Train the spam classifier and view performance metrics
4. **Spam Checker**: Input messages to check if they are spam or not

## Project Structure

```
spam-classifier-app/
│
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── spam.csv             # Dataset
└── README.md            # This file
```

## Technologies Used

- Python
- Streamlit
- scikit-learn
- NLTK
- pandas
- matplotlib
- seaborn
- wordcloud

## License

This project is licensed under the MIT License - see the LICENSE file for details.