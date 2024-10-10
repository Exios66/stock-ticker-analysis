# Stock Market Forecasting with GRU and LSTM Networks

       __  __                              _             
      |  \/  | ___  _ __ _ __ _____      _| |_ ___  _ __  
      | |\/| |/ _ \| '__| '__/ _ \ \ /\ / / __/ _ \| '_ \ 
      | |  | | (_) | |  | | | (_) \ V  V /| || (_) | | | |
      |_|  |_|\___/|_|  |_|  \___/ \_/\_/  \__\___/|_| |_|

                 M O R N I N G S T A R  
                  D E V E L O P M E N T S

Welcome to the **Stock Market Forecasting** project repository! This project utilizes **Gated Recurrent Units (GRUs)** and **Long Short-Term Memory (LSTM)** networks to predict stock prices. It demonstrates proficiency in deep learning applications for financial data, handling multi-step forecasting, and processing multiple stock datasets simultaneously.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
4. [Data Collection](#data-collection)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Development](#model-development)
    - [Building LSTM and GRU Models](#building-lstm-and-gru-models)
    - [Training the Models](#training-the-models)
7. [Model Evaluation](#model-evaluation)
8. [Deployment](#deployment)
    - [Streamlit App](#streamlit-app)
9. [Usage](#usage)
10. [Advanced Techniques](#advanced-techniques)
11. [Contributing](#contributing)
12. [License](#license)
13. [Contact](#contact)

---

## Project Overview

The goal of this project is to predict future stock prices using advanced deep learning models. By leveraging GRUs and LSTMs, the project aims to handle the complexities of financial time series data, perform multi-step forecasting, and manage datasets from multiple stocks concurrently.

**Key Features:**
- **Multi-Stock Forecasting:** Handle and predict multiple stock prices simultaneously.
- **Deep Learning Models:** Implement and compare LSTM and GRU networks.
- **Technical Indicators:** Incorporate Moving Averages (MA) and Relative Strength Index (RSI) for enriched feature sets.
- **Interactive Dashboard:** Deploy predictions through a user-friendly Streamlit application.

---

## Repository Structure

```bash
stock-market-forecasting/
├── data/
│   ├── raw/
│   │   └── AAPL.csv
│   │   └── MSFT.csv
│   │   └── GOOGL.csv
│   │   └── AMZN.csv
│   │   └── TSLA.csv
│   ├── processed/
│       └── AAPL_scaled.csv
│       └── MSFT_scaled.csv
│       └── GOOGL_scaled.csv
│       └── AMZN_scaled.csv
│       └── TSLA_scaled.csv
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Model_Evaluation.ipynb
├── src/
│   ├── data_processing.py
│   ├── model_building.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── utils.py
├── models/
│   ├── AAPL_lstm.h5
│   ├── AAPL_gru.h5
│   ├── MSFT_lstm.h5
│   ├── MSFT_gru.h5
│   └── ... # Additional trained models
├── app/
│   ├── app.py
│   └── requirements.txt
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

**Directory Breakdown:**

- **data/**: Contains both raw and processed datasets for each stock.
- **notebooks/**: Jupyter notebooks for exploratory data analysis, preprocessing, training, and evaluation.
- **src/**: Python scripts/modules for handling data processing, model building, training, and evaluation.
- **models/**: Saved trained models in `.h5` format.
- **app/**: Streamlit application for deploying the forecasting model.
- **requirements.txt**: Lists all Python dependencies.
- **README.md**: Project documentation (this file).
- **LICENSE**: License information.
- **.gitignore**: Specifies files and directories to ignore in Git.

---

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- **Python 3.7 or higher**
- **Git**
- **pip** (Python package installer)

### Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/stock-market-forecasting.git
    cd stock-market-forecasting
    ```

2. **Set Up Virtual Environment (Recommended)**

    It's advisable to use a virtual environment to manage project dependencies.

    - **Using `venv`:**

        ```bash
        python3 -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
        ```

    - **Using `conda`:**

        ```bash
        conda create -n stock-forecast python=3.8
        conda activate stock-forecast
        ```

3. **Install Dependencies**

    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **Note:** The `app/requirements.txt` contains dependencies specific to the Streamlit application.

4. **Download or Prepare Data**

    Ensure that the `data/raw/` directory contains the raw CSV files for each stock. If not present, you can download them using the provided scripts or APIs like Yahoo Finance.

    ```bash
    # Example using yfinance in a Python script
    python src/download_data.py
    ```

    *(Ensure to implement `download_data.py` to fetch and save the data accordingly.)*

---

## Data Collection

The project fetches historical stock data using the `yfinance` library, which interfaces with Yahoo Finance APIs.

**Selected Stocks:**

- Apple Inc. (`AAPL`)
- Microsoft Corporation (`MSFT`)
- Alphabet Inc. (`GOOGL`)
- Amazon.com Inc. (`AMZN`)
- Tesla Inc. (`TSLA`)

**Data Features:**

- Open, High, Low, Close, Adj Close, Volume
- Technical Indicators: Moving Averages (50-day) and Relative Strength Index (RSI)

**Data Source:**

- [Yahoo Finance](https://finance.yahoo.com/) via the `yfinance` Python library.

---

## Data Preprocessing

Data preprocessing ensures the dataset is clean, consistent, and suitable for model training.

### Steps:

1. **Handling Missing Values**

    - Checked for missing values across all datasets.
    - Applied forward fill and backward fill methods where necessary.

2. **Feature Engineering**

    - **Moving Averages (MA):** Calculated 50-day MA to capture trend.
    - **Relative Strength Index (RSI):** Calculated 14-day RSI as a momentum indicator.

3. **Normalization/Scaling**

    - Utilized **Min-Max Scaling** to scale features between 0 and 1.
    - Ensured scaling parameters are consistent across training, validation, and test sets to prevent data leakage.

4. **Data Splitting**

    - **Training Set:** 70%
    - **Validation Set:** 15%
    - **Test Set:** 15%
    - Employed time-based splitting to maintain temporal order.

5. **Creating Sequences**

    - Applied a sliding window approach with a sequence length of 60 days to predict the next day's closing price.

**Scripts Involved:**

- `data_processing.py`: Contains functions for data cleaning, feature engineering, scaling, and sequence creation.

---

## Model Development

The project implements both **LSTM** and **GRU** models to forecast stock prices.

### Building LSTM and GRU Models

**LSTM Model:**

- **Architecture:**
    - Two LSTM layers with 50 units each.
    - Dropout layers with a rate of 0.2 to prevent overfitting.
    - Dense output layer with linear activation.
- **Compilation:**
    - **Optimizer:** Adam
    - **Loss Function:** Mean Squared Error (MSE)
    - **Metrics:** Mean Absolute Error (MAE)

**GRU Model:**

- **Architecture:**
    - Two GRU layers with 50 units each.
    - Dropout layers with a rate of 0.2 to prevent overfitting.
    - Dense output layer with linear activation.
- **Compilation:**
    - **Optimizer:** Adam
    - **Loss Function:** Mean Squared Error (MSE)
    - **Metrics:** Mean Absolute Error (MAE)

**Script:**

- `model_building.py`: Defines functions to build and compile LSTM and GRU models.

### Training the Models

**Process:**

1. Split the preprocessed data into training, validation, and test sets.
2. Train both LSTM and GRU models using the training set.
3. Monitor performance on the validation set using Early Stopping to prevent overfitting.
4. Save the best-performing models.

**Script:**

- `train_models.py`: Handles the training loop, including Early Stopping callbacks and model saving.

---

## Model Evaluation

After training, models are evaluated on the test set using multiple performance metrics.

### Performance Metrics:

- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **Mean Absolute Percentage Error (MAPE)**
- **R² Score**

### Visualization:

- **Actual vs. Predicted Close Prices:** Line plots to visually assess model performance.
- **Residual Analysis:** Scatter plots of residuals to identify patterns.

**Script:**

- `evaluate_models.py`: Contains functions to calculate metrics and generate visualizations.

---

## Deployment

Deploy the trained models through an interactive Streamlit application, allowing users to input a stock ticker and view predictions.

### Streamlit App

**Features:**

- Select stock ticker from a dropdown menu.
- View historical closing prices.
- Display next-day predicted closing price.
- Visual comparison of actual vs. predicted prices.

**Files:**

- `app/app.py`: Main Streamlit application script.
- `app/requirements.txt`: Dependencies specific to the Streamlit app.

### Deployment Steps:

1. **Navigate to the `app/` Directory**

    ```bash
    cd app/
    ```

2. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit App**

    ```bash
    streamlit run app.py
    ```

4. **Access the App**

    The app will open in your default web browser at `http://localhost:8501`.

---

## Usage

### Running the Entire Pipeline

1. **Data Collection**

    ```bash
    python src/download_data.py
    ```

2. **Data Preprocessing**

    ```bash
    python src/data_processing.py
    ```

3. **Model Training**

    ```bash
    python src/train_models.py
    ```

4. **Model Evaluation**

    ```bash
    python src/evaluate_models.py
    ```

5. **Deploying the App**

    Refer to the [Deployment](#deployment) section.

### Streamlit Application

1. **Launch the App**

    ```bash
    cd app/
    streamlit run app.py
    ```

2. **Interact with the App**

    - Select a stock ticker.
    - View the historical closing prices.
    - Obtain the next-day predicted closing price.
    - Visual comparisons between actual and predicted prices.

---

## Advanced Techniques

To enhance model performance and robustness, consider implementing the following:

### Incorporating Attention Mechanisms

- **Purpose:** Allows the model to focus on specific parts of the input sequence, potentially improving prediction accuracy.
- **Implementation:** Add attention layers to the LSTM or GRU architectures.

### Ensemble Models

- **Purpose:** Combines predictions from multiple models to improve overall performance.
- **Implementation:** Average predictions from LSTM and GRU models or use a weighted ensemble approach.

### Transfer Learning

- **Purpose:** Leverages knowledge from one stock's behavior to improve predictions on another, especially if they share similar patterns.
- **Implementation:** Fine-tune pre-trained models on new stock data.

### Sentiment Analysis

- **Purpose:** Incorporate news headlines or social media sentiments to enrich the feature set.
- **Implementation:** Use NLP techniques to extract sentiment scores and include them as additional features.

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the Repository**

2. **Create a Feature Branch**

    ```bash
    git checkout -b feature/YourFeature
    ```

3. **Commit Your Changes**

    ```bash
    git commit -m "Add Your Feature"
    ```

4. **Push to the Branch**

    ```bash
    git push origin feature/YourFeature
    ```

5. **Create a Pull Request**

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

**Your Name**  
Data Scientist  
Email: [your.email@example.com](mailto:your.email@example.com)  
LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)  
GitHub: [github.com/yourusername](https://github.com/yourusername)

---

## Repository Tree Map

For a quick overview of the project structure, here's the repository tree map:

```bash
stock-market-forecasting/
├── data/
│   ├── raw/
│   │   ├── AAPL.csv
│   │   ├── MSFT.csv
│   │   ├── GOOGL.csv
│   │   ├── AMZN.csv
│   │   └── TSLA.csv
│   ├── processed/
│   │   ├── AAPL_scaled.csv
│   │   ├── MSFT_scaled.csv
│   │   ├── GOOGL_scaled.csv
│   │   ├── AMZN_scaled.csv
│   │   └── TSLA_scaled.csv
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Model_Evaluation.ipynb
├── src/
│   ├── data_processing.py
│   ├── download_data.py
│   ├── model_building.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── utils.py
├── models/
│   ├── AAPL_lstm.h5
│   ├── AAPL_gru.h5
│   ├── MSFT_lstm.h5
│   ├── MSFT_gru.h5
│   ├── GOOGL_lstm.h5
│   ├── GOOGL_gru.h5
│   ├── AMZN_lstm.h5
│   ├── AMZN_gru.h5
│   └── TSLA_lstm.h5
├── app/
│   ├── app.py
│   └── requirements.txt
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

**Description:**

- **data/raw/**: Contains raw CSV files downloaded from Yahoo Finance for each selected stock.
- **data/processed/**: Contains scaled and preprocessed CSV files ready for model training.
- **notebooks/**: Jupyter notebooks for each stage of the project:
    - **01_Data_Exploration.ipynb**: Exploratory Data Analysis (EDA)
    - **02_Data_Preprocessing.ipynb**: Data cleaning and feature engineering
    - **03_Model_Training.ipynb**: Building and training LSTM and GRU models
    - **04_Model_Evaluation.ipynb**: Evaluating model performance
- **src/**: Python scripts handling different functionalities:
    - **data_processing.py**: Functions for data cleaning, feature engineering, scaling, and sequence creation
    - **download_data.py**: Script to download stock data using `yfinance`
    - **model_building.py**: Functions to build LSTM and GRU models
    - **train_models.py**: Scripts to train the models with training and validation data
    - **evaluate_models.py**: Scripts to evaluate models on test data
    - **utils.py**: Utility functions used across multiple scripts
- **models/**: Saved trained models in `.h5` format for each stock and model type
- **app/**: Contains the Streamlit application:
    - **app.py**: Main Streamlit script to run the web app
    - **requirements.txt**: Dependencies for the Streamlit app
- **requirements.txt**: Lists all Python dependencies for the project
- **README.md**: Project documentation
- **LICENSE**: License information
- **.gitignore**: Specifies files and directories to ignore in Git tracking

---

## Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/) for providing accessible financial data.
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for deep learning frameworks.
- [Streamlit](https://streamlit.io/) for facilitating easy deployment of interactive web applications.

---

*This README was generated to provide comprehensive documentation for the Stock Market Forecasting project. For any questions or suggestions, feel free to reach out!*
