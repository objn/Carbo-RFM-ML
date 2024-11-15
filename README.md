# Carbo-RFM-ML

This project provides a dashboard to view charts, key performance indicators (KPIs), and a machine learning module to predict which customers should receive specific promotional discounts.

## Prerequisites

- **Python**: Version 3.10.9 - 3.10.12 (tested and recommended versions)

## Installation & Setup

Follow these steps to set up and run the project:

[(Optional) Colab version](https://colab.research.google.com/drive/1Bdjs4kvNF_i3yfmL1RpbXCvZn-BwzMA1#scrollTo=ZAuCvhgkcYCS)

1. **Install Python**  
   Make sure Python 3.10.9 - 3.10.12 is installed on your system. You can download Python from the official [Python website](https://www.python.org/downloads/).

2. **Create a Virtual Environment (venv)**  
   In your project directory, create a virtual environment by running:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux

   pip install -r requirements.txt
   
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```
View the Dashboard & KPIs
  ```bash
   python chart.py
  ```
![Dashboard](https://github.com/user-attachments/assets/e9502f4f-af14-40dd-9128-0ebfbcd4aef1)
![Dashboard2](https://github.com/user-attachments/assets/5a7b261a-a36a-452d-8201-110eac6fe343)

Run the Machine Learning Dashboard
  ```bash
   python ML.py
  ```
![ML](https://github.com/user-attachments/assets/3812ff80-93fd-4058-8f07-483cccfbee85)
![result_predict](https://github.com/user-attachments/assets/e9c47d10-d2d7-4337-99ab-7f6a9c01e69f)
