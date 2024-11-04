# Resume Classification System: Automating the Hiring Process

A NLP tool that streamlines the recruitment process by analyzing candidate resumes against job descriptions to identify the best-fit candidates efficiently.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)

## Introduction

In today's fast-paced job market, finding the right talent can be overwhelming. The Automated Resume Screening System empowers recruiters by harnessing the power of machine learning to transform the hiring process, enabling them to discover top candidates efficiently and fairly. This innovative solution not only saves time but also promotes diversity and inclusivity, ensuring that every candidate gets the opportunity they deserve.

## Features

- Automated Screening: Efficiently filters resumes based on job descriptions, reducing manual effort and speeding up the hiring process.
- User-Friendly Interface: Offers an intuitive dashboard for recruiters to easily upload job descriptions and review candidate profiles.
- Keyword and Skill Matching: Analyzes resumes for relevant keywords and skills, ensuring candidates meet the essential criteria for the role.

## Requirements

Make sure you have the following Python packages installed:

- `numpy`: A library for numerical computations and array manipulations.
- `pandas`: A powerful data manipulation and analysis library.
- `matplotlib`: A library for creating static, animated, and interactive visualizations in Python.
- `seaborn`: A statistical data visualization library based on Matplotlib.
- `scikit-learn`: A machine learning library that provides simple and efficient tools for data mining and data analysis.
- `streamlit`: A framework for building interactive web applications, especially useful for data science projects.
- `nltk`: The Natural Language Toolkit, a library for working with human language data (text).

You can install these packages using the `requirements.txt` file:

```sh
pip install -r requirements.txt
```

## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Hiral25p/NLP-sem7
   ```
2. Change your working directory to the project folder:
   ```sh
   cd NLP-sem7
   ```
   _Ensure that you have the required packages installed (see the "Requirements" section)._
3. Run the program:
   ```sh
   streamlit run app.py
   ```
   Or
    ```sh
   python -m streamlit run app.py
   ```
    
## Usage

- Upload your document files (e.g., PDFs, Word documents) through the provided interface for processing.
- Wait for the application to finish analyzing your resume.
- Once the analysis is complete, the application will provide a final output indicating which position the resume is most suited for.

## File Structure

- app.py: The main Streamlit application script that handles user interaction and resume analysis.
- requirements.txt: A list of required Python packages for the project.
- model.pkl: The trained machine learning model used for analyzing resumes.
- clf.pkl: The trained classifier model used for analyzing resumes.
- tfidf.pkl: The fitted TF-IDF vectorizer used for transforming the text data.
- Resume Screening with Python.ipynb: Jupyter notebook containing the analysis and development of the resume screening model.
- UpdatedResumeDataSet.csv: Dataset used for training and evaluating the resume screening model.
