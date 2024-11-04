import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Load models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

# Main app
def main():
    # Custom CSS for styling
    st.markdown("""
        <style>
            /* Background color and fonts */
            .main {
                background: rgb(64,85,187);
                background: linear-gradient(18deg, rgba(64,85,187,1) 23%, rgba(141,135,206,1) 70%);
                font-family: 'Arial', sans-serif;
            }
            body {
                background-color: #f0f8ff; /* Light background */
                color: #333333;
            }

            /* Title and description styling */
            h1, h3, p {
                color: #ffffff;
                text-align: left;
                padding-left: 10px;
            }

            /* Uploader section */
            .stFileUploader label {
                color: #ffd700;
                font-weight: bold;
                font-size: 1.2em;
                text-align: left;
                display: block;
                padding-left: 10px;
            }

            /* Button with animation */
            .stButton button {
                background-color: #ff6347;
                color: white;
                font-size: 1.1em;
                font-weight: bold;
                border-radius: 12px;
                padding: 10px 20px;
                transition: background-color 0.3s ease, transform 0.2s ease;
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                margin-left: 10px;
            }
            .stButton button:hover {
                background-color: #ff4500;
                transform: translateY(-2px);
            }

            /* Animated box for results */
            .result-box {
                background-color: #ffddc1;
                color: #333;
                padding: 20px;
                border-radius: 12px;
                font-weight: bold;
                font-size: 1.3em;
                text-align: left;
                margin-top: 20px;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
                animation: fadeIn 1s ease-out;
                margin-left: 10px;
            }

            /* Fade-in animation */
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }

            /* Upload area styling */
            .stFileUploader div {
                background-color: #e0f7fa;
                padding: 10px;
                border-radius: 8px;
                text-align: left;
                margin-left: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Title and instructions with left alignment
    st.markdown("<h1>🌟 Resume Screening App</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Upload your resume to identify the predicted job category</h3>", unsafe_allow_html=True)
    st.markdown("<p>This app helps classify resumes into job categories based on content. Upload a resume in text or PDF format to begin.</p>", unsafe_allow_html=True)

    # File uploader with a styled label
    uploaded_file = st.file_uploader("Upload Resume", type=['txt', 'pdf'])

    # Prediction process with styled result box
    if uploaded_file is not None:
        with st.spinner("Analyzing your resume..."):
            try:
                resume_bytes = uploaded_file.read()
                resume_text = resume_bytes.decode('utf-8')
            except UnicodeDecodeError:
                resume_text = resume_bytes.decode('latin-1')

            # Clean and predict
            cleaned_resume = clean_resume(resume_text)
            input_features = tfidfd.transform([cleaned_resume])
            prediction_id = clf.predict(input_features)[0]

            # Category mapping
            category_mapping = {
                15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "Python Developer", 24: "Web Designing",
                12: "HR", 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer", 18: "Operations Manager", 6: "Data Science",
                22: "Sales", 16: "Mechanical Engineer", 1: "Arts", 7: "Database", 11: "Electrical Engineering",
                14: "Health and Fitness", 19: "PMO", 4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing",
                17: "Network Security Engineer", 21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate"
            }

            category_name = category_mapping.get(prediction_id, "Unknown")

            # Display results in a styled and animated box
            st.markdown("<div class='result-box'>🎯 Predicted Category: <strong>{}</strong></div>".format(category_name), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
