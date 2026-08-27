import streamlit as st
import pandas as pd
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Setup & Configuration ---
st.set_page_config(page_title="Mitacs AI Matchmaker", layout="wide")
st.title("🎯 Mitacs Project Matchmaker")

# Load the NLP Model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- UI Components ---
st.markdown("### 1. Upload Your CV")
cv_file = st.file_uploader("Upload your CV (PDF)", type=['pdf'])

st.markdown("### 2. Set Your Preferences (Optional)")
# Defaults to your AI/DevOps focus, but your friends can backspace and type their own!
user_keywords = st.text_input(
    "Any specific keywords you want the AI to focus on?",
    value="Artificial Intelligence, Machine Learning, Deep Learning, DevOps, Docker, CI/CD, AWS, Cloud Infrastructure, Python"
)

# --- Processing Logic ---
if st.button("Find My Top 10 Projects 🚀"):
    if cv_file:
        with st.spinner("Analyzing CV and calculating semantic matches..."):
            
            # 1. Parse the CV
            cv_text = ""
            with pdfplumber.open(cv_file) as pdf:
                for page in pdf.pages:
                    cv_text += page.extract_text() + "\n"
            
            # 2. Inject Custom Bias
            if user_keywords.strip():
                cv_text += f"\nPRIMARY FOCUS AND PREFERENCES: {user_keywords}"
            
            # 3. Read the default Projects CSV right from the repository
            try:
                df = pd.read_csv("mitacs_projects.csv")
            except FileNotFoundError:
                st.error("Error: 'mitacs_projects.csv' not found. Please make sure it is uploaded to GitHub.")
                st.stop()
            
            # Safely extract specific fields
            if 'Description' in df.columns:
                df['Supervisor'] = df['Description'].str.extract(r'Faculty supervisor:\s*([^\n]*)')
                df['Location'] = df['Description'].str.extract(r'Project Location:\s*([^\n]*)')
                df['Language'] = df['Description'].str.extract(r'Language:\s*([^\n]*)')
                df['Start_Date'] = df['Description'].str.extract(r'Preferred start date:\s*([^\n]*)')
                df['Combined_Text'] = df['Title'].fillna('') + " " + df['Description'].fillna('')
            else:
                st.error("The CSV must contain a 'Description' column.")
                st.stop()
            
            # 4. Generate AI Embeddings
            cv_embedding = model.encode([cv_text])
            project_embeddings = model.encode(df['Combined_Text'].tolist())
            
            # 5. Calculate Cosine Similarity
            similarities = cosine_similarity(cv_embedding, project_embeddings)[0]
            
            # 6. Rank the Top 10
            df['Match_Score'] = np.round(similarities * 100, 2)
            top_10 = df.sort_values(by='Match_Score', ascending=False).head(10)
            
            # --- Display Results ---
            st.success("Analysis Complete!")
            st.markdown("### 🏆 Your Top 10 Project Matches")
            
            display_columns = [
                'Match_Score', 'Project_ID', 'Title', 
                'Supervisor', 'Location', 'Language', 'Start_Date', 'Description'
            ]
            existing_columns = [col for col in display_columns if col in top_10.columns]
            
            st.dataframe(
                top_10[existing_columns],
                hide_index=True,
                use_container_width=True
            )
    else:
        st.warning("Please upload your CV to begin.")