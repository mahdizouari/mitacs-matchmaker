import streamlit as st
import pandas as pd
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Setup & Configuration ---
st.set_page_config(page_title="Mitacs AI Matchmaker", layout="wide")
st.title("🎯 Mitacs Project Matchmaker (AI & DevOps Focus)")

# Load the NLP Model (cached so it doesn't reload on every button click)
@st.cache_resource
def load_model():
    # all-MiniLM-L6-v2 is fast, free, and highly accurate for semantic matching
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- UI Components ---
st.markdown("### 1. Upload Your Data")
col1, col2 = st.columns(2)

with col1:
    cv_file = st.file_uploader("Upload your CV (PDF)", type=['pdf'])

with col2:
    projects_file = st.file_uploader("Upload Mitacs Projects (CSV)", type=['csv'])
    st.caption("Ensure your CSV has these columns: Project_ID, Title, Description")

# --- Processing Logic ---
if st.button("Find My Top 10 Projects 🚀"):
    if cv_file and projects_file:
        with st.spinner("Analyzing CV and calculating semantic matches..."):
            
            # 1. Parse the CV
            cv_text = ""
            with pdfplumber.open(cv_file) as pdf:
                for page in pdf.pages:
                    cv_text += page.extract_text() + "\n"
            
            # 2. Inject AI & DevOps Bias
            # This forces the AI vector to pivot away from ASP.NET/DW and prioritize your actual interests.
            target_focus = """
            PRIMARY FOCUS: Artificial Intelligence, Machine Learning, Deep Learning, 
            DevOps, Docker, CI/CD, AWS, Cloud Infrastructure, Python, Algorithmic Trading.
            """
            cv_text_biased = cv_text + "\n" + target_focus
            
            # 3. Read the Projects CSV and Extract Metadata
            df = pd.read_csv(projects_file)
            
            # Extract specific fields from the Description block into their own columns
            df['Supervisor'] = df['Description'].str.extract(r'Faculty supervisor:\s*([^\n]*)')
            df['Location'] = df['Description'].str.extract(r'Project Location:\s*([^\n]*)')
            df['Language'] = df['Description'].str.extract(r'Language:\s*([^\n]*)')
            df['Start_Date'] = df['Description'].str.extract(r'Preferred start date:\s*([^\n]*)')
            
            df['Combined_Text'] = df['Title'].fillna('') + " " + df['Description'].fillna('')
            
            # 4. Generate AI Embeddings
            cv_embedding = model.encode([cv_text_biased])
            project_embeddings = model.encode(df['Combined_Text'].tolist())
            
            # 5. Calculate Cosine Similarity
            similarities = cosine_similarity(cv_embedding, project_embeddings)[0]
            
            # 6. Rank the Top 10
            df['Match_Score'] = np.round(similarities * 100, 2)
            top_10 = df.sort_values(by='Match_Score', ascending=False).head(10)
            
            # --- Display Results ---
            st.success("Analysis Complete!")
            st.markdown("### 🏆 Your Top 10 Project Matches")
            
            # Display beautifully using Streamlit dataframe with the new columns
            display_columns = [
                'Match_Score', 'Project_ID', 'Title', 
                'Supervisor', 'Location', 'Language', 'Start_Date', 'Description'
            ]
            
            # Only display columns that actually exist in the CSV to prevent errors
            existing_columns = [col for col in display_columns if col in top_10.columns]
            
            st.dataframe(
                top_10[existing_columns],
                hide_index=True,
                use_container_width=True
            )
    else:
        st.warning("Please upload both your CV and the Projects CSV to begin.")