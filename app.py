import streamlit as st
import pandas as pd
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Setup & Configuration ---
st.set_page_config(page_title="Mitacs AI Matchmaker", layout="wide")
st.title("🎯 Mitacs Project Matchmaker (Advanced Matching)")

# Load the NLP Model (Cached)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Load and Process Projects (Cached)
@st.cache_data
def load_and_embed_projects():
    try:
        df = pd.read_csv("mitacs_projects.csv")
    except FileNotFoundError:
        return None, None, "Error: 'mitacs_projects.csv' not found."

    # Check for the newly requested columns
    target_columns = ['Project_Description', 'Student_Roles', 'Required_Skills', 'Project_Activities']
    missing_cols = [col for col in target_columns if col not in df.columns]
    
    if missing_cols:
        return df, None, f"Missing required columns in CSV: {', '.join(missing_cols)}"
        
    # Combine fields strategically. 
    # We put Required_Skills and Student_Roles first so they aren't cut off by token limits.
    df['Combined_Text'] = (
        "Required Skills: " + df['Required_Skills'].fillna('') + ". " +
        "Student Roles: " + df['Student_Roles'].fillna('') + ". " +
        "Activities: " + df['Project_Activities'].fillna('') + ". " +
        "Description: " + df['Project_Description'].fillna('')
    )
        
    # Pre-compute all project embeddings
    project_embeddings = model.encode(df['Combined_Text'].tolist())
    return df, project_embeddings, "Success"

# --- UI Components ---
st.markdown("### 1. Upload Your CV")
cv_file = st.file_uploader("Upload your CV (PDF)", type=['pdf'])

st.markdown("### 2. Set Your Preferences (Optional)")
user_keywords = st.text_input(
    "Any specific keywords you want the AI to focus on?",
    value="Artificial Intelligence, Machine Learning, Deep Learning, DevOps, Docker, CI/CD, AWS, Cloud Infrastructure, Python"
)

# --- Processing Logic ---
if st.button("Find My Top 10 Projects 🚀"):
    if cv_file:
        with st.spinner("Analyzing CV against Skills, Roles, and Activities..."):
            
            # 1. Load cached dataset and embeddings
            df, project_embeddings, status_msg = load_and_embed_projects()
            
            if project_embeddings is None:
                st.error(status_msg)
                st.stop()

            # 2. Parse the CV
            cv_text = ""
            with pdfplumber.open(cv_file) as pdf:
                for page in pdf.pages:
                    cv_text += page.extract_text() + "\n"
            
            # 3. Inject Custom Bias (Prepend to avoid truncation)
            final_cv_text = cv_text
            if user_keywords.strip():
                final_cv_text = f"PRIMARY SKILLS AND FOCUS: {user_keywords}\n\n" + cv_text
            
            # 4. Generate AI Embedding for the CV
            cv_embedding = model.encode([final_cv_text])
            
            # 5. Calculate Cosine Similarity
            similarities = cosine_similarity(cv_embedding, project_embeddings)[0]
            
            # 6. Rank the Top 10
            df['Match_Score'] = np.round(similarities * 100, 2)
            top_10 = df.sort_values(by='Match_Score', ascending=False).head(10)
            
            # --- Display Results ---
            st.success("Analysis Complete!")
            st.markdown("### 🏆 Your Top 10 Project Matches")
            
            # Display the specific columns you care about
            display_columns = [
                'Match_Score', 'Project_ID', 'Title', 
                'Required_Skills', 'Student_Roles', 'Project_Activities', 'Project_Description'
            ]
            
            # Only show columns that actually exist in the CSV to prevent crashes
            existing_columns = [col for col in display_columns if col in top_10.columns]
            
            st.dataframe(
                top_10[existing_columns],
                hide_index=True,
                use_container_width=True
            )
    else:
        st.warning("Please upload your CV to begin.")