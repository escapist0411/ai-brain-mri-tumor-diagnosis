import streamlit as st

st.set_page_config(page_title="Doctor Portal", layout="wide")

st.markdown(
    """
    <h1 style='color:#0b5ed7;'>🧠 AI Brain MRI Diagnosis System</h1>
    <h3 style='color:#1f6f8b;'>Doctor Portal Dashboard</h3>
    <hr>
    <p>
    This system assists doctors in detecting and analyzing brain tumors
    using AI-powered deep learning models.
    </p>
    """,
    unsafe_allow_html=True
)

st.info("👈 Use the left sidebar to navigate between modules.")

st.markdown("""
### 🔹 Available Modules
- Patient Registration
- MRI Diagnosis
- Report Generation
- Report History
""")
