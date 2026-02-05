import streamlit as st
from fpdf import FPDF
from datetime import datetime
import os

st.title("📄 Generate Medical Report")

# ---------------- CHECK FLOW ----------------
if "patient" not in st.session_state or "diagnosis" not in st.session_state:
    st.warning("⚠️ Please complete patient registration and MRI diagnosis first.")
    st.stop()

patient = st.session_state["patient"]
diag = st.session_state["diagnosis"]

# ---------------- GENERATE PDF ----------------
if st.button("Generate Professional PDF Report"):

    os.makedirs("reports/history", exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # -------- HEADER --------
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AI-Powered Brain MRI Diagnostic Report", ln=True, align="C")

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, "Department of Radiology | Doctor Portal System", ln=True, align="C")

    pdf.ln(8)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)

    # -------- PATIENT DETAILS --------
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Patient Information", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Patient Name: {patient['name']}", ln=True)
    pdf.cell(0, 8, f"Age: {patient['age']}", ln=True)
    pdf.cell(0, 8, f"Gender: {patient['gender']}", ln=True)
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%d %B %Y')}", ln=True)

    pdf.ln(8)

    # -------- FINDINGS --------
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Diagnostic Findings", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(
        0, 8,
        f"The uploaded brain MRI scan was analyzed using an AI-based deep learning "
        f"system. The model predicts the presence of a {diag['tumor']} tumor with "
        f"a confidence score of {diag['confidence']:.2f}%. "
        f"The classification model was trained using GAN-augmented MRI data."
    )

    pdf.ln(8)

    # -------- SEGMENTATION IMAGE --------
    if "segmentation_image" in st.session_state and os.path.exists(st.session_state["segmentation_image"]):

        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 8, "Tumor Segmentation Result", ln=True)
        pdf.ln(5)

        # Center image
        pdf.image(
            st.session_state["segmentation_image"],
            x=30,
            w=150
        )

        pdf.ln(8)

    # -------- CONCLUSION --------
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Conclusion", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(
        0, 8,
        "The AI-based analysis indicates abnormal tissue regions consistent with a "
        "brain tumor. The segmentation highlights the suspected tumor location to "
        "support clinical interpretation. This report is intended as a decision-"
        "support tool and should be reviewed by a qualified medical professional."
    )

    pdf.ln(10)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 8, "This is a computer-generated report and does not require a signature.", ln=True)

    # -------- SAVE --------
    path = f"reports/history/{patient['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(path)

    st.success("✅ Professional Medical Report Generated")
    st.download_button(
        "⬇️ Download Report",
        open(path, "rb"),
        file_name=os.path.basename(path)
    )
