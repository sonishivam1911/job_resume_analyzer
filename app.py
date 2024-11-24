import streamlit as st
from main import (
    extract_text_from_pdf,
    split_into_sections,
    summarize_section,
    extract_keywords,
    initialize_summarizer,
)

st.title("📄 Research Paper Summarizer")
st.write("Upload a research paper in PDF format and select specific sections to summarize.")

# File uploader widget
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    try:
        # Model selection dropdown
        model_name = st.selectbox(
            "Select Summarization Model",
            options=[
                "t5-small",                      # Lightweight T5 model
                "sshleifer/distilbart-cnn-12-6", # Lightweight distilled BART model
                "facebook/bart-large-cnn",       # General-purpose summarization
                "google/pegasus-xsum",           # Concise summaries  
                "t5-base",                       # Balanced T5 model
            ],
            index=0  # Lightweight T5 model (lightweight model)
        )

        # Initialize selected summarizer
        with st.spinner(f"Loading {model_name}..."):
            summarizer = initialize_summarizer(model_name)

        # Extract text from uploaded PDF
        with st.spinner("Extracting text from PDF..."):
            full_text = extract_text_from_pdf(uploaded_file)

        # Split into predefined sections (with fallback for poorly formatted PDFs)
        with st.spinner("Detecting sections..."):
            sections = split_into_sections(full_text)  # Returns a dictionary

        # Display available sections and let users select which ones to summarize
        selected_sections = st.multiselect(
            "Select Sections to Summarize",
            options=list(sections.keys()),  # Use unique, uppercase section names
            default=["ABSTRACT"] if "ABSTRACT" in sections else list(sections.keys())
        )

        # Add summary length control
        summary_length = st.selectbox(
            "Select Summary Length",
            options=["Short", "Medium", "Long"],
            index=1  # Default to Medium
        )

        # Map user selection to model parameters
        length_params = {
            "Short": {"max_length": 50, "min_length": 20},
            "Medium": {"max_length": 150, "min_length": 50},
            "Long": {"max_length": 300, "min_length": 100},
        }

        if selected_sections:
            summaries = {}

            # Summarize selected sections with selected length parameters
            for section_name in selected_sections:
                if section_name in sections:
                    with st.spinner(f"Summarizing {section_name}..."):
                        summaries[section_name] = summarize_section(
                            summarizer,
                            section_name,
                            sections[section_name],
                            max_length=length_params[summary_length]["max_length"],
                            min_length=length_params[summary_length]["min_length"]
                        )

            # Display summaries
            st.subheader("Summaries")
            for section_name, summary in summaries.items():
                st.write(f"### {section_name}")
                st.write(summary)

        # Optionally extract keywords from the full paper or specific sections
        if st.checkbox("Extract Keywords"):
            with st.spinner("Extracting keywords..."):
                keywords = extract_keywords(full_text)

            # Display extracted keywords as a list or table
            st.subheader("Top Keywords")
            for keyword, score in keywords:
                st.write(f"- {keyword}: {score:.4f}")

    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {e}")
else:
    st.warning("Please upload a valid PDF.")