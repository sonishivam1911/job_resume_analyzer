import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
from wordcloud import WordCloud  # For generating word clouds
import matplotlib.pyplot as plt  # For displaying the word cloud

# Function to initialize the summarization pipeline dynamically
def initialize_summarizer(model_name="facebook/bart-large-cnn"):
    """
    Initialize the Hugging Face summarization pipeline with the selected model.
    :param model_name: Name of the model to use for summarization.
    :return: Summarization pipeline object.
    """
    return pipeline("summarization", model=model_name)

# Function to extract text from a PDF
def extract_text_from_pdf(uploaded_file):
    """
    Extract text from a PDF file uploaded via Streamlit's st.file_uploader.
    :param uploaded_file: Streamlit UploadedFile object.
    :return: Full text extracted from the PDF.
    """
    pdf_bytes = uploaded_file.read()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        full_text = ""
        for page in doc:
            full_text += page.get_text()
    return full_text

# Function to split text into sections
def split_into_sections(text):
    """
    Splits the text into predefined sections based on headings.
    Ensures all relevant sections are captured and handles duplicates by appending content.
    :param text: Full text of the research paper.
    :return: Dictionary with section names as keys and combined content as values.
    """
    section_headings = [
        r'\bABSTRACT\b', r'\bINTRODUCTION\b', r'\bBACKGROUND\b',
        r'\bMETHODS?\b', r'\bMATERIALS AND METHODS\b',
        r'\bRESULTS?\b', r'\bDISCUSSION\b',
        r'\bCONCLUSION\b', r'\bREFERENCES?\b',
        r'\bACKNOWLEDGMENTS?\b', r'\bKEYWORDS?\b',
        r'\bREVIEW OF LITERATURE\b', r'\bRELATED WORK\b',
        r'\bLIMITATIONS?\b', r'\bFUTURE WORK\b',
        r'\bAPPENDICES?\b', r'\bFIGURES AND TABLES\b',
        r'\bETHICS STATEMENT\b',
        r'\bFUNDING STATEMENT\b',
        r'\bCONFLICT OF INTEREST STATEMENT\b'
    ]
    
    pattern = "|".join(section_headings)
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    
    sections = {}
    for i, match in enumerate(matches):
        start_idx = match.start()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_name = match.group().strip().upper()
        section_content = text[start_idx:end_idx].strip()
        
        if section_name in sections:
            sections[section_name] += "\n" + section_content
        else:
            sections[section_name] = section_content
    
    return sections

# Function to summarize a specific section
def summarize_section(summarizer, section_name, section_content, max_length=150, min_length=50):
    """
    Summarizes a specific section of the research paper using the selected model.
    :param summarizer: Summarization pipeline object.
    :param section_name: Name of the section (e.g., Abstract).
    :param section_content: Content of the section.
    :param max_length: Maximum length of the summary.
    :param min_length: Minimum length of the summary.
    :return: Summarized text for the section.
    """
    if len(section_content.split()) > 500:
        chunks = [section_content[i:i + 1000] for i in range(0, len(section_content), 1000)]
        summarized_chunks = [
            summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            for chunk in chunks
        ]
        summary = " ".join(summarized_chunks)
    else:
        summary = summarizer(section_content, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    
    return summary

# Function to extract keywords using TF-IDF
def extract_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    
    keywords = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray().flatten()
    
    sorted_indices = scores.argsort()[::-1]
    top_keywords = [(keywords[i], scores[i]) for i in sorted_indices[:top_n]]
    
    return top_keywords

# Function to generate a word cloud from extracted keywords
def generate_word_cloud(keywords):
    keyword_dict = {word: score for word, score in keywords}
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keyword_dict)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Keyword Word Cloud", fontsize=16)
    plt.show()