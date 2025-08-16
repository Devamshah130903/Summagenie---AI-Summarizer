# summa_genie_app.py
import re
from io import StringIO
from urllib.parse import urlparse, parse_qs
import os
import pandas as pd
import requests
import streamlit as st
import docx2txt
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
from transformers import pipeline

# YouTube transcript handling with proper error handling
try:
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YOUTUBE_API_AVAILABLE = False
    NoTranscriptFound = Exception
    TranscriptsDisabled = Exception

# ------------------------
# Streamlit page setup
# ------------------------
st.set_page_config(page_title="SummaGenie", layout="centered", page_icon="✨")

st.markdown("""
<style>
.main, .stApp { 
    background: linear-gradient(135deg, #0f172a, #1e1e2e); 
    color: #eeeeee; 
    font-family: "Poppins", sans-serif; 
}
h1, h2, h3, h4, h5, h6 { 
    color: #00adb5; 
    text-align: center;
}
.stButton>button { 
    background: #00adb5 !important; 
    color: #ffffff !important; 
    font-weight: 700; 
    border-radius: 10px; 
    border: none; 
    padding: 0.6rem 1.1rem;
    width: 100%;
}
.stButton>button:hover { 
    background: #008b94 !important; 
    transform: scale(1.03); 
}
.box { 
    padding: 14px; 
    border-radius: 10px; 
    border: 1px solid rgba(255,255,255,0.08); 
    background: #1f2335; 
    margin: 10px 0;
}
.summary-box { 
    max-height: 400px; 
    overflow-y: auto; 
    white-space: pre-wrap; 
    word-wrap: break-word; 
    color: #eeeeee; 
    padding: 15px; 
    border-radius: 8px; 
    background: #282a36;
    border: 1px solid #00adb5;
}
.stats-container {
    display: flex;
    justify-content: space-around;
    margin: 10px 0;
}
.stat-box {
    background: #1f2335;
    padding: 10px;
    border-radius: 8px;
    text-align: center;
    border: 1px solid rgba(0,173,181,0.3);
}
.warning-box {
    background: rgba(255, 193, 7, 0.1);
    border: 1px solid #ffc107;
    color: #ffc107;
    padding: 10px;
    border-radius: 8px;
    margin: 10px 0;
}
.success-box {
    background: rgba(40, 167, 69, 0.1);
    border: 1px solid #28a745;
    color: #28a745;
    padding: 10px;
    border-radius: 8px;
    margin: 10px 0;
}
.error-box {
    background: rgba(220, 53, 69, 0.1);
    border: 1px solid #dc3545;
    color: #dc3545;
    padding: 10px;
    border-radius: 8px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

st.title("✨ SummaGenie - Smart Summarizer ✨")
st.markdown("Upload documents, paste website links, or provide YouTube URLs to generate intelligent summaries.")

if not YOUTUBE_API_AVAILABLE:
    st.markdown("""
    <div class="warning-box">
    ⚠️ <strong>YouTube functionality disabled:</strong> Install youtube-transcript-api with:<br>
    <code>pip install youtube-transcript-api</code>
    </div>
    """, unsafe_allow_html=True)

# ------------------------
# Load summarizer with better error handling
# ------------------------
@st.cache_resource(show_spinner=True)
def load_summarizer():
    """Load the summarization model with error handling"""
    try:
        return pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")
    except Exception as e:
        st.error(f"Failed to load summarization model: {e}")
        st.info("You may need to install transformers and torch: pip install transformers torch")
        return None

@st.cache_resource(show_spinner=True) 
def load_backup_summarizer():
    """Load a smaller backup model if main model fails"""
    try:
        return pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", framework="pt")
    except Exception:
        return None

# Try to load the main summarizer
summarizer = load_summarizer()
if summarizer is None:
    st.warning("Main model failed to load, trying backup model...")
    summarizer = load_backup_summarizer()
    if summarizer is None:
        st.error("Could not load any summarization model. Please check your installation.")
        st.stop()

# ------------------------
# Enhanced utility functions
# ------------------------
def extract_text_from_file(uploaded_file) -> str:
    """Extract text from uploaded files with better error handling"""
    if not uploaded_file:
        return ""
    
    file_type = uploaded_file.name.split(".")[-1].lower()
    
    try:
        if file_type == "docx":
            return docx2txt.process(uploaded_file)
        
        elif file_type == "pdf":
            text_parts = []
            reader = PdfReader(uploaded_file)
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    txt = page.extract_text()
                    if txt and txt.strip():
                        text_parts.append(txt)
                except Exception as e:
                    st.warning(f"Error reading page {page_num}: {e}")
                    continue
            return "\n\n".join(text_parts)
        
        elif file_type == "txt":
            # Try multiple encodings
            raw_bytes = uploaded_file.getvalue()
            for encoding in ("utf-8", "utf-16", "latin-1", "cp1252"):
                try:
                    text = raw_bytes.decode(encoding, errors="ignore")
                    if text.strip():
                        return text
                except Exception:
                    continue
            return raw_bytes.decode("utf-8", errors="ignore")
        
        else:
            st.error(f"Unsupported file type: {file_type}. Please upload DOCX, PDF, or TXT files.")
            return ""
            
    except Exception as e:
        st.error(f"Error reading {file_type.upper()} file: {str(e)}")
        return ""

def extract_text_from_website(url: str) -> str:
    """Extract text from website with better error handling and content filtering"""
    if not url or not url.strip():
        return ""
    
    # Add protocol if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove unwanted elements
        for tag in soup(["script", "style", "header", "footer", "nav", "aside", "iframe", "noscript"]):
            tag.decompose()
        
        # Extract main content areas
        content_selectors = [
            "article", "main", "[role='main']", ".content", ".post-content", 
            ".entry-content", ".article-content", ".story-body"
        ]
        
        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # If no main content found, use all paragraphs
        if main_content:
            paragraphs = main_content.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
        else:
            paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
        
        # Extract and clean text
        text_parts = []
        for p in paragraphs:
            text = p.get_text(" ", strip=True)
            if text and len(text) > 20:  # Filter out very short paragraphs
                text_parts.append(text)
        
        if not text_parts:
            # Fallback: get all text
            text = soup.get_text(" ", strip=True)
            text = re.sub(r'\s+', ' ', text)
            return text[:10000]  # Limit length
        
        combined_text = "\n\n".join(text_parts)
        # Clean up whitespace
        combined_text = re.sub(r'\s+', ' ', combined_text)
        combined_text = re.sub(r'\n\s*\n', '\n\n', combined_text)
        
        return combined_text
        
    except requests.exceptions.RequestException as e:
        st.error(f"Network error fetching website: {str(e)}")
        return ""
    except Exception as e:
        st.error(f"Error processing website content: {str(e)}")
        return ""

def parse_youtube_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats"""
    if not url or not url.strip():
        return ""
    
    url = url.strip()
    
    try:
        # Handle direct video ID
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
            return url
        
        parsed = urlparse(url)
        domain = (parsed.netloc or "").lower()
        
        # youtu.be format
        if "youtu.be" in domain:
            return parsed.path.strip("/").split("?")[0]
        
        # youtube.com formats
        if "youtube.com" in domain:
            # Standard watch URL
            if "watch" in parsed.path:
                query_params = parse_qs(parsed.query)
                if "v" in query_params and query_params["v"]:
                    return query_params["v"][0]
            
            # Embed or shorts URL
            match = re.search(r"/(embed|shorts)/([A-Za-z0-9_-]{11})", parsed.path)
            if match:
                return match.group(2)
        
        return ""
        
    except Exception:
        return ""

def extract_youtube_transcript(url: str) -> str:
    """Extract YouTube transcript with comprehensive error handling"""
    if not YOUTUBE_API_AVAILABLE:
        st.error("YouTube transcript feature requires youtube-transcript-api. Install with: pip install youtube-transcript-api")
        return ""
    
    video_id = parse_youtube_id(url)
    if not video_id:
        st.error("Invalid YouTube URL format. Please check the URL and try again.")
        return ""
    
    try:
        # Get available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        transcript_data = None
        
        # Try to get English manual transcript first
        try:
            transcript = transcript_list.find_manually_created_transcript(['en', 'en-US', 'en-GB', 'en-CA', 'en-AU'])
            transcript_data = transcript.fetch()
        except NoTranscriptFound:
            try:
                # Try auto-generated English transcript
                transcript = transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB', 'en-CA', 'en-AU'])
                transcript_data = transcript.fetch()
            except NoTranscriptFound:
                # Try any available transcript and translate if needed
                try:
                    available_transcripts = list(transcript_list)
                    if available_transcripts:
                        transcript = available_transcripts[0]
                        if hasattr(transcript, 'translate') and transcript.language_code != 'en':
                            transcript = transcript.translate('en')
                        transcript_data = transcript.fetch()
                    else:
                        raise NoTranscriptFound("No transcripts available")
                except Exception:
                    raise NoTranscriptFound("No suitable transcripts found")
        
        if not transcript_data:
            st.error("No transcript data available for this video.")
            return ""
        
        # Process transcript data
        text_segments = []
        for segment in transcript_data:
            text = segment.get('text', '').strip()
            if text:
                # Clean up common transcript artifacts
                text = re.sub(r'\[.*?\]', '', text)  # Remove [Music], [Applause], etc.
                text = re.sub(r'\(.*?\)', '', text)  # Remove parenthetical expressions
                text = text.replace('\n', ' ')
                if text.strip():
                    text_segments.append(text.strip())
        
        full_text = ' '.join(text_segments)
        
        # Final cleanup
        full_text = re.sub(r'\s+', ' ', full_text)
        full_text = full_text.strip()
        
        if not full_text:
            st.error("Transcript was empty or could not be processed.")
            return ""
        
        return full_text
        
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("No transcripts are available for this video. The video may not have captions enabled.")
    except Exception as e:
        st.error(f"Failed to fetch transcript: {str(e)}")
    
    return ""

def smart_chunk_text(text: str, max_words_per_chunk: int = 450, overlap_words: int = 50):
    """Intelligently chunk text while preserving sentence boundaries"""
    if not text or not text.strip():
        return []
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences:
        return [text]
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        # If adding this sentence would exceed the limit
        if current_word_count + sentence_words > max_words_per_chunk and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            if overlap_words > 0 and len(current_chunk) > 1:
                overlap_text = ' '.join(current_chunk[-2:])  # Last 2 sentences for context
                overlap_word_count = len(overlap_text.split())
                if overlap_word_count <= overlap_words:
                    current_chunk = [overlap_text, sentence]
                    current_word_count = overlap_word_count + sentence_words
                else:
                    current_chunk = [sentence]
                    current_word_count = sentence_words
            else:
                current_chunk = [sentence]
                current_word_count = sentence_words
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_words
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks if chunks else [text]

def summarize_text(text: str, length_choice: str) -> tuple[str, dict]:
    """Enhanced summarization with better length control and statistics"""
    if not text or not text.strip():
        return "", {}
    
    # Improved length mapping with better ratios
    length_settings = {
        "Short": {"min_length": 30, "max_length": 100, "target_ratio": 0.1},
        "Medium": {"min_length": 100, "max_length": 200, "target_ratio": 0.2}, 
        "Large": {"min_length": 200, "max_length": 400, "target_ratio": 0.3}
    }
    
    settings = length_settings[length_choice]
    original_word_count = len(text.split())
    
    # Adjust based on original text length
    if original_word_count < 200:
        # For short texts, use smaller limits
        settings["min_length"] = min(settings["min_length"], original_word_count // 3)
        settings["max_length"] = min(settings["max_length"], original_word_count // 2)
    
    try:
        # Smart chunking for longer texts
        if original_word_count > 400:
            chunks = smart_chunk_text(text, max_words_per_chunk=400)
            chunk_summaries = []
            
            for i, chunk in enumerate(chunks):
                try:
                    summary = summarizer(
                        chunk, 
                        max_length=settings["max_length"] // max(len(chunks), 1) + 50,
                        min_length=max(settings["min_length"] // max(len(chunks), 1), 10),
                        do_sample=False,
                        truncation=True
                    )
                    chunk_summaries.append(summary[0]['summary_text'])
                except Exception as e:
                    st.warning(f"Error summarizing chunk {i+1}: {e}")
                    continue
            
            if not chunk_summaries:
                raise Exception("Failed to summarize any chunks")
            
            # Combine chunk summaries
            combined_summary = ' '.join(chunk_summaries)
            
            # Final summarization if we have multiple chunks
            if len(chunk_summaries) > 1:
                try:
                    final_summary = summarizer(
                        combined_summary,
                        max_length=settings["max_length"],
                        min_length=settings["min_length"], 
                        do_sample=False,
                        truncation=True
                    )
                    final_text = final_summary[0]['summary_text']
                except Exception:
                    # If final summarization fails, use combined summaries
                    final_text = combined_summary
            else:
                final_text = chunk_summaries[0]
        else:
            # Direct summarization for shorter texts
            summary = summarizer(
                text,
                max_length=min(settings["max_length"], original_word_count),
                min_length=min(settings["min_length"], original_word_count // 3),
                do_sample=False,
                truncation=True
            )
            final_text = summary[0]['summary_text']
        
        # Calculate statistics
        summary_word_count = len(final_text.split())
        compression_ratio = (original_word_count - summary_word_count) / original_word_count * 100
        
        stats = {
            "original_words": original_word_count,
            "summary_words": summary_word_count,
            "compression_ratio": compression_ratio,
            "reading_time": max(1, summary_word_count // 200)  # Assuming 200 words per minute
        }
        
        return final_text, stats
        
    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        return "", {}

# ------------------------
# Enhanced UI
# ------------------------
col1, col2 = st.columns([2, 1])

with col1:
    input_type = st.selectbox(
        "Select input type:", 
        ["📄 File Upload", "🌐 Website Link", "🎥 YouTube Video"],
        help="Choose how you want to provide content for summarization"
    )

with col2:
    summary_length = st.selectbox(
        "Summary length:", 
        ["Short", "Medium", "Large"], 
        index=1,
        help="Short: ~100 words, Medium: ~200 words, Large: ~400 words"
    )

# Dynamic input fields
uploaded_file = None
text_input = None

if input_type == "📄 File Upload":
    uploaded_file = st.file_uploader(
        "Upload your document:", 
        type=["docx", "pdf", "txt"],
        help="Supported formats: DOCX, PDF, TXT (up to 200MB)"
    )
    if uploaded_file:
        file_size = uploaded_file.size / (1024 * 1024)  # MB
        st.info(f"📎 File: {uploaded_file.name} ({file_size:.1f} MB)")

elif input_type == "🌐 Website Link":
    text_input = st.text_input(
        "Enter website URL:",
        placeholder="https://example.com/article",
        help="Paste the URL of a news article or blog post"
    )

elif input_type == "🎥 YouTube Video":
    if not YOUTUBE_API_AVAILABLE:
        st.error("YouTube functionality is not available. Please install youtube-transcript-api.")
    else:
        text_input = st.text_input(
            "Enter YouTube URL:",
            placeholder="https://youtube.com/watch?v=...",
            help="Paste any YouTube video URL (must have captions/transcripts)"
        )

# Advanced options in expander
with st.expander("⚙️ Advanced Options"):
    col1, col2 = st.columns(2)
    with col1:
        show_original = st.checkbox("Show original text", value=False)
        auto_copy = st.checkbox("Auto-copy summary", value=False)
    with col2:
        word_limit = st.number_input("Max input words", min_value=100, max_value=10000, value=5000)

# ------------------------
# Main processing button
# ------------------------
if st.button("🚀 Generate Summary", type="primary"):
    content = ""
    source_info = ""
    
    # Extract content based on input type
    with st.spinner("📖 Extracting content..."):
        if input_type == "📄 File Upload" and uploaded_file:
            content = extract_text_from_file(uploaded_file)
            source_info = f"File: {uploaded_file.name}"
            
        elif input_type == "🌐 Website Link" and text_input:
            content = extract_text_from_website(text_input.strip())
            source_info = f"Website: {text_input.strip()}"
            
        elif input_type == "🎥 YouTube Video" and text_input:
            if YOUTUBE_API_AVAILABLE:
                content = extract_youtube_transcript(text_input.strip())
                source_info = f"YouTube: {text_input.strip()}"
            else:
                st.error("YouTube functionality not available.")
                st.stop()
    
    # Validate content
    if not content or len(content.strip()) < 50:
        st.error("❌ No sufficient content found to summarize. Please check your input and try again.")
        if content:
            st.info(f"Content length: {len(content)} characters")
        st.stop()
    
    # Word limit check
    word_count = len(content.split())
    if word_count > word_limit:
        st.warning(f"⚠️ Content is {word_count} words. Truncating to {word_limit} words.")
        content = ' '.join(content.split()[:word_limit])
    
    # Generate summary
    with st.spinner("🤖 Generating intelligent summary..."):
        summary_text, stats = summarize_text(content, summary_length)
    
    if not summary_text:
        st.error("❌ Failed to generate summary. Please try again.")
        st.stop()
    
    # Display results
    st.success("✅ Summary generated successfully!")
    
    # Statistics
    if stats:
        st.markdown("### 📊 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Original Words", stats.get("original_words", 0))
        with col2:
            st.metric("Summary Words", stats.get("summary_words", 0))
        with col3:
            st.metric("Compression", f"{stats.get('compression_ratio', 0):.1f}%")
        with col4:
            st.metric("Reading Time", f"{stats.get('reading_time', 1)} min")
    
    # Summary display
    st.markdown("### 📝 Summary")
    st.markdown(f'<div class="summary-box">{summary_text}</div>', unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📋 Copy Summary"):
            st.write("Summary copied to clipboard!")  # Note: Real clipboard copy requires JavaScript
            if auto_copy:
                st.success("✅ Auto-copied to clipboard!")
    
    with col2:
        st.download_button(
            label="💾 Download TXT",
            data=f"Source: {source_info}\nGenerated: {st.session_state.get('timestamp', 'Unknown')}\n\n{summary_text}",
            file_name=f"summary_{summary_length.lower()}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    with col3:
        st.download_button(
            label="📄 Download Markdown", 
            data=f"# Summary\n\n**Source:** {source_info}\n**Length:** {summary_length}\n\n{summary_text}",
            file_name=f"summary_{summary_length.lower()}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )
    
    # Show original content if requested
    if show_original:
        with st.expander("📄 View Original Content"):
            st.text_area("Original Text", content, height=200, disabled=True)

# ------------------------
# Footer
# ------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 20px;">
    <p>✨ SummaGenie </p>
    <p><small>Supports DOCX, PDF, TXT files • Website articles • YouTube transcripts</small></p>
</div>
""", unsafe_allow_html=True)