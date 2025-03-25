import os
import sys
import time
import shutil
import logging
import zipfile
import urllib.request
import json
import urllib.parse
from pathlib import Path
from pdfminer.high_level import extract_text
from ebooklib import epub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Download NLTK data (if not already downloaded)
nltk.download('wordnet')
nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# BISAC Subject Headings List
BISAC_TOPICS = [
    "Antiques & Collectibles", "Architecture", "Art", "Bibles", "Biography & Autobiography", "Body",
    "Business & Economics", "Comics & Graphic Novels", "Computers", "Cooking", "Crafts & Hobbies", "Design",
    "Drama", "Education", "Family & Relationships", "Fiction", "Foreign Language Study", "Games & Activities",
    "Gardening", "Health & Fitness", "History", "House & Home", "Humor", "Juvenile Fiction", "Juvenile Nonfiction",
    "Language Arts & Disciplines", "Law", "Literary Collections", "Literary Criticism", "Mathematics", "Medical",
    "Music", "Nature", "Performing Arts", "Pets", "Philosophy", "Photography", "Poetry", "Political Science",
    "Psychology", "Reference", "Religion", "Science", "Self-help", "Social Science", "Sports & Recreation",
    "Study Aids", "Technology & Engineering", "Transportation", "Travel", "True Crime", "Young Adult Fiction",
    "Young Adult Nonfiction"
]
NUM_TOPICS = len(BISAC_TOPICS)

# Keyword-to-Category Mapping
KEYWORD_TO_CATEGORY = {
    "python": "Computers",
    "java": "Computers",
    "programming": "Computers",
    "geometry": "Mathematics",
    "algebra": "Mathematics",
    "calculus": "Mathematics",
    "medical": "Medical",
    "surgery": "Medical",
    "physics": "Science",
    "chemistry": "Science",
    # Add more mappings as needed
}

# Google Books API Key (replace with your own key)
GOOGLE_BOOKS_API_KEY = "YOUR_GOOGLE_BOOKS_API_KEY"

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    """Preprocess text by lemmatizing and removing stop words."""
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words]
    return " ".join(words)

# Function to extract metadata from EPUB
def extract_metadata_from_epub(file_path):
    """Extract metadata (title, author, ISBN) from an EPUB file."""
    try:
        book = epub.read_epub(file_path)
        title = book.get_metadata('DC', 'title')
        author = book.get_metadata('DC', 'creator')
        isbn = book.get_metadata('DC', 'identifier')
        return (
            title[0][0] if title else None,
            author[0][0] if author else None,
            isbn[0][0] if isbn else None
        )
    except Exception as e:
        logging.error(f"Error extracting metadata from EPUB file {file_path}: {e}")
        return None, None, None

# Function to extract metadata from PDF
def extract_metadata_from_pdf(file_path):
    """Extract metadata (title, author, ISBN) from a PDF file."""
    try:
        # PDF metadata extraction is limited; this is a placeholder
        # You may need to use a library like PyPDF2 or pdfminer.six for better metadata extraction
        return None, None, None
    except Exception as e:
        logging.error(f"Error extracting metadata from PDF file {file_path}: {e}")
        return None, None, None

# Function to look up book details using Google Books API
def lookup_book_details(title, author, isbn):
    """Look up book details (category, ISBN) using the Google Books API."""
    query = f"intitle:{title}" if title else ""
    query += f" inauthor:{author}" if author else ""
    query += f" isbn:{isbn}" if isbn else ""
    
    if not query:
        logging.info("Google Books API: No query generated (missing title, author, and ISBN).")
        return None, None
    
    # URL-encode the query string
    query_encoded = urllib.parse.quote(query)
#    url = f"https://www.googleapis.com/books/v1/volumes?q={query_encoded}&key={GOOGLE_BOOKS_API_KEY}"
    url = f"https://www.googleapis.com/books/v1/volumes?q={query_encoded}"
    
    # Log the API request URL (without the API key for security)
    logging.info(f"Google Books API Request: q={query_encoded}")
    
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            if data.get("items"):
                book = data["items"][0]["volumeInfo"]
                isbn = book.get("industryIdentifiers", [{}])[0].get("identifier")
                categories = book.get("categories", [])
                category = categories[0] if categories else None
                return isbn, category
    except Exception as e:
        logging.error(f"Error looking up book details: {e}")
    return None, None




# Function to extract text from EPUB
def extract_text_from_epub(file_path):
    """Extract text from an EPUB file, handling errors."""
    logging.info(f"Processing file: {file_path}")
    
    try:
        book = epub.read_epub(file_path)
    except (zipfile.BadZipFile, KeyError) as e:
        logging.error(f"Error loading EPUB file {file_path}: {e}. Skipping...")
        return ""
    
    text = []
    for item in book.get_items():
        if isinstance(item, epub.EpubHtml):
            try:
                content = item.get_content().decode('utf-8')
                text.append(content)
            except Exception as e:
                logging.warning(f"Error processing item in EPUB file {file_path}: {e}. Skipping...")
                continue
    return "\n".join(text)

# Function to extract text from PDFs and EPUBs
def extract_text_from_file(file_path):
    """Extract text from a PDF or EPUB file, handling errors."""
    try:
        if file_path.suffix.lower() == ".pdf":
            return extract_text(file_path)
        elif file_path.suffix.lower() == ".epub":
            return extract_text_from_epub(file_path)
    except Exception as e:
        logging.error(f"Error processing file {file_path.name}: {e}. Skipping...")
    return ""

# Function to categorize based on file name
def categorize_by_filename(file_name):
    """Categorize a file based on keywords in its name."""
    file_name_lower = file_name.lower()  # Case-insensitive matching
    for keyword, category in KEYWORD_TO_CATEGORY.items():
        if keyword in file_name_lower:
            return category
    return None  # No match found

# Function to log top words for each topic
def log_top_words(model, vectorizer, n_top_words=10):
    """Log the top words for each topic."""
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        logging.info(f"Topic {topic_idx}: {', '.join(top_words)}")


def sanitize_category_name(category):
    """Replace invalid filesystem characters in category names."""
    return category.replace("/", "-")  # Replace / with -

def categorize_documents(directory):
    directory = Path(directory)
    files = list(directory.glob("*.pdf")) + list(directory.glob("*.epub"))
    
    if not files:
        logging.info("No PDF or EPUB files found in the directory.")
        return
    
    logging.info(f"Processing {len(files)} files...")
    start_time = time.time()
    
    texts = []
    valid_files = []
    
    for file in files:
        logging.info(f"Processing file: {file.name}")
        
        # Step 1: Extract metadata
        logging.info("Trying metadata extraction...")
        if file.suffix.lower() == ".epub":
            title, author, isbn = extract_metadata_from_epub(file)
        else:
            title, author, isbn = extract_metadata_from_pdf(file)
        
        if title or author or isbn:
            logging.info(f"Metadata found - Title: {title}, Author: {author}, ISBN: {isbn}")
            # Step 2: Look up book details using Google Books API
            logging.info("Trying Google Books API lookup...")
            new_isbn, category = lookup_book_details(title, author, isbn)
            if new_isbn:
                isbn = new_isbn
            if category:
                logging.info(f"Google Books API lookup successful - Category: {category}, ISBN: {isbn}")
                # Sanitize category name
                category = sanitize_category_name(category)
                # Append ISBN to filename
                new_filename = f"{file.stem}_{isbn}{file.suffix}" if isbn else file.name
                new_file_path = file.with_name(new_filename)
                file.rename(new_file_path)
                
                # Move file to category folder
                topic_dir = directory / category
                topic_dir.mkdir(exist_ok=True)
                shutil.move(str(new_file_path), str(topic_dir / new_filename))
                logging.info(f"Moved {new_filename} to {category}/ (based on metadata and API lookup)")
                continue
        
        # Step 3: Try Google Books API lookup using file name
        logging.info("Trying Google Books API lookup using file name...")
        file_name_query = file.stem.replace("_", " ")
        new_isbn, category = lookup_book_details(file_name_query, None, None)
        if category:
            logging.info(f"Google Books API lookup successful - Category: {category}, ISBN: {new_isbn}")
            # Sanitize category name
            category = sanitize_category_name(category)
            # Append ISBN to filename
            new_filename = f"{file.stem}_{new_isbn}{file.suffix}" if new_isbn else file.name
            new_file_path = file.with_name(new_filename)
            file.rename(new_file_path)
            
            # Move file to category folder
            topic_dir = directory / category
            topic_dir.mkdir(exist_ok=True)
            shutil.move(str(new_file_path), str(topic_dir / new_filename))
            logging.info(f"Moved {new_filename} to {category}/ (based on file name and API lookup)")
            continue
        
        # Step 4: Fall back to file name-based categorization
        logging.info("Trying file name-based categorization...")
        category = categorize_by_filename(file.name)
        if category:
            logging.info(f"File name-based categorization successful - Category: {category}")
            # Sanitize category name
            category = sanitize_category_name(category)
            topic_dir = directory / category
            topic_dir.mkdir(exist_ok=True)
            shutil.move(str(file), str(topic_dir / file.name))
            logging.info(f"Moved {file.name} to {category}/ (based on file name)")
            continue
        
        # Step 5: Extract text and use LDA
        logging.info("Trying text extraction and LDA-based categorization...")
        text = extract_text_from_file(file)
        if text.strip():
            text = preprocess_text(text)
            texts.append(text)
            valid_files.append(file)
        else:
            logging.warning(f"Skipping file {file.name} because it contains no text.")
    
    if not texts:
        logging.error("No valid text extracted from any files. Exiting.")
        return
    
    # Convert documents to TF-IDF representation
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    # Perform topic modeling
    lda = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=42, max_iter=100)
    lda.fit(doc_term_matrix)
    
    # Log top words for each topic (for debugging)
    log_top_words(lda, vectorizer)
    
    topics = lda.transform(doc_term_matrix)
    
    # Move files into categorized folders
    for i, file in enumerate(valid_files):
        topic_index = topics[i].argmax()
        topic_name = BISAC_TOPICS[topic_index]
        # Sanitize category name
        topic_name = sanitize_category_name(topic_name)
        topic_dir = directory / topic_name
        topic_dir.mkdir(exist_ok=True)
        shutil.move(str(file), str(topic_dir / file.name))
        logging.info(f"Moved {file.name} to {topic_name}/ (based on LDA)")
    
    elapsed_time = time.time() - start_time
    logging.info(f"Completed processing in {elapsed_time:.2f} seconds.")


# Main execution
def main():
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = input("Enter the directory path: ")
    
    if not os.path.isdir(directory):
        logging.error("Invalid directory. Please provide a valid path.")
        sys.exit(1)
    
    categorize_documents(directory)

if __name__ == "__main__":
    main()
