# ebook-organiser
Categorise ebooks into subject areas

This works on a directory continaing ebooks (PDF or epub) and tries to sort them into maeningful subdirectories.


How it works
Metadata Extraction:
Tries to extract metadata (title, author, ISBN) from EPUB and PDF files.

Google Books API Lookup:
If metadata extraction fails, tries the Google Books API using the file name (with underscores replaced by spaces).

File Name-Based Categorization:
If both metadata and Google Books API lookup fail, falls back to file name-based categorization.

LDA-Based Categorization:
If all else fails, uses text extraction and LDA to categorize the file.

Detailed Logging:
Logs the current file being processed, the method being tried, whether it was successful, and the results.

Installing:
download and extract to a directory in your path
chmod +x ./run_pdf_categorizer.sh 

This will create a virtual python environment, and install any missing libaries.

Running: 
./run_pdf_categorizer.sh {path to ebook directory}
