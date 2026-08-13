from app.processing.html_cleaner import HTMLCleaner

if __name__ == "__main__":
    cleaner = HTMLCleaner()
    cleaner.clean_all()