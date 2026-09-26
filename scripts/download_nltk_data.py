"""One-time NLTK data check/download for the detector's POS tagging."""
import nltk

resources = [
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("tokenizers/punkt_tab", "punkt_tab"),
]

for find_path, download_name in resources:
    try:
        nltk.data.find(find_path)
        print(f"  {download_name} already present")
    except LookupError:
        print(f"  Downloading {download_name}...")
        nltk.download(download_name, quiet=True)
        print(f"  Done: {download_name}")