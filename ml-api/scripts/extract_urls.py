import pandas as pd

# Correct file path
df = pd.read_csv("../data/phishing_enhanced.csv")

# Extract URLs
urls = df['original_url'].dropna().unique()

# Save to text file
with open("../data/phishing_urls.txt", "w", encoding="utf-8") as f:
    for url in urls:
        f.write(url + "\n")

print(f"Saved {len(urls)} phishing URLs to phishing_urls.txt")
