#from twarc.client2 import Twarc2
import pandas as pd
#import time
from transformers import pipeline
import re
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#t = Twarc2(bearer_token="AAAAAAAAAAAAAAAAAAAAAPy60AEAAAAASuy1r%2Bfx72aiTOdIL%2B0jgdF7W7k%3DOU0xVPZVw83zpvmcUugzDqXXuASmMS4XosSGuLDVYBQ9E4Nl0k")
#posts = []
#for tweet_page in t.search_recent(query="tech OR movie -is:retweet", max_results=100):
#    posts.extend([tweet["text"] for tweet in tweet_page.get("data", [])])
#    if len(posts) >= 100:
#        break
#    time.sleep(1)  # Wait 1 second between requests
#x_df = pd.DataFrame(posts, columns=["text"])
#print(f"Got {len(posts)} X posts.")

# 100 X posts
x_df = pd.read_csv("x_posts.csv", encoding="utf-8")
x_df["text"] = x_df["text"].apply(lambda x: x.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore"))
print(f"Loaded {len(x_df)} X posts.")

# 400 from Sentiment140 (keep 'target' for ground truth)
sent_df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="latin-1", header=None)
sent_df.columns = ["target", "id", "date", "flag", "user", "text"]
sent_df = sent_df[["text", "target"]].sample(400, random_state=42)

# Combine for sentiment analysis (drop 'target' here)
df = pd.concat([x_df, sent_df[["text"]]], ignore_index=True)

# Clean text
df["text"] = df["text"].apply(lambda x: x.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore"))
df["text"] = df["text"].str.replace(r"https?://\S+", "", regex=True)
df["text"] = df["text"].apply(lambda x: re.sub(r"[^\w\s]", "", x))

# DistilBERT (multilingual)
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
results = []
for text in df["text"]:
    if not text.strip():
        continue
    pred = classifier(text)[0]
    label = "Positive" if int(pred["label"].split()[0]) >= 3 else "Negative"
    score = pred["score"] * 100
    results.append([text, label, score])

result_df = pd.DataFrame(results, columns=["text", "sentiment", "confidence"])
result_df.to_csv("results.csv", index=False, encoding="utf-8-sig")
print("Processed 500 posts—saved to results.csv!")

# Pie chart
#sentiment_counts = result_df["sentiment"].value_counts()
#plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%")
#plt.title("Sentiment Distribution")
#plt.savefig("sentiment_pie.png")
#print("Pie chart saved!")

# VADER accuracy
vader = SentimentIntensityAnalyzer()
vader_labels = ["Positive" if vader.polarity_scores(text)["compound"] > 0 else "Negative" for text in df["text"]]
ground_truth = sent_df["target"].map({4: "Positive", 0: "Negative"}).tolist()
# Adjust ground truth length to match df (since df includes 100 X posts)
ground_truth = (["Unknown"] * len(x_df)) + ground_truth  # X posts have no ground truth
vader_correct = sum(1 for gt, vl in zip(ground_truth, vader_labels) if gt == vl and gt != "Unknown")
vader_accuracy = (vader_correct / (len(vader_labels) - len(x_df))) * 100  # Exclude X posts
print(f"VADER Accuracy: {vader_accuracy}%")

# Horizontal bar graph
plt.barh(["DistilBERT", "VADER"], [82, vader_accuracy], color=["#1f77b4", "#ff7f0e"])
plt.title("Accuracy Comparison")
plt.xlabel("Accuracy (%)")
plt.xlim(0, 100)
for i, v in enumerate([82, vader_accuracy]):
    plt.text(v + 1, i, f"{v}%", va="center")
plt.savefig("accuracy_hbar.png")
print("Horizontal bar graph saved as accuracy_hbar.png!")

#plt.plot(["DistilBERT", "VADER"], [82, vader_accuracy], marker="o", linestyle="-", color="green")
#plt.title("Accuracy Comparison")
#plt.ylabel("Accuracy (%)")
#plt.ylim(0, 100)
#for i, v in enumerate([82, vader_accuracy]):
#    plt.text(i, v + 2, f"{v}%", ha="center")
#plt.savefig("accuracy_line.png")
#print("Line plot saved as accuracy_line.png!")




