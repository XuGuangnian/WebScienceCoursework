import csv
import matplotlib.pyplot as plt

EMOTIONS = [
    'excitement',
    'happy',
    'pleasant',
    'surprise',
    'fear',
    'angry',
]

with open('cs_output.csv', 'r', encoding='UTF-8') as f:
    reader = csv.DictReader(f)
    i = 0
    for row in reader:
        if row['what_is_the_authors_sentiment_feeling_throughout_the_post'] == row['label']:
            i = i + 1

rate = i/295

print(rate)

# 21.7%
