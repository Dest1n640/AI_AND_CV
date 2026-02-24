tp = 0
tn = 0
fp = 0
fn = 0
with open('responses.txt', 'r', encoding='utf-8') as file:
  for line in file:
    words = line.split()
    print(words)

    if words[0] == "positive" and words[1] == "positive":
      tp += 1
    elif words[0] == "negative" and words[1] == "positive":
      fp += 1
    elif words[0] == "positive" and words[1] == "negative":
      fn += 1
    elif words[0] == "negative" and words[1] == "negative":
      tn += 1

print(tp, tn, fp, fn)
acurracy =  (tp + tn) / (tp + tn + fp + fn)
recall = tp / (tp + fn)
precision = tp / (tp + fp)
f1_score = 2 * ((precision * recall) / (precision + recall))

print(acurracy, recall, precision, f1_score)
