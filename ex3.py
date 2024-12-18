import nltk
from nltk.corpus import brown
from collections import Counter, defaultdict
import numpy as np
import itertools

nltk.download('brown')
nltk.download('universal_tagset')

# Load the "news" category of the Brown corpus
data = brown.tagged_sents(categories='news', tagset='universal')

# Split into training and test sets
split_point = int(len(data) * 0.9)
train_data = data[:split_point]
test_data = data[split_point:]

# Helper function to strip complex tags
def simplify_tag(tag):
    return tag.split('+')[0].split('-')[0]

# Preprocess training and test data to simplify tags
train_data = [[(word, simplify_tag(tag)) for word, tag in sentence] for sentence in train_data]
test_data = [[(word, simplify_tag(tag)) for word, tag in sentence] for sentence in test_data]

# (b) Most likely tag baseline
# i. Compute the most likely tag for each word in the training set
word_tag_counts = Counter((word, tag) for sentence in train_data for word, tag in sentence)
word_counts = Counter(word for word, _ in word_tag_counts)
tag_counts = Counter(tag for _, tag in word_tag_counts)

most_likely_tag = {word: max(
    [(tag, count / word_counts[word]) for (w, tag), count in word_tag_counts.items() if w == word],
    key=lambda x: x[1]
)[0] for word in word_counts}

def predict_most_likely_tag(word):
    return most_likely_tag.get(word, "NOUN")  # Default to "NOUN" for unknown words

# ii. Compute error rates on the test set
def evaluate_baseline(test_data):
    total, correct, known_correct, unknown_correct = 0, 0, 0, 0
    known_total, unknown_total = 0, 0

    train_vocab = set(word_counts.keys())

    for sentence in test_data:
        for word, true_tag in sentence:
            predicted_tag = predict_most_likely_tag(word)
            total += 1
            if word in train_vocab:
                known_total += 1
                if predicted_tag == true_tag:
                    known_correct += 1
            else:
                unknown_total += 1
                if predicted_tag == true_tag:
                    unknown_correct += 1
            if predicted_tag == true_tag:
                correct += 1

    total_error = 1 - (correct / total)
    known_error = 1 - (known_correct / known_total)
    unknown_error = 1 - (unknown_correct / unknown_total)

    return total_error, known_error, unknown_error

baseline_errors = evaluate_baseline(test_data)

# Print the results for (b)
print("Baseline Error Rates:")
print(f"Known Words Error Rate: {baseline_errors[1]:.4f}")
print(f"Unknown Words Error Rate: {baseline_errors[2]:.4f}")
print(f"Total Error Rate: {baseline_errors[0]:.4f}")


# (c) Bigram HMM Tagger
# i. Training phase: Compute transition and emission probabilities
def train_bigram_hmm(data):
    transition_counts = defaultdict(Counter)
    emission_counts = defaultdict(Counter)
    tag_counts = Counter()

    for sentence in data:
        prev_tag = '<START>'
        tag_counts[prev_tag] += 1
        for word, tag in sentence:
            transition_counts[prev_tag][tag] += 1
            emission_counts[tag][word] += 1
            tag_counts[tag] += 1
            prev_tag = tag
        transition_counts[prev_tag]['<END>'] += 1

    transition_probs = {prev_tag: {tag: count / sum(tag_counts.values()) for tag, count in tags.items()} for prev_tag, tags in transition_counts.items()}
    emission_probs = {tag: {word: count / sum(words.values()) for word, count in words.items()} for tag, words in emission_counts.items()}

    return transition_probs, emission_probs

transition_probs, emission_probs = train_bigram_hmm(train_data)

# ii. Implement the Viterbi algorithm
def viterbi(sentence, transition_probs, emission_probs):
    states = list(emission_probs.keys())
    n = len(sentence)
    viterbi_matrix = np.zeros((len(states), n))
    backpointer = np.zeros((len(states), n), dtype=int)

    state_index = {state: i for i, state in enumerate(states)}

    # Initialize
    for i, state in enumerate(states):
        viterbi_matrix[i, 0] = transition_probs.get('<START>', {}).get(state, 0) * emission_probs.get(state, {}).get(sentence[0], 1e-6)

    # Recursion
    for t in range(1, n):
        for i, state in enumerate(states):
            max_prob, max_state = max(
                (viterbi_matrix[j, t - 1] * transition_probs.get(prev_state, {}).get(state, 0) * emission_probs.get(state, {}).get(sentence[t], 1e-6), j)
                for j, prev_state in enumerate(states)
            )
            viterbi_matrix[i, t] = max_prob
            backpointer[i, t] = max_state

    # Termination
    best_path = []
    best_last_state = np.argmax(viterbi_matrix[:, -1])
    best_path.append(states[best_last_state])

    for t in range(n - 1, 0, -1):
        best_last_state = backpointer[best_last_state, t]
        best_path.append(states[best_last_state])

    best_path.reverse()
    return best_path

# iii. Evaluate the HMM tagger
def evaluate_hmm(test_data, transition_probs, emission_probs):
    total, correct = 0, 0

    for sentence in test_data:
        words, true_tags = zip(*sentence)
        predicted_tags = viterbi(words, transition_probs, emission_probs)
        total += len(true_tags)
        correct += sum(p == t for p, t in zip(predicted_tags, true_tags))

    return 1 - (correct / total)

hmm_error = evaluate_hmm(test_data, transition_probs, emission_probs)
print(f"HMM Error Rate: {hmm_error:.4f}")

# (d) Add-One Smoothing
def add_one_smoothing(emission_counts, tag_counts):
    smoothed_probs = {}
    vocab_size = len(emission_counts)

    for tag, words in emission_counts.items():
        smoothed_probs[tag] = {word: (count + 1) / (tag_counts[tag] + vocab_size) for word, count in words.items()}

    return smoothed_probs

smoothed_emission_probs = add_one_smoothing(emission_counts, tag_counts)
hmm_error_smoothed = evaluate_hmm(test_data, transition_probs, smoothed_emission_probs)

print("HMM Error Rate (Original):", hmm_error)
print("HMM Error Rate (Smoothed):", hmm_error_smoothed)

# (e) Using pseudo-words
# i. Design pseudo-words for unknown and low-frequency words
def assign_pseudo_word(word):
    if word.isdigit():
        return "<NUMERIC>"
    elif any(char.isdigit() for char in word):
        return "<ALPHANUMERIC>"
    elif word.isupper():
        return "<ALL_CAPS>"
    elif word[0].isupper():
        return "<INIT_CAPS>"
    elif len(word) <= 2:
        return "<SHORT>"
    else:
        return "<RARE>"

# Replace low-frequency words in training set with pseudo-words
low_freq_threshold = 5
word_frequencies = Counter(word for sentence in train_data for word, _ in sentence)

train_data_pseudo = [[(assign_pseudo_word(word) if word_frequencies[word] < low_freq_threshold else word, tag)
                      for word, tag in sentence] for sentence in train_data]

# Train HMM with pseudo-words
transition_probs_pseudo, emission_probs_pseudo = train_bigram_hmm(train_data_pseudo)

# ii. Evaluate HMM with pseudo-words (MLE)
def preprocess_with_pseudo_words(test_data):
    return [[(assign_pseudo_word(word) if word not in word_frequencies else word, tag) for word, tag in sentence]
            for sentence in test_data]

test_data_pseudo = preprocess_with_pseudo_words(test_data)
hmm_error_pseudo = evaluate_hmm(test_data_pseudo, transition_probs_pseudo, emission_probs_pseudo)

# iii. Evaluate HMM with pseudo-words and Add-One smoothing
smoothed_emission_probs_pseudo = add_one_smoothing(emission_counts, tag_counts)
hmm_error_pseudo_smoothed = evaluate_hmm(test_data_pseudo, transition_probs_pseudo, smoothed_emission_probs_pseudo)

print("HMM Error Rate with Pseudo-Words (MLE):", hmm_error_pseudo)
print("HMM Error Rate with Pseudo-Words (Smoothed):", hmm_error_pseudo_smoothed)

# Confusion matrix
def build_confusion_matrix(test_data, transition_probs, emission_probs):
    tags = list(emission_probs.keys())
    tag_to_index = {tag: i for i, tag in enumerate(tags)}
    confusion_matrix = np.zeros((len(tags), len(tags)), dtype=int)

    for sentence in test_data:
        words, true_tags = zip(*sentence)
        predicted_tags = viterbi(words, transition_probs, emission_probs)
        for true_tag, predicted_tag in zip(true_tags, predicted_tags):
            true_idx = tag_to_index[true_tag]
            pred_idx = tag_to_index[predicted_tag]
            confusion_matrix[true_idx, pred_idx] += 1

    return confusion_matrix, tags

conf_matrix, tag_labels = build_confusion_matrix(test_data_pseudo, transition_probs_pseudo, smoothed_emission_probs_pseudo)

print("Confusion Matrix:")
print(conf_matrix)
print("Tags:", tag_labels)
