from collections import defaultdict, Counter
import random
import re
from typing import Dict, List, Tuple

class BigramModel:
    def __init__(self, corpus: List[str], frequency_threshold: int = 5):
        self.corpus = corpus
        self.frequency_threshold = frequency_threshold
        self.vocab, self.bigram_probs = self.analyze_bigrams(" ".join(corpus))

    def simple_tokenizer(self, text: str) -> List[str]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not self.frequency_threshold or self.frequency_threshold <= 1:
            return tokens
        counts = Counter(tokens)
        return [t for t in tokens if counts[t] >= self.frequency_threshold]

    def analyze_bigrams(self, text: str) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
        words = self.simple_tokenizer(text)
        if len(words) < 2:
            return list(sorted(set(words))), {}

        bigrams = list(zip(words[:-1], words[1:]))
        bigram_counts = Counter(bigrams)
        unigram_counts = Counter(words)

        probs: Dict[str, Dict[str, float]] = defaultdict(dict)
        for (w1, w2), c in bigram_counts.items():
            if unigram_counts[w1] > 0:
                probs[w1][w2] = c / unigram_counts[w1]

        vocab = list(unigram_counts.keys())
        return vocab, probs

    def generate_text(self, start_word: str, num_words: int = 20) -> str:
        if not start_word:
            return ""
        current_word = start_word.lower()
        generated_words = [current_word]

        for _ in range(max(1, num_words) - 1):
            next_words = self.bigram_probs.get(current_word)
            if not next_words:
                break
            next_word = random.choices(
                list(next_words.keys()), weights=list(next_words.values()), k=1
            )[0]
            generated_words.append(next_word)
            current_word = next_word

        return " ".join(generated_words)

    def print_bigram_probs_matrix_python(self):
        vocab = sorted(set(self.vocab))
        print(f"{'':<15}", end="")
        for word in vocab:
            print(f"{word:<15}", end="")
        print("\n" + "-" * (15 * (len(vocab) + 1)))

        for w1 in vocab:
            print(f"{w1:<15}", end="")
            for w2 in vocab:
                prob = self.bigram_probs.get(w1, {}).get(w2, 0.0)
                print(f"{prob:<15.2f}", end="")
            print()
