import math
import itertools
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple


class NGramSentenceDecoder:
    """
    Statistical N-gram language model re-ranker.

    It does NOT use hardcoded target sentences.
    It learns word transition probabilities from an external corpus file.

    Input:
        model top-k predictions for each sign segment

    Output:
        the most linguistically plausible sentence
        based on:
            1. sign model confidence
            2. n-gram language probability
    """

    def __init__(
        self,
        corpus_path: str,
        n: int = 2,
        alpha: float = 0.65,
        beta: float = 0.35,
        top_k_per_segment: int = 5,
        max_combinations: int = 5000,
    ):
        self.corpus_path = Path(corpus_path)
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.top_k_per_segment = top_k_per_segment
        self.max_combinations = max_combinations

        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.context_counts = Counter()
        self.vocab = set()

        self._load_corpus()

    # -------------------------------
    # Text cleaning
    # -------------------------------

    def clean_word(self, word: str) -> str:
        word = str(word or "").lower().strip()

        # Remove ASL gloss numbers:
        # WANT1 -> want
        # DINNER2 -> dinner
        word = re.sub(r"\d+$", "", word)

        # Convert common compact labels
        word = word.replace("_", " ")
        word = word.replace("-", " ")

        # Keep letters/numbers/spaces only
        word = "".join(ch for ch in word if ch.isalnum() or ch.isspace())
        word = " ".join(word.split())

        # Simple lexical normalization, not hardcoded sentences
        replacements = {
            "myself": "i",
            "hardofhearing": "hard of hearing",
            "lickenvelope": "lick envelope",
            "whatfor": "what for",
            "allofsudden": "all of sudden",
            "8hour": "eight hour",
        }

        return replacements.get(word, word)

    def tokenize(self, text: str) -> List[str]:
        text = str(text or "").lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = " ".join(text.split())

        if not text:
            return []

        return text.split()

    def detokenize(self, words: List[str]) -> str:
        words = [w for w in words if w]

        if not words:
            return ""

        sentence = " ".join(words)

        # Basic cleanup
        sentence = sentence.replace(" i ", " I ")
        if sentence.startswith("i "):
            sentence = "I " + sentence[2:]
        elif sentence == "i":
            sentence = "I"

        sentence = sentence.strip()

        if sentence:
            sentence = sentence[0].upper() + sentence[1:]

        return sentence

    # -------------------------------
    # Corpus loading
    # -------------------------------

    def _load_corpus(self):
        if not self.corpus_path.exists():
            raise FileNotFoundError(
                f"Sentence corpus not found at: {self.corpus_path}"
            )

        lines = self.corpus_path.read_text(encoding="utf-8").splitlines()

        for line in lines:
            line = line.strip()

            if not line:
                continue

            tokens = self.tokenize(line)

            if not tokens:
                continue

            tokens = ["<s>"] + tokens + ["</s>"]

            for token in tokens:
                self.unigram_counts[token] += 1
                self.vocab.add(token)

            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i + 1])
                self.bigram_counts[bigram] += 1
                self.context_counts[tokens[i]] += 1

        self.vocab.add("<unk>")
        self.vocab_size = max(len(self.vocab), 1)

    # -------------------------------
    # Language model scoring
    # -------------------------------

    def bigram_log_prob(self, prev_word: str, word: str) -> float:
        """
        Add-one smoothed bigram probability:
        P(word | prev_word)
        """

        bigram_count = self.bigram_counts[(prev_word, word)]
        context_count = self.context_counts[prev_word]

        prob = (bigram_count + 1.0) / (context_count + self.vocab_size)

        return math.log(prob)

    def sentence_lm_score(self, words: List[str]) -> float:
        """
        Average log probability of the sentence.
        """

        if not words:
            return -999.0

        tokens = ["<s>"] + words + ["</s>"]
        log_probs = []

        for i in range(len(tokens) - 1):
            log_probs.append(self.bigram_log_prob(tokens[i], tokens[i + 1]))

        return sum(log_probs) / max(len(log_probs), 1)

    # -------------------------------
    # Candidate extraction
    # -------------------------------

    def candidates_from_segment(self, segment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract candidate words from segment top_k.
        """

        candidates = []
        seen = set()

        # Direct prediction if exists
        direct_word = self.clean_word(segment.get("text", ""))
        direct_gloss = segment.get("gloss", "")
        direct_conf = float(segment.get("confidence", 0.0) or 0.0)

        if direct_word:
            candidates.append({
                "word": direct_word,
                "gloss": direct_gloss,
                "confidence": direct_conf,
                "rank": 1,
            })
            seen.add(direct_word)

        # Top-k predictions
        top_k = segment.get("top_k", []) or []

        for idx, item in enumerate(top_k[:self.top_k_per_segment]):
            word = self.clean_word(item.get("text", ""))
            gloss = item.get("gloss", "")
            confidence = float(item.get("confidence", 0.0) or 0.0)

            if not word:
                continue

            if word in seen:
                continue

            candidates.append({
                "word": word,
                "gloss": gloss,
                "confidence": confidence,
                "rank": idx + 1,
            })

            seen.add(word)

        return candidates

    def sign_score_for_combination(self, combination: Tuple[Dict[str, Any], ...]) -> float:
        """
        Score from model confidence and rank.
        """

        scores = []

        for cand in combination:
            conf = float(cand.get("confidence", 0.0) or 0.0)
            rank = int(cand.get("rank", 1) or 1)

            # Avoid log(0)
            conf = max(conf, 1e-6)

            # Rank penalty:
            # top-1 stronger than top-5
            rank_penalty = 1.0 / (rank ** 0.35)

            score = math.log(conf) + math.log(rank_penalty)

            scores.append(score)

        if not scores:
            return -999.0

        return sum(scores) / len(scores)

    # -------------------------------
    # Main decoding
    # -------------------------------

    def decode(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode the most probable sentence from segment top-k candidates.
        """

        if not isinstance(result, dict):
            return result

        segments = result.get("segments", []) or []

        if not segments:
            result["lm_decoder_used"] = False
            result["lm_reason"] = "No segments found."
            return result

        all_candidates = []

        for seg in segments:
            cands = self.candidates_from_segment(seg)

            if not cands:
                result["lm_decoder_used"] = False
                result["lm_reason"] = "At least one segment has no candidates."
                return result

            all_candidates.append(cands)

        total_combinations = 1
        for cands in all_candidates:
            total_combinations *= len(cands)

        if total_combinations > self.max_combinations:
            result["lm_decoder_used"] = False
            result["lm_reason"] = f"Too many combinations: {total_combinations}"
            return result

        best = None
        best_score = -999999.0

        all_combinations = itertools.product(*all_candidates)

        for combination in all_combinations:
            words = [cand["word"] for cand in combination]

            sign_score = self.sign_score_for_combination(combination)
            lm_score = self.sentence_lm_score(words)

            final_score = (self.alpha * sign_score) + (self.beta * lm_score)

            if final_score > best_score:
                best_score = final_score
                best = {
                    "words": words,
                    "glosses": [cand.get("gloss", "") for cand in combination],
                    "confidences": [float(cand.get("confidence", 0.0) or 0.0) for cand in combination],
                    "ranks": [int(cand.get("rank", 1) or 1) for cand in combination],
                    "sign_score": sign_score,
                    "lm_score": lm_score,
                    "final_score": final_score,
                }

        if not best:
            result["lm_decoder_used"] = False
            result["lm_reason"] = "No valid combination found."
            return result

        decoded_sentence = self.detokenize(best["words"])

        raw_sentence = result.get("text", "")

        result["raw_model_sentence"] = raw_sentence
        result["lm_sentence"] = decoded_sentence
        result["full_sentence"] = decoded_sentence
        result["text"] = decoded_sentence

        result["word_sequence"] = best["words"]
        result["gloss_sequence"] = best["glosses"]

        result["lm_decoder_used"] = True
        result["lm_scores"] = {
            "sign_score": round(best["sign_score"], 4),
            "lm_score": round(best["lm_score"], 4),
            "final_score": round(best["final_score"], 4),
        }
        result["lm_details"] = {
            "candidate_confidences": best["confidences"],
            "candidate_ranks": best["ranks"],
            "total_combinations": total_combinations,
            "decoder_type": "bigram_ngram_reranker",
            "corpus_path": str(self.corpus_path),
        }

        return result


_decoder_cache = None


def load_language_decoder(corpus_path: str) -> NGramSentenceDecoder:
    global _decoder_cache

    if _decoder_cache is None:
        _decoder_cache = NGramSentenceDecoder(
            corpus_path=corpus_path,
            n=2,
            alpha=0.70,
            beta=0.30,
            top_k_per_segment=5,
            max_combinations=5000,
        )

    return _decoder_cache