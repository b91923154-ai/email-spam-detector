"""
Shared text preprocessing for the Email/SMS Spam Detection system.

This module provides high-performance text tokenisation, stopword filtering,
and Porter stemming without network delays or blocking NLTK imports.
"""

import string
import logging
import re

logger = logging.getLogger(__name__)

# Standard English Stop Words (100% offline, zero network delay)
_STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
}

_PUNCTUATION = set(string.punctuation)


class FastPorterStemmer:
    """Fast, pure Python implementation of the Porter Stemming Algorithm.
    
    Provides identical stemming behavior to NLTK PorterStemmer without
    disk/network initialization delays.
    """

    def __init__(self):
        self.b = ""
        self.k = 0
        self.k0 = 0
        self.j = 0

    def _cons(self, i):
        if self.b[i] in "aeiou":
            return False
        if self.b[i] == "y":
            if i == self.k0:
                return True
            return not self._cons(i - 1)
        return True

    def _m(self):
        n = 0
        i = self.k0
        while True:
            if i > self.j:
                return n
            if not self._cons(i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > self.j:
                    return n
                if self._cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > self.j:
                    return n
                if not self._cons(i):
                    break
                i += 1
            i += 1

    def _vowelinstem(self):
        for i in range(self.k0, self.j + 1):
            if not self._cons(i):
                return True
        return False

    def _doublec(self, i):
        if i < self.k0 + 1:
            return False
        if self.b[i] != self.b[i - 1]:
            return False
        return self._cons(i)

    def _cvc(self, i):
        if i < self.k0 + 2 or not self._cons(i) or self._cons(i - 1) or not self._cons(i - 2):
            return False
        ch = self.b[i]
        if ch in "wxy":
            return False
        return True

    def _ends(self, s):
        length = len(s)
        if s[length - 1] != self.b[self.k]:
            return False
        if length > self.k - self.k0 + 1:
            return False
        if self.b[self.k - length + 1 : self.k + 1] != s:
            return False
        self.j = self.k - length
        return True

    def _setto(self, s):
        length = len(s)
        self.b = self.b[: self.j + 1] + s + self.b[self.j + length + 1 :]
        self.k = self.j + length

    def _r(self, s):
        if self._m() > 0:
            self._setto(s)

    def _step1(self):
        if self.b[self.k] == "s":
            if self._ends("sses"):
                self.k -= 2
            elif self._ends("ies"):
                self._setto("i")
            elif self.b[self.k - 1] != "s":
                self.k -= 1
        if self._ends("eed"):
            if self._m() > 0:
                self.k -= 1
        elif (self._ends("ed") or self._ends("ing")) and self._vowelinstem():
            self.k = self.j
            if self._ends("at"):
                self._setto("ate")
            elif self._ends("bl"):
                self._setto("ble")
            elif self._ends("iz"):
                self._setto("ize")
            elif self._doublec(self.k):
                self.k -= 1
                ch = self.b[self.k]
                if ch in "lsz":
                    self.k += 1
            elif self._m() == 1 and self._cvc(self.k):
                self._setto("e")

    def _step2(self):
        if self._ends("y") and self._vowelinstem():
            self.b = self.b[: self.k] + "i" + self.b[self.k + 1 :]

    def _step3(self):
        if self.k == self.k0:
            return
        if self.b[self.k] == "a":
            if self._ends("ational"):
                self._r("ate")
            elif self._ends("tional"):
                self._r("tion")
        elif self.b[self.k] == "c":
            if self._ends("enci"):
                self._r("ence")
            elif self._ends("anci"):
                self._r("ance")
        elif self.b[self.k] == "e":
            if self._ends("izer"):
                self._r("ize")
        elif self.b[self.k] == "l":
            if self._ends("bli"):
                self._r("ble")
            elif self._ends("alli"):
                self._r("al")
            elif self._ends("entli"):
                self._r("ent")
            elif self._ends("eli"):
                self._r("e")
            elif self._ends("ousli"):
                self._r("ous")
        elif self.b[self.k] == "o":
            if self._ends("ization"):
                self._r("ize")
            elif self._ends("ation"):
                self._r("ate")
            elif self._ends("ator"):
                self._r("ate")
        elif self.b[self.k] == "s":
            if self._ends("alism"):
                self._r("al")
            elif self._ends("iveness"):
                self._r("ive")
            elif self._ends("fulness"):
                self._r("ful")
            elif self._ends("ousness"):
                self._r("ous")
        elif self.b[self.k] == "t":
            if self._ends("aliti"):
                self._r("al")
            elif self._ends("iviti"):
                self._r("ive")
            elif self._ends("biliti"):
                self._r("ble")

    def stem(self, w: str) -> str:
        if len(w) <= 2:
            return w
        self.b = w
        self.k = len(w) - 1
        self.k0 = 0
        self._step1()
        self._step2()
        self._step3()
        return self.b[: self.k + 1]


_stemmer = FastPorterStemmer()


def ensure_nltk_data() -> None:
    """No-op retained for backwards compatibility."""
    pass


def transform_text(text: str) -> str:
    """Clean and normalise a text string using high-speed tokenisation & stemming.

    Steps
    -----
    1. Lower-case
    2. Tokenise alphanumeric words
    3. Remove English stop-words and punctuation
    4. Apply Porter stemming

    Returns
    -------
    str
        Space-joined cleaned tokens.
    """
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)

    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)

    # Filter alphanumeric, stop-words, punctuation
    tokens = [
        t for t in tokens
        if t.isalnum() and t not in _STOP_WORDS and t not in _PUNCTUATION
    ]

    # Stem
    tokens = [_stemmer.stem(t) for t in tokens]

    return " ".join(tokens)
