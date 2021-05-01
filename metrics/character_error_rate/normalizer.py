#!/usr/bin/env python3
"""
    Normalize sentence to same format.
    Make sure sentences are lower characters and no punctuations.
    For chinese, use pinyin to compute error rate.
"""
import re
import string
import inflect
import pinyin
import cn2an
from zhon.hanzi import punctuation


class Normalizer(object):
    """Normalize sentence to same format."""

    def __init__(self):
        punctuations = string.punctuation.replace("'", "") + punctuation
        self.punctuation_table = str.maketrans(dict.fromkeys(punctuations))
        self.p = inflect.engine()

    def normalize_sentence(self, sentence):
        """lower, no punctuations"""

        # Delete punctuations.
        sentence = sentence.translate(self.punctuation_table)
        # Convert digit into english.
        digits = re.findall(r'\d+', sentence)
        for digit in digits:
            sentence = sentence.replace(digit, self.p.number_to_words(digit), 1)
        # remove white space in two end of string
        sentence = sentence.lstrip().rstrip()
        # Convert all characters to lower.
        sentence = sentence.lower()
        return sentence

    def normalize_sentence_cn(self, sentence):
        """
            Convert digit to chinese characters.
            Convert chinese characters to pinyin.
        """

        # Convert digit to chinese characters.
        sentence = cn2an.transform(sentence, "an2cn")
        # Delete punctuations.
        sentence = sentence.translate(self.punctuation_table)
        # Convert chinese characters to pinyin.
        sentence = pinyin.get(sentence, format="strip", delimiter=" ")
        # remove white space in two end of string
        sentence = sentence.lstrip().rstrip()

        return sentence
