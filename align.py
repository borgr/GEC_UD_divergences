import json
from collections import OrderedDict
import os
from munkres import Munkres, print_matrix
import re
import numpy as np
import distance
EMPTY_WORD = "emptyWord"


def preprocess_word(word):
    """standardize word form for the alignment task"""
    return word.strip().lower()


def regularize_word(word):
    """changes structure of the word to the same form (e.g. lowercase)"""

    # remove non-alphanumeric
    pattern = re.compile(r"[\W_]+")
    word = pattern.sub("", word)
    return word


def word_tokenize(s):
    """tokenizes a sentence to words list"""
    return re.split(r"\W", s)


def align(sen1, sen2, string=True):
    """finds the best mapping of words from one sentence to the another
    string = a boolean representing if sentences are given as strings or as list of ucca terminal nodes
    returns list of word tuples and the corresponding list of indexes tuples"""
    if string:
        sen1 = list(map(preprocess_word, word_tokenize(sen1)))
        sen2 = list(map(preprocess_word, word_tokenize(sen2)))
    else:
        sen1 = [preprocess_word(terminal.text) for terminal in sen1]
        sen2 = [preprocess_word(terminal.text) for terminal in sen2]

    # find lengths
    length_dif = len(sen1) - len(sen2)
    if length_dif > 0:
        shorter = sen2
        longer = sen1
        switched = False
    else:
        shorter = sen1
        longer = sen2
        switched = True
        length_dif = abs(length_dif)
    shorter += [EMPTY_WORD] * length_dif

    # create matrix
    matrix = np.zeros((len(longer), len(longer)))
    for i in range(len(longer)):
        for j in range(len(longer) - length_dif):
            matrix[i, j] = distance.levenshtein(
                longer[i], shorter[j]) + float(abs(i - j)) / len(longer)
    if matrix.size > 0:
        # compare with munkres
        m = Munkres()
        indexes = m.compute(matrix)
    else:
        indexes = []
    # remove indexing for emptywords and create string mapping
    refactored_indexes = []
    mapping = []
    start = 0 if string else 1
    for i, j in indexes:
        if j >= len(longer) - length_dif:
            j = -1 - start
        if switched:
            refactored_indexes.append((j + start, i + start))
            mapping.append((shorter[j], longer[i]))
        else:
            refactored_indexes.append((i + start, j + start))
            mapping.append((longer[i], shorter[j]))
    return mapping, refactored_indexes

def get_tokenized(conllu):
    res = []
    tokens = []
    with open(conllu) as fl:
        for line in fl:
            id_symb="# sent_id ="
            if line.startswith(id_symb):
                sent_id = line.split(id_symb)[-1]
            elif not line.strip():
                assert (not res) or sent_id != res[-1][0]
                res.append((sent_id, tokens))
                tokens = []
            elif not line.startswith("#"):
                tokens.append(line.split("\t")[1])
    return res

if __name__ == '__main__':
    for root, dirs, filenames in os.walk("data/UD_English-ESL/data"):
        for filename in filenames:
            corrected_path = os.path.join(root, "corrected", filename.replace("_esl", "_cesl"))
            if filename.startswith("en_esl") and os.path.isfile(corrected_path):
                esl_path = os.path.join(root, filename)
                print("aligning", esl_path)
                esl_tokenized = get_tokenized(esl_path)
                cesl_tokenized = get_tokenized(corrected_path)
                res = []
                for (esl_id, esl), (cesl_id, cesl) in zip(esl_tokenized, cesl_tokenized):
                    assert(esl_id == cesl_id)
                    res.append((esl_id, align(" ".join(esl), " ".join(cesl))))
                with open(esl_path + ".align.json", "w") as fl:
                    json.dump(res, fl, indent=1)



