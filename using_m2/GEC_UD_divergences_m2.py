import os
import re
import sys
from collections import Counter
from html import unescape
from itertools import combinations as combs
from queue import Queue
from string import punctuation

import numpy as np
import pandas as pd
import six


def get_annotation_from_m2_russian(m2_path):
    res = []
    sentence_id = 1
    with open(m2_path) as m2_file:
        line = m2_file.readline()
        errors = []
        sentence = []
        while line:
            if line[0] == 'S':
                # if sentence:
                #     sentence_id = add_results_russian(errors, res, sentence, sentence_id)
                words_in_sentence = line.split()[1:]
                sentence = []
                index = 0
                while index < len(words_in_sentence):
                    sentence.append([words_in_sentence[index]])
                    index += 1
                errors = []
            elif line[0] == 'A':
                error_type, (start_index, end_index), correction = get_error_type_general(
                    line)
                errors.append(
                    [error_type, (start_index, end_index), correction])
            elif line == '\n':
                sentence_id = add_results_general(
                    errors, res, sentence, sentence_id)
                errors = []
            line = m2_file.readline()
    return res


def get_error_type_general(line):
    split_line = line.split("|||")
    start_index = int(split_line[0].split()[1])
    end_index = int(split_line[0].split()[2])
    error_type = split_line[1]
    correction = split_line[2]
    return (error_type, (start_index, end_index), correction)


def add_results_general(errors, res, sentence, sentence_id):
    """
    adds results of m2 corrections in a single sentencce
    :param errors: list of errors and their type/location
    :param res: results list
    :param sentence: sentence (each word a different list)
    :param sentence_id: sentence id
    :return: sentence id (of the next sentence)
    """
    index_shift = 0
    corrected_sentence = sentence
    original_sentence = sentence
    for error in errors:
        if error[0] == "Вставить":  # insertion
            if (error[1][0] - error[1][1]) == 0:
                original_sentence = original_sentence[0:error[1][0] + index_shift] + [[""]] + \
                    original_sentence[error[1][1] + index_shift:]
                corrected_sentence = corrected_sentence[0:error[1][0] + index_shift] + [[error[2]]] + \
                    corrected_sentence[error[1][1] + index_shift:]
                index_shift += 1
            else:
                corrected_sentence = corrected_sentence[0:error[1][0] + index_shift] + [
                    [error[2]]] + corrected_sentence[error[1][1] + index_shift:]

        else:
            corrected_sentence = corrected_sentence[0:error[1][0] + index_shift] + [[error[2]]] + \
                corrected_sentence[error[1][1] + index_shift:]
            if error[1][1] - error[1][0] > 1:
                merged_replacement = ""
                for i in range(error[1][0], error[1][1]):
                    merged_replacement += " " + sentence[i][0]
                original_sentence = original_sentence[0:error[1][0] + index_shift] + [[merged_replacement]] + \
                    original_sentence[error[1][1] + index_shift:]
                index_shift -= (error[1][1] - error[1][0] - 1)
    is_edits = [0] * len(original_sentence)
    assert len(original_sentence) == len(corrected_sentence)
    for index in range(len(original_sentence)):
        assert len(original_sentence) == len(corrected_sentence)
        if original_sentence[index] != corrected_sentence[index]:
            is_edits[index] = 1
    res.append((sentence_id, original_sentence, corrected_sentence, is_edits))
    sentence_id += 1
    return sentence_id


def get_annotation_from_m2(m2_path):
    """
    gets annotation from m2 file as extracted for the m2 scorer manually (e.g. NUCLE) or automatically (e.g. ERRANT)
    :param m2_path: filepath of M2 file
    :return: array of results:
    format of results [[[sentence id],[original sentence (separated by words)], [corrected sentence (separated by words)], [edits (0 if no edits
    in a specific place, 1 if there was an edit)]], etc..]
    """
    res = []
    sentence_id = 1
    with open(m2_path) as m2_file:
        errors = []
        sentence = []
        for line in m2_file:
            if (line[0] == 'S'):
                if sentence:  # its new sentence sentence
                    sentence_id = add_results_errant(
                        errors, res, sentence, sentence_id)
                words_in_sentence = line.split()[1:]
                sentence = []  # clear previous sentence
                index = 0
                while (index < len(words_in_sentence)):
                    sentence.append([words_in_sentence[index]])
                    index += 1
                errors = []  # clear previous errors
            elif (line[0] == 'A'):  # error line
                if (line[2] == '-'):  # no error actually
                    errors = []
                else:  # actually error occurred - separate between missing, replacement, unneccessary and adjust the
                    error_type, (start_index,
                                 end_index), correction = get_error_type_errant(line)
                    errors.append(
                        [error_type, (start_index, end_index), correction])
        add_results_errant(errors, res, sentence, sentence_id)
    return res


def add_results_errant(errors, res, sentence, sentence_id):
    """
    adds results of m2 corrections in a single sentencce
    :param errors: list of errors and their type/location
    :param res: results list
    :param sentence: sentence (each word a different list)
    :param sentence_id: sentence id
    :return: sentence id (of the next sentence)
    """
    index_shift = 0
    corrected_sentence = sentence
    original_sentence = sentence
    for error in errors:
        if error[0] == "R" or error[0] == 'U':
            corrected_sentence = corrected_sentence[0:error[1][0] + index_shift] + [[error[2]]] + \
                corrected_sentence[error[1][1] + index_shift:]
            if (error[1][1] - error[1][0] > 1):
                merged_replacement = ""
                for i in range(error[1][0], error[1][1]):
                    merged_replacement += " " + sentence[i][0]
                original_sentence = original_sentence[0:error[1][0] + index_shift] + [
                    [merged_replacement]] + original_sentence[error[1][1] + index_shift:]
                index_shift -= (error[1][1] - error[1][0] - 1)
            continue
        elif error[0] == 'M':
            if (error[1][0] - error[1][1]) == 0:
                original_sentence = original_sentence[0:error[1][0] + index_shift] + [[""]] + original_sentence[
                    error[1][
                        1] + index_shift:]
                corrected_sentence = corrected_sentence[0:error[1][0] + index_shift] + [[error[2]]] + \
                    corrected_sentence[error[1][1] + index_shift:]
                index_shift += 1
            else:
                corrected_sentence = corrected_sentence[0:error[1][0] + index_shift] + [
                    [error[2]]] + corrected_sentence[error[1][1] + index_shift:]
        elif error[0] == "UNK":
            continue  # figure out what should be done here
    is_edits = [0] * len(original_sentence)
    for index in range(len(original_sentence)):
        assert len(original_sentence) == len(corrected_sentence)
        if original_sentence[index] != corrected_sentence[index]:
            is_edits[index] = 1
    res.append((sentence_id, original_sentence, corrected_sentence, is_edits))
    sentence_id += 1
    return sentence_id


def get_error_type_errant(line):
    """
    gets line from M2 file and extracts error types, start and end index of error and correction
    :param line: line from file
    :return: error type (unknown, unneccassary, missing, replacement), start and end index (tuple), correction
    """
    split_line = line.split('|||')
    if split_line[1] == "UNK":
        error_type = "UNK"
    else:
        error_type = split_line[1][0]
    index = split_line[0]
    (start, end) = (int(index.split()[1]), int(index.split()[2]))
    if error_type == 'U':  # unneccessary
        correction = ""
    else:
        correction = split_line[2]
    return error_type, (start, end), correction


def get_tokenized(conllu):
    """
    This function gets a conllu filepath and divides each sentence into tokens
    :param conllu: file path
    :return: format of result: [(sentence id#, [tokens, in, sentence]), etc ]
    """
    res = []
    tokens = []
    with open(conllu) as fl:
        counter = 0
        sent_id = 0
        for line in fl:
            id_symb = "# sent_id = "
            if line.startswith(id_symb):
                counter += 1
                sent_id = line.split(id_symb)[-1]
                try:
                    sent_id = int(sent_id[0:-1])
                except:
                    pass
            # add becuase after 10,000 sentences udpipe count restarts
            elif line == '\n' or (counter == 10000 and line.startswith("# newdoc")):
                # there is #newdoc without a new line, shifting all sentences
                assert (not res) or sent_id != res[-1][0]
                res.append((sent_id, tokens))
                tokens = []
            elif not line.startswith("#"):
                tokens.append(line.split("\t")[1])
        # if there is a blank space in the end of the file, remove!
        res.append((sent_id, tokens))
    return res

def preprocess_word(word):
    """standardize word form for the alignment task"""
    return word.strip().lower()


def regularize_word(word):
    """changes structure of the word to the same form (e.g. lowercase)"""

    # remove non-alphanumeric
    word = unescape(word)
    word = word.lower()
    pattern = re.compile(r"[^\w" + punctuation + "]+")
    word = pattern.sub("", word)
    return word


def word_tokenize(s):
    """tokenizes a sentence to words list"""
    return re.split(r"\W", s)


def count_combine_to_word(tokens, word):
    i = 0
    subword = ""
    reg_word = regularize_word(word)
    while reg_word.startswith(regularize_word(subword)):
        try:
            subword += tokens[i]
        except:
            raise
        i += 1
        if regularize_word(subword) == reg_word:
            return i
    return 0

def retokenize(tokenized, word_lists, spare=None):
    res_tokens = []
    res_ids = []
    token_id = 0
    if len(word_lists) > 2 and word_lists[-1] == ["."] and word_lists[-2] == ["etc"]:
        word_lists[-1] = []
        word_lists[-2] = ["etc", "."]
    for word_list_id, word_list in enumerate(word_lists):
        res_tokens.append([])
        res_ids.append([])
        word_id = 0
        while word_id < len(word_list):
            word = word_list[word_id]
            combine_id = count_combine_to_word(tokenized[token_id:], word)
            combine_tokens_id = count_combine_to_word(
                word_list[word_id:], tokenized[token_id])
            if combine_id is not 0:
                res_tokens[-1] += tokenized[token_id:token_id + combine_id]
                res_ids[-1] += list(range(token_id + 1,
                                          token_id + 1 + combine_id))
                token_id += combine_id
                word_id += 1
            elif combine_tokens_id:
                res_tokens[-1].append(tokenized[token_id])
                res_ids[-1].append(token_id + 1)
                token_id += 1
                word_id += combine_tokens_id
            elif spare:
                if spare != ([], []) and spare != word_lists:
                    return retokenize(tokenized, spare)
                else:
                    raise NotImplementedError
            else:
                print(tokenized, word_lists)
                raise NotImplementedError
    return res_tokens, res_ids

def get_alignments(alignments, esl_tokenized, cesl_tokenized, comparison):
    """
    This function gets the alignments between the original and corrected sentences
    :param alignments: list of dictionaries (representing alignment of each sentence)
    :param esl_tokenized: tokenized original sentence
    :param cesl_tokenized: tokenized corrected sentence
    :param comparison: errors and list of corrected vs. original sentences
    :return: nothing
    """
    assert len(esl_tokenized) == len(cesl_tokenized) == len(comparison), "len 1: " + str(
        len(esl_tokenized)) + "len2: " + str(len(cesl_tokenized)) + "len3: " + str(len(comparison))
    for i in range(len(comparison)):
        align_dict = {}
        index = 1
        osentence = esl_tokenized[i][1]
        csentence = cesl_tokenized[i][1]
        shift = 0
        if csentence == ['-']:
            for k in range(1, len(osentence) + 1):
                align_dict[str(k)] = ['1']
            alignments.append(align_dict)
            continue
        for j in range(len(comparison[i][1])):
            word = comparison[i][1][j]
            cword = comparison[i][2][j]
            w = word[0].split()
            cw = cword[0].split()
            if len(w) == 0:
                if index < len(osentence):
                    val = []
                    for k in range(len(cw)):
                        val.append(str(index + k + shift))
                    align_dict[str(index)] = val
                    shift += len(cw)
                else:
                    if index + shift - 1 >= 1:
                        val = align_dict[str(index - 1)]
                        for k in range(len(cw)):
                            val.append(str(index - 1 + k + shift))
                        align_dict[str(index - 1)] = val
                    shift += len(cw)
            elif len(cw) == 0:
                for k in range(len(w)):
                    if str(index) in align_dict and k == 0:
                        val = align_dict[str(index)]
                        val.append(str(index + shift - 1))
                        align_dict[str(index)] = val
                    else:
                        if (index + shift - 1) < 1:
                            align_dict[str(index + k)] = [str(index + shift)]
                        else:
                            align_dict[str(index + k)
                                       ] = [str(index + shift - 1)]
                shift -= len(w)
                index += len(w)
            elif (len(w) == 1) and (len(cw) == 1):
                if str(index) in align_dict:
                    val = align_dict[str(index)]
                    val.append(str(index + shift))
                    align_dict[str(index)] = val
                else:
                    align_dict[str(index)] = [str(index + shift)]
                index += 1
            elif (len(w) >= 2) and len(cw) == 1:
                for k in range(len(w)):
                    if str(index) in align_dict and k == 0:
                        val = align_dict[str(index)]
                        val.append(str(index + shift))
                        align_dict[str(index)] = val
                    else:
                        align_dict[str(index + k)] = [str(index + shift)]
                shift -= (len(w) - 1)
                index += len(w)
            elif (len(cw) >= 2) and len(w) == 1:
                val = []
                for k in range(len(cw)):
                    val.append(str(index + shift + k))
                if str(index) in align_dict:
                    val.extend(align_dict[str(index)])
                align_dict[str(index)] = val
                index += 1
                shift += len(cw) - 1
            # word order
            elif len(cw) == 2 and len(w) == 2 and cw[0].lower() == w[1].lower():
                if str(index) in align_dict:
                    val = align_dict[str(index)]
                    val.append(str(index + 1 + shift))
                    align_dict[str(index)] = val
                else:
                    align_dict[str(index)] = [str(index + 1 + shift)]
                align_dict[str(index + 1)] = [str(index + shift)]
                index += 2
            elif len(cw) >= 2 and len(w) >= 2:
                if str(index) in align_dict:
                    val = align_dict[str(index)]
                else:
                    val = []
                if len(cw) == len(w):
                    for k in range(len(w)):
                        if k == 0:
                            val.append(str(index + shift))
                            align_dict[str(index)] = val
                        else:
                            align_dict[str(index + k)
                                       ] = [str(index + shift + k)]
                elif len(cw) < len(w):
                    for k in range(len(cw)):
                        if k == 0:
                            val.append(str(index + shift))
                            align_dict[str(index)] = val
                        else:
                            align_dict[str(index + k)
                                       ] = [str(index + shift + k)]
                    for l in range(len(w) - len(cw)):
                        align_dict[str(index + len(cw) + l)
                                   ] = [str(index + shift + len(cw) - 1)]
                    shift -= (len(w) - len(cw))
                else:
                    for k in range(len(w)):
                        if k == 0:
                            val.append(str(index + shift))
                            align_dict[str(index)] = val
                        else:
                            align_dict[str(index + k)
                                       ] = [str(index + shift + k)]
                    kval = align_dict[str(index + len(w) - 1)]
                    for l in range(len(cw) - len(w)):
                        kval.append(str(index + shift + len(w) + l))
                    align_dict[str(index + len(w) - 1)] = kval
                    shift += (len(cw) - len(w))
                index += len(w)

        alignments.append(align_dict)
        # check that all indexes are in the range
        for key, val in align_dict.items():
            assert (int(key)) <= len(osentence), str(i) + " " + \
                str(csentence) + " " + str(key) + " " + str(val)
            assert (int(key)) > 0, str(i) + " " + \
                str(csentence) + " " + str(key) + " " + str(val)
            for j in val:
                assert (int(j)) <= len(csentence), str(i) + \
                    str(csentence) + str(key) + str(val)
                assert (int(j)) > 0, str(i) + " " + str(csentence) + \
                    " " + str(key) + " " + str(val)


def conll2graph(record):
    """Converts sentences described using CoNLL-U format
    (http://universaldependencies.org/format.html) to graphs.
    Returns a dictionary of nodes (wordforms and POS tags indexed
    by line numbers) together with a graph of the dependencies encoded
    as adjacency lists of (node_key, relation_label, direction[up or down])
    tuples."""
    graph = {}
    nodes = {}
    for line in record.splitlines():
        if line.startswith("#"):
            continue
        fields = line.strip("\n").split("\t")
        key = fields[0]
        # Ignore compound surface keys for aux, du, etc.
        # Ignore hidden additional nodes for orphan handling
        if "-" in key or "." in key:
            continue
        wordform = fields[1]
        pos = fields[3]
        parent = fields[6]
        relation = fields[7]
        nodes[key] = {
            "wordform": wordform,
            "pos": pos,
            "relation": relation,
            "parent": parent,
        }
        if key not in graph:
            graph[key] = []
        if parent not in graph:
            graph[parent] = []
        graph[key].append((parent, relation, "up"))
        graph[parent].append((key, relation, "down"))
    return (nodes, graph)


def highest_or_none(indices, graph):
    if indices[0] == "X":
        return None
    min_depth = 1000
    argmin = None
    for i in indices:
        key = str(i)
        depth = get_node_depth(key, graph)
        if depth < min_depth:
            min_depth = depth
            argmin = key
    assert argmin is not None
    return argmin


def get_node_depth(node, graph):
    """A BFS-based implementation."""
    q = Queue()
    q.put(("0", 0))
    visited = set()
    visited.add("0")
    while not q.empty():
        current_node, current_depth = q.get()
        for neighbour, *_ in graph[current_node]:
            if neighbour == node:
                return current_depth + 1
            elif neighbour not in visited:
                q.put((neighbour, current_depth + 1))
            visited.add(neighbour)
    raise IndexError("Target node unreachable")

def regularize_word(word):
    """changes structure of the word to the same form (e.g. lowercase)"""

    # remove non-alphanumeric
    word = unescape(word)
    word = word.lower()
    pattern = re.compile(r"[^\w" + punctuation + "]+")
    word = pattern.sub("", word)
    return word


def cut_tokenized_by_text(text, tokens):
    if not text:
        return 0
    if not isinstance(text, six.string_types):
        text = "".join(text)
    i = 0
    subword = ""
    reg_word = regularize_word(text)
    while reg_word.startswith(regularize_word(subword)):
        try:
            subword += tokens[i]
        except Exception as e:
            raise
        i += 1
        if regularize_word(subword) == reg_word:
            return i
    return 0


def syntactic_m2(src, corr, m2_path, pos=True, out_path=None):
    """
    Converts an m2 file and parsed source and correction conllus and m2 to a syntax based m2 file
    :param src: parsed conllu of the source (see parse_conllu for expected output)
    :param corr: parsed conllu of the correction (see parse_conllu for expected output)
    :param m2_path: path to an m2 file
    :param pos: whether to use pos as edges
    :param out_path: place to write the m2 file, if unspecificed the original m2 location will be preserved and stx extension will be added before the .m2
    :return:
    """
    assert len(src) == len(corr), " len en: " + str(len(src)) + " len of corr: " + \
                                                     str(len(corr)))
    if not out_path:
        out_path = os.path.splitext(m2_path)
        out_path = "".join([out_path[0], ".stx", out_path[1]])
    with open(out_path, "w") as outm2:
        with open(m2_path) as m2:
            i = -1
            for m2_line in m2:
                out_line = m2_line
                if m2_line.strip().startswith("S"):
                    i += 1
                    assert i <= len(src)
                    src_n, src_g = conll2graph(src[i])
                    src_idxs = sorted([int(key) for key in src_n.keys()])
                    src_tokens = [src_n[str(idx)]["wordform"] for idx in src_idxs]
                    corr_n, corr_g = conll2graph(corr[i])
                    corr_idxs = sorted([int(key) for key in corr_n.keys()])
                    corr_tokens = [corr_n[str(idx)]["wordform"] for idx in corr_idxs]
                    source_m2_tokens = m2_line[1:].strip().split()
                    last_source_end = 0
                    src_used_tok = 0
                    corr_used_tok = 0
                elif m2_line.strip().startswith("A") and "noop" not in m2_line:
                    # Remember the case where error only changes spaces
                    error_type, (start_index, end_index), corr_m2_tokens = get_error_type_general(
                        m2_line)

                    if "".join(source_m2_tokens[start_index: end_index]).strip() == "".join(corr_m2_tokens).strip() != "":
                        print("Correction changes only non-punctuation and non-alpha characters", "".join(source_m2_tokens[start_index: end_index]), "->", "".join(corr_m2_tokens), "In sentence:", " ".join(source_m2_tokens))
                    corr_m2_tokens = corr_m2_tokens.strip().split()
                    # calculate start_index for source
                    src_start_tok = cut_tokenized_by_text(source_m2_tokens[last_source_end:start_index], src_tokens[src_used_tok:]) + src_used_tok
                    # calcualte end index for source
                    src_end_tok = cut_tokenized_by_text(source_m2_tokens[start_index: end_index], src_tokens[src_start_tok:]) + src_start_tok
                    # calculate start_index for correction
                    corr_start_tok = cut_tokenized_by_text(source_m2_tokens[last_source_end:start_index], corr_tokens[corr_used_tok:]) + corr_used_tok
                    # calcualte end index for correction
                    corr_end_tok = cut_tokenized_by_text(corr_m2_tokens, corr_tokens[corr_start_tok:]) + corr_start_tok

                    last_source_end = end_index
                    src_used_tok = src_end_tok
                    corr_used_tok = corr_end_tok

                    if pos:
                        if src_end_tok == src_start_tok:
                            head = "None"
                        else:
                            # add 1 as conll starts counting from 1
                            head = str(highest_or_none([str(idx) for idx in range(src_start_tok + 1, src_end_tok + 1)],
                                                       src_g))  # TODO check how it works (including when multiple source words
                        if corr_end_tok == corr_start_tok:
                            tail = "None"
                        else:
                            # add 1 as conll starts counting from 1
                            tail = str(highest_or_none([str(idx) for idx in range(corr_start_tok + 1, corr_end_tok + 1)],
                                                       corr_g))

                        # Skip technical additional nodes
                        if "." in head:
                            continue
                        if head == "None":
                            src_pos = "None"
                        else:
                            src_pos = src_n[head]["pos"]

                        if tail == "None":
                            corr_pos = "None"
                        else:
                            corr_pos = corr_n[tail]["pos"]
                        if head == tail == "None":
                            if start_index == end_index and not corr_m2_tokens:
                                print("Correction replaces empty span by an empty correction", " ".join(source_m2_tokens))
                            else:
                                print("Warning: both head and tail in error is None in sentence:", " ".join(source_m2_tokens))

                        syntactic_error = src_pos + "->" + corr_pos
                        out_line = m2_line.split("|||")
                        out_line[1] = syntactic_error
                        out_line = "|||".join(out_line)
                    else:
                        raise NotImplementedError # edge m2
                elif "noop" in m2_line:
                    pass
                elif m2_line.strip():
                    raise ValueError("Unexpected line, m2 lines should start with A S or be blank")
                outm2.write(out_line)


def get_confusion_matrix(src, corr, alignments, confusion_dict_pos, confusion_dict_paths):
    assert len(src) == len(corr) == len(alignments), "len en: " + str(len(src)) + " len of corr: " + \
                                                     str(len(corr)) + " len all: " + str(len(alignments))

    def strip_direction(x): return x.split("_")[0]

    for i in range(len(src)):
        src_n, src_g = conll2graph(src[i])
        corr_n, corr_g = conll2graph(corr[i])
        alignment = alignments[i]
        # Simplify the alignment to a set of one-to-one pairs
        one_to_one = []
        for k, v in alignment.items():
            if k == "X":
                # Do not analyse stuff added on the Ko side for now
                continue
            head = k
            tail = str(highest_or_none(v, corr_g))
            one_to_one.append((head, tail))
        # POS confusion dict
        for pair in one_to_one:
            head, tail = pair
            # Skip technical additional nodes
            if "." in head:
                continue
            try:
                src_pos = src_n[head]["pos"]
            except KeyError:
                print(i, src[i])
                continue
            if tail == "None":
                corr_pos = "None"
            else:
                corr_pos = corr_n[tail]["pos"]
            if src_pos not in confusion_dict_pos:
                confusion_dict_pos[src_pos] = Counter()
            confusion_dict_pos[src_pos][corr_pos] += 1
        # Path confusion dict
        for pair in combs(one_to_one, 2):
            (src_head, corr_head), (src_tail, corr_tail) = pair
            # Skip technical additional nodes
            if "." in head:
                continue
            src_path_arr = get_path(src_head, src_tail, src_g)
            if len(src_path_arr) > 1:
                continue
            src_path = strip_direction(src_path_arr[0])
            if corr_head == corr_tail:
                corr_path = "Nodes collapsed"
            elif corr_head == "None" and corr_tail == "None":
                corr_path = "Both endpoints unaligned"
            elif corr_head == "None" or corr_tail == "None":
                corr_path = "One endpoint unaligned"
            else:
                corr_path_arr = get_path(corr_head, corr_tail, corr_g)
                corr_path = "->".join(list(map(strip_direction, corr_path_arr)))
            if src_path not in confusion_dict_paths:
                confusion_dict_paths[src_path] = Counter()
            confusion_dict_paths[src_path][corr_path] += 1


def extract_matrices(confusion_dict_paths, confusion_dict_pos, data_name):
    confusion_dict2matrix(confusion_dict_pos).to_csv(
        data_name + "_pos_matrix.csv", index=False)
    confusion_dict2matrix(confusion_dict_paths).to_csv(
        data_name + "_path_matrix.csv", index=False)


def get_path(node1, node2, graph):
    if node1 == node2:
        return []

    # BFS with edge labels for paths
    q = Queue()
    # Remembers where we came from and the edge label
    sources = {}

    q.put(node1)
    visited = set()
    visited.add(node1)

    while not q.empty():
        current = q.get()
        for neighbour, relation, direction in graph[current]:
            if neighbour == node2:
                path = [relation + "_" + direction]
                source = current
                while source != node1:
                    prev_node, prev_relation, prev_direction = sources[source]
                    path.append(prev_relation + "_" + prev_direction)
                    source = prev_node
                return list(reversed(path))
            elif neighbour not in visited:
                sources[neighbour] = (current, relation, direction)
                q.put(neighbour)
            visited.add(neighbour)

    raise ValueError("UD graph is not connected.")


def confusion_dict2matrix(cd):
    """Takes as input a map[string -> Counter[string -> int]].
    Returns a Pandas dataframe."""

    row_keys = sorted(cd)
    column_keys = row_keys + ["Other"]
    conf_matrix = np.zeros((len(row_keys), len(column_keys)), int)
    conf_df = pd.DataFrame(conf_matrix)
    conf_df.index = row_keys
    conf_df.columns = column_keys
    for row_key, counter in cd.items():
        for k, val in counter.items():
            if k in column_keys:
                column_key = k
            else:
                column_key = "Other"
            conf_df.loc[row_key][column_key] += val
    return conf_df


def parse_conllu(conllu_filepath):
    """
    this function parses an conllu file, placing the entire text of a sentence as a string an an array
    :param conllu_filepath: file path of conllu file
    :return: array of strings- each string representing the whole conllu format of the sentence
    """
    parsed = []
    single_sentence = ''
    with open(conllu_filepath) as file:
        for line in file:
            if line.startswith("# sent_id ="):
                if (single_sentence != ''):
                    parsed.append(single_sentence)
                single_sentence = ''
                single_sentence += line
            elif not line.startswith("#") and line != '\n':
                single_sentence += line
        if single_sentence != '':
            parsed.append(single_sentence)
    return parsed


def main():
    """
    This is the main function of the program. Takes two conllu files (the original and corrected)
    and m2 file, gets edits and alignments, and creates
    confusion matrixes (if correction involved same or different POS)
    :return: nothing
    """
    if (len(sys.argv) != 4):
        print("Usage: <conllu file> <conllu file corrected> <m2 file>")
    else:
        conllu_path = sys.argv[1]
        conllu_path_corrected = sys.argv[2]
        m2_path = sys.argv[3]
        esl_tokenized = get_tokenized(conllu_path)
        cesl_tokenized = get_tokenized(conllu_path_corrected)
        comparison = get_annotation_from_m2(m2_path)
        alignments = []  # will be a list of dictionaries
        get_alignments(alignments, esl_tokenized, cesl_tokenized, comparison)
        src = parse_conllu(conllu_path)
        corr = parse_conllu(conllu_path_corrected)
        assert len(src) == len(corr) == len(alignments), "len src: " + str(len(src)) + " len of corr: " + \
            str(len(corr)) + " len all: " + str(len(alignments))
        confusion_dict_paths = {}
        confusion_dict_pos = {}
        syntactic_m2(src, corr, m2_path)
        get_confusion_matrix(src, corr, alignments,
                             confusion_dict_pos, confusion_dict_paths)
        extract_matrices(confusion_dict_paths,
                         confusion_dict_pos, conllu_path.split('/')[-1])


if __name__ == '__main__':
    main()
