import sys
import itertools
import json
import pandas as pd
import numpy as np
from collections import Counter
from queue import Queue
from itertools import combinations as combs

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
                    index +=1
                errors = []
            elif line[0] == 'A':
                error_type, (start_index, end_index), correction = get_error_type_russian(line)
                errors.append([error_type, (start_index, end_index), correction])
            elif line == '\n':
                sentence_id = add_results_russian(errors, res, sentence, sentence_id)
                errors = []
            line = m2_file.readline()
    return res

def get_error_type_russian(line):
    split_line = line.split("|||")
    start_index = int(split_line[0].split()[1])
    end_index = int(split_line[0].split()[2])
    error_type = split_line[1]
    correction = split_line[2]
    return (error_type, (start_index, end_index), correction)

def add_results_russian(errors, res, sentence, sentence_id):
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
        if error[0] == "Вставить": #insertion
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
                    merged_replacement += " "+sentence[i][0]
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
        line = m2_file.readline()
        errors = []
        sentence = []
        while line:
            if (line[0] == 'S'):
                if sentence:  # its new sentence sentence
                    sentence_id = add_results(errors, res, sentence, sentence_id)
                words_in_sentence = line.split()[1:]
                sentence = []  # clear previous sentence
                index = 0
                while (index < len(words_in_sentence)):
                    sentence.append([words_in_sentence[index]])
                    index += 1
                errors = []  # clear previous errors
                line = m2_file.readline()
                continue
            elif (line[0] == 'A'):  # error line
                if (line[2] == '-'):  # no error actually
                    errors = []
                    line = m2_file.readline()
                    continue
                else:  # actually error occurred - separate between missing, replacement, unneccessary and adjust the
                    error_type, (start_index, end_index), correction = get_error_type(line)
                    errors.append([error_type, (start_index, end_index), correction])
                    line = m2_file.readline()
                    continue
            else:
                line = m2_file.readline()
                continue
        add_results(errors, res, sentence, sentence_id)
    return res


def add_results(errors, res, sentence, sentence_id):
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


def get_error_type(line):
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
                sent_id = int(sent_id[0:-1])
            elif line == '\n' or (counter == 10000 and line.startswith("# newdoc")): #add becuase afte 10,000 sentence
                # there is #newdoc without a new line, shifting all sentences
                assert (not res) or sent_id != res[-1][0]
                res.append((sent_id, tokens))
                tokens = []
            elif not line.startswith("#"):
                tokens.append(line.split("\t")[1])
        res.append((sent_id, tokens)) #if there is a blank space in the end of the file, remove!
    return res


def get_alignments(alignments, esl_tokenized, cesl_tokenized, comparison):
    """
    This function gets the alignments between the original and corrected sentences
    :param alignments: list of dictionaries (representing alignment of each sentence)
    :param esl_tokenized: tokenized original sentence
    :param cesl_tokenized: tokenized corrected sentence
    :param comparison: errors and list of corrected vs. original sentences
    :return: nothing
    """
    assert len(esl_tokenized) == len(cesl_tokenized) == len(comparison), "len 1:" + str(
        len(esl_tokenized)) + "len2: " + str(len(cesl_tokenized)) + "len3:" + str(len(comparison))
    for i in range(len(comparison)):
        align_dict = {}
        index = 1
        osentence = esl_tokenized[i][1]
        csentence = cesl_tokenized[i][1]
        shift = 0
        if csentence == ['-']:
            for k in range(1,len(osentence) +1):
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
                            align_dict[str(index + k)] = [str(index + shift - 1)]
                shift -=  len(w)
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
            elif len(cw) == 2 and len(w) == 2 and cw[0].lower() == w[1].lower(): # word order
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
                            align_dict[str(index + k)] = [str(index + shift + k)]
                elif len(cw) < len(w):
                    for k in range(len(cw)):
                        if k == 0:
                            val.append(str(index + shift))
                            align_dict[str(index)] = val
                        else:
                            align_dict[str(index + k)] = [str(index + shift + k)]
                    for l in range(len(w) - len(cw)):
                        align_dict[str(index +len(cw)+ l)] = [str(index + shift + len(cw) - 1)]
                    shift -= (len(w) - len(cw))
                else:
                    for k in range(len(w)):
                        if k == 0:
                            val.append(str(index + shift))
                            align_dict[str(index)] = val
                        else:
                            align_dict[str(index + k)] = [str(index + shift + k)]
                    kval = align_dict[str(index + len(w) - 1)]
                    for l in range(len(cw) - len(w)):
                        kval.append(str(index + shift + len(w) + l))
                    align_dict[str(index + len(w) - 1)] = kval
                    shift += (len(cw) - len(w))
                index += len(w)

        alignments.append(align_dict)
        # check that all indexes are in the range
        for key, val in align_dict.items():
            assert (int(key)) <= len(osentence), str(i) + " " +str(csentence) + " "+ str(key) + " " + str(val)
            assert (int(key)) > 0, str(i) + " " +str(csentence) + " "+ str(key) + " " + str(val)
            for j in val:
                assert (int(j)) <= len(csentence), str(i) + str(csentence) + str(key) + str(val)
                assert (int(j)) > 0 , str(i) + " " +str(csentence) + " "+ str(key) + " " + str(val)


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


def get_confusion_matrix(en, ko, alignments, confusion_dict_pos, confusion_dict_paths):
    assert len(en) == len(ko) == len(alignments), "len en: " + str(len(en)) + " len of ko: "+ \
                                                  str(len(ko)) + " len all: "+ str(len(alignments))

    strip_direction = lambda x: x.split("_")[0]

    for i in range(len(en)):
        en_n, en_g = conll2graph(en[i])
        ko_n, ko_g = conll2graph(ko[i])
        alignment = alignments[i]
        # Simplify the alignment to a set of one-to-one pairs
        one_to_one = []
        for k, v in alignment.items():
            if k == "X":
                # Do not analyse stuff added on the Ko side for now
                continue
            head = k
            tail = str(highest_or_none(v, ko_g))
            one_to_one.append((head, tail))
        # POS confusion dict
        for pair in one_to_one:
            head, tail = pair
            # Skip technical additional nodes
            if "." in head:
                continue
            try:
                en_pos = en_n[head]["pos"]
            except KeyError:
                print(i, en[i])
                continue
            if tail == "None":
                ko_pos = "None"
            else:
                ko_pos = ko_n[tail]["pos"]
            if en_pos not in confusion_dict_pos:
                confusion_dict_pos[en_pos] = Counter()
            confusion_dict_pos[en_pos][ko_pos] += 1
        # Path confusion dict
        for pair in combs(one_to_one, 2):
            (en_head, ko_head), (en_tail, ko_tail) = pair
            # Skip technical additional nodes
            if "." in head:
                continue
            en_path_arr = get_path(en_head, en_tail, en_g)
            if len(en_path_arr) > 1:
                continue
            en_path = strip_direction(en_path_arr[0])
            if ko_head == ko_tail:
                ko_path = "Nodes collapsed"
            elif ko_head == "None" and ko_tail == "None":
                ko_path = "Both endpoints unaligned"
            elif ko_head == "None" or ko_tail == "None":
                ko_path = "One endpoint unaligned"
            else:
                ko_path_arr = get_path(ko_head, ko_tail, ko_g)
                ko_path = "->".join(list(map(strip_direction, ko_path_arr)))
            if en_path not in confusion_dict_paths:
                confusion_dict_paths[en_path] = Counter()
            confusion_dict_paths[en_path][ko_path] += 1


def extract_matrices(confusion_dict_paths, confusion_dict_pos, data_name):
    confusion_dict2matrix(confusion_dict_pos).to_csv(data_name+"_pos_matrix.csv", index=False)
    confusion_dict2matrix(confusion_dict_paths).to_csv(data_name+"_path_matrix.csv", index=False)


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
        en = parse_conllu(conllu_path)
        ko = parse_conllu(conllu_path_corrected)
        assert len(en) == len(ko) == len(alignments), "len en: " + str(len(en)) + " len of ko: "+ \
                                                      str(len(ko)) + " len all: "+ str(len(alignments))
        confusion_dict_paths = {}
        confusion_dict_pos = {}
        get_confusion_matrix(en, ko, alignments, confusion_dict_pos, confusion_dict_paths)
        extract_matrices(confusion_dict_paths, confusion_dict_pos, conllu_path.split('/')[-1])


if __name__ == '__main__':
    main()
