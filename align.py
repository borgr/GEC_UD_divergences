from html import unescape
from itertools import zip_longest
import six
from collections.abc import Iterable
import json
from collections import OrderedDict
import os
from munkres import Munkres, print_matrix
import re
import numpy as np
import distance
from string import punctuation
EMPTY_WORD = "emptyWord"


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


def align(sen1, sen2, string=False):
    """finds the best mapping of words from one sentence to the another
    string = a boolean representing if sentences are given as strings or as list of ucca terminal nodes
    returns list of word tuples and the corresponding list of indexes tuples"""
    if string:
        sen1 = list(map(preprocess_word, word_tokenize(sen1)))
        sen2 = list(map(preprocess_word, word_tokenize(sen2)))
    else:
        sen1 = [preprocess_word(terminal) for terminal in sen1]
        sen2 = [preprocess_word(terminal) for terminal in sen2]

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
            id_symb = "# sent_id ="
            if line.startswith(id_symb):
                sent_id = line.split(id_symb)[-1]
            elif not line.strip():
                assert (not res) or sent_id != res[-1][0]
                res.append((sent_id, tokens))
                tokens = []
            elif not line.startswith("#"):
                tokens.append(line.split("\t")[1])
    return res


def join_recursively(lsts, sep=" "):
    res = []
    try:
        for lst in lsts:
            if isinstance(lst, six.string_types):
                res.append(lst)
            else:
                res.append(join_recursively(lst, sep))
        return sep.join(res)
    except TypeError as te:
        return str(lsts)


def get_annotation(path, graceful=False):
    if path.endswith("conllu"):
        return get_annotation_from_conllu(path, graceful)
    elif path.endswith(".m2"):
        return get_annotation_from_m2(path, graceful)
    else:
        raise ValueError("Unknown file extension: " + path)

def get_annotation_from_m2(m2_path, graceful=False):
    """ gets annotation from m2 file as extracted for the m2 scorer manually (e.g. NUCLE) or automatically (e.g. ERRANT)"""
    raise NotImplementedError

def get_annotation_from_conllu(conllu, graceful=False):
    """ gets annotation from conllu file, as found in the fce based error annotation"""
    res = []
    sent_id = None
    with open(conllu) as fl:
        for line in fl:
            id_symb = "# sent_id ="
            if line.startswith(id_symb):
                sent_id = line.split(id_symb)[-1]
                continue

            err_symb = "# error_annotation ="
            if line.startswith(err_symb):
                try:
                    source = []
                    ref = []
                    is_edit = []
                    sent = line
                    # clean tags
                    sent = re.sub(r"\<ns[^\<]*\>", r"<ns>", sent)
                    sent = re.sub(r"\<c[^\<]*\>", r"<c>", sent)
                    sent = re.sub(r"\<i[^\<]*\>", r"<i>", sent)
                    # clean text
                    sent = re.sub(r"\<ns[^\<]*\>([^\<]*?)\</ns\>", r"\1", sent)
                    sent = sent.replace("<", " <")
                    sent = sent.replace("</>", "")
                    sent = sent.replace(">", " ")
                    sent = re.sub(r"\s+", " ", sent)

                    # sent = re.sub(r"\<c [^\<]* \</c \</ns \</i", "</i", sent)
                    # remove nesting
                    for i in range(10):
                        sent = re.sub(
                            r"(\<i[^\<]*)\<ns \<i([^\<]*)\</i (\<c[^\<]*\</c )?\</ns", r"\1\2", sent)
                        sent = re.sub(
                            r"(\<i[^\<]*)\<ns (\<c[^\<]*\</c )?\</ns", r"\1", sent)
                        sent = re.sub(
                            r"(\<c[^\<]*)\<ns (\<i[^\<]*\</i )?\<c([^\<]*)\</c \</ns", r"\1\3", sent)

                    sent = re.sub(r"\s+", " ", sent)
                    # sent = re.sub(r"\</i \<c [^\<]* \</c \</ns ([^\<]*?\</i)", r"\1", sent)
                    # sent = re.sub(r"(\<i [^\<]*?) \<ns [^\<]*?\<i", r"\1", sent)
                    # sent = re.sub(r"\</i \</ns ([^\<]*?)\</i", r"\1 </i", sent)
                    # sent = re.sub(r"\</c \</ns ([^\<]*?)\</c", r"\1 </c", sent)
                    # sent = re.sub(r"\<c ([^\<]*?)\<ns \<c", r"<c \1", sent)

                    sent = re.sub(r"(\</i )+", "</i ", sent)
                    sent = sent.split(err_symb)[-1]
                    words = sent.split()
                    correction = True
                    learner = True
                    error_started = False
                    unr_symbs = []
                    learner_words = []
                    correction_words = []
                    for i, word in enumerate(words):
                        if word.startswith("<ns"):
                            error_started = True
                            correction = True
                            learner = True
                            continue
                        if word.startswith("</ns"):
                            if correction_words:
                                print("Broken xml", sent)
                                ref.append(correction_words)
                                is_edit.append(1)
                                correction_words = []
                            if learner_words:
                                print("Broken xml", sent)
                                source.append(learner_words)
                                ref.append([])
                                is_edit.append(1)
                                learner_words = []
                            correction = True
                            learner = True
                            continue
                        if word.startswith("<c"):
                            if learner == True:
                                pass
                            correction = True
                            learner = False
                            correction_words = []
                            if not words[i - 1].startswith("</i"):
                                # print("empty source", sent)
                                source.append([])
                            assert learner == False, sent
                            continue
                        if word.startswith("</c"):
                            if i + 2 < len(words) and words[i + 2].startswith(("</i")):
                                correction = False
                                correction_words = []
                            else:
                                ref.append(correction_words)
                                is_edit.append(1)

                                correction = False
                                correction_words = []
                                # assert learner==False, sent
                            continue
                        if word.startswith("<i"):
                            learner = True
                            correction = False
                            learner_words = []
                            assert correction == False, sent
                            continue
                        if word.startswith("</i"):
                            # if i > 0 and words[i -1].startswith(("<")):
                            #     continue
                            if not error_started:
                                print(
                                    "Bad xml in sent, error starts from sentence start:", sent)
                                ref = []
                                is_edit = []
                                print("source", source,
                                      join_recursively(source))
                                source = [join_recursively(source).split()]
                            else:
                                source.append(learner_words)
                            learner = False
                            learner_words = []
                            # assert correction==False, sent
                            if not words[i + 1].startswith("<c"):
                                # print("empty correction", sent)
                                ref.append([])
                                is_edit.append(1)
                            continue
                        if unr_symbs:
                            unr_symbs = [
                                symb for symb in unr_symbs if "</" + symb != word]
                            print(word, unr_symbs)
                            assert word.startswith("<"), (word, sent)
                            continue
                        if word.startswith("<"):
                            print("unrecognized symbol:", word)
                            unr_symbs.append(word.lstrip("<"))
                            continue
                        if word.startswith("</"):
                            print(
                                "Warning: unrecognized ending unmatched, this could be due to nesting of unrecognized symbols or closing without opening")
                            continue
                        if learner and correction:
                            source.append([word])
                            ref.append([word])
                            is_edit.append(0)

                        elif learner:
                            learner_words.append(word)
                        elif correction:
                            correction_words.append(word)
                    assert "<" not in join_recursively(
                        source) + join_recursively(ref), (source, ref, sent)
                    assert sent_id is not None
                    assert len(source) == len(
                        ref), (list(zip_longest(source, ref)), sent)
                    res.append((sent_id, source, ref, is_edit))
                    sent_id = None
                except Exception as e:
                    if graceful:
                        print("Exception ", e, "caught on", line)
                        res.append((sent_id, [], [], []))
                    else:
                        raise e
    return res


def main_wordalign():
    for root, dirs, filenames in os.walk("data/UD_English-ESL/data"):
        for filename in filenames:
            corrected_path = os.path.join(
                root, "corrected", filename.replace("_esl", "_cesl"))
            if filename.startswith("en_esl") and os.path.isfile(corrected_path):
                esl_path = os.path.join(root, filename)
                print("aligning", esl_path)
                esl_tokenized = get_tokenized(esl_path)
                cesl_tokenized = get_tokenized(corrected_path)
                res = []
                for (esl_id, esl), (cesl_id, cesl) in zip(esl_tokenized, cesl_tokenized):
                    assert(esl_id == cesl_id)
                    res.append((esl_id, align(esl, cesl)))
                with open(esl_path + ".align.json", "w") as fl:
                    json.dump(res, fl, indent=1)
                idx_alignments = [alignment[1][1] for alignment in res]
                str_idx_alignments = "\n".join((
                    " ".join((
                        "-".join((
                            str(alignment[0]) if alignment[0] != -1 else "X", str(alignment[1]) if alignment[1] != -1 else "X"))
                        for alignment in alignments))
                    for alignments in idx_alignments))
                print("Writing to", esl_path + ".align")
                with open(esl_path + ".align", "w") as fl:
                    fl.write(str_idx_alignments)


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


def main_multiwordalign(autoalign=False, parallel=True):
    """ A function that writes alignments given a path
    autoalign - alignment of multi to multi are automatically to be transformed to one to one (or one to None)
    parallel - if True creates src and target text files, one sentence per line.
    """
    for root, dirs, filenames in reversed(list(os.walk("data/UD_English-ESL/data"))):
        for filename in filenames:
            paralels = []
            corrected_path = os.path.join(
                root, "corrected", filename.replace("_esl", "_cesl"))
            if filename.startswith("en_esl") and os.path.isfile(corrected_path):
                esl_path = os.path.join(root, filename)
                print("aligning", esl_path)
                esl_tokenized = get_tokenized(esl_path)
                cesl_tokenized = get_tokenized(corrected_path)
                comparison = get_annotation(esl_path)
                spare_comp = get_annotation(corrected_path, graceful=True)
                res = []
                spares = {sa_id: (saesl, sacesl, is_edits)
                          for sa_id, saesl, sacesl, is_edits in spare_comp}
                for (esl_id, esl), (cesl_id, cesl), (a_id, aesl, acesl, is_edits) in zip(esl_tokenized, cesl_tokenized, comparison):
                    assert(esl_id == cesl_id == a_id)
                    if a_id in spares:
                        saesl, sacesl, sis_edits = spares[a_id]
                    else:
                        saesl, sacesl, sis_edits = None, None, None

                    tokenized = retokenize(esl, aesl, saesl)
                    ctokenized = retokenize(cesl, acesl, sacesl)
                    if not (len(tokenized[0]) == len(tokenized[1]) == len(
                            ctokenized[0]) == len(ctokenized[1])):
                        tokenized = retokenize(esl, saesl)
                        ctokenized = retokenize(cesl, sacesl)
                    if len(tokenized[0]) != len(is_edits):
                        is_edits = sis_edits
                    if not (len(tokenized[0]) == len(tokenized[1]) == len(
                            ctokenized[0]) == len(ctokenized[1]) == len(is_edits)):
                        print("Error")
                    assert(len(tokenized[0]) == len(tokenized[1]) == len(
                        ctokenized[0]) == len(ctokenized[1]) == len(is_edits)), (tokenized[0], is_edits)
                    if autoalign:
                        tok_al = []
                        id_al = []
                        is_edit_al = []
                        for tok, ctok, tok_id, ctok_id, is_edit in zip(tokenized[0], ctokenized[0], tokenized[1], ctokenized[1], is_edits):
                            if len(tok) < 2 or len(ctok) < 2:
                                tok_al.append((tok, ctok))
                                id_al.append((tok_id, ctok_id))
                                is_edit_al.append(is_edit)
                            else:
                                aligned_tok, rel_aligned_ids = align(tok, ctok)
                                aligned_tok = [([val[0]], [val[1]])
                                               for val in aligned_tok]
                                tok_al += aligned_tok
                                is_edit_al += [is_edit for val in aligned_tok]
                                # calculate absolute ids
                                abs_aligned_ids = []
                                for rel_tok, rel_ctok in rel_aligned_ids:
                                    rel_tok -= 1
                                    rel_ctok -= 1
                                    abs_tok = tok_id[
                                        rel_tok] if rel_tok >= 0 else ""
                                    abs_ctok = ctok_id[
                                        rel_ctok] if rel_ctok >= 0 else ""
                                    abs_aligned_ids.append(
                                        ([abs_tok], [abs_ctok]))
                                id_al += abs_aligned_ids
                        alignment = (tok_al, id_al, is_edit_al)
                    else:
                        is_edit_al = is_edits
                        alignment = (list(zip(tokenized[0], ctokenized[0])), list(
                            zip(tokenized[1], ctokenized[1])), is_edit_al)
                    assert(len(alignment[0]) == len(
                        alignment[1]) == len(alignment[2]))

                    for a, b in alignment[0]:
                        if ([], []) == (a, b):
                            return
                    res.append((esl_id, alignment))
                    if parallel:
                        paralels.append((" ".join([x[0] for x in tokenized[0]if x]), " ".join(
                            [x[0] for x in ctokenized[0] if x])))
                #     res.append((esl_id, align(esl, cesl)))
                if parallel:
                    with open(esl_path + ".ref", "w") as ref_fl:
                        with open(esl_path + ".src", "w") as src_fl:
                            for src, ref in paralels:
                                if src and ref:
                                    src_fl.write(src.strip() + "\n")
                                    ref_fl.write(ref.strip() + "\n")

                ext = ".malign" if not autoalign else ".amalign"
                with open(esl_path + ext + ".json", "w") as fl:
                    json.dump(res, fl, indent=1)
                idx_alignments = [alignment[1][1] for alignment in res]
                str_idx_alignments = "\n".join((
                    " ".join((
                        "-".join((
                            str(alignment[0]) if alignment[0] != -1 else "X", str(alignment[1]) if alignment[1] != -1 else "X"))
                        for alignment in alignments))
                    for alignments in idx_alignments))
                print("Writing to", esl_path + ext)
                with open(esl_path + ext, "w") as fl:
                    fl.write(str_idx_alignments)
if __name__ == '__main__':
    main_multiwordalign(True)
    main_multiwordalign()
