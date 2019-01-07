Please reference the following paper if you are using this dataset:

@inproceedings{aa2011,
author = {Yannakoudakis, Helen and Briscoe, Ted and Medlock, Ben},
booktitle = {The 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
title = {{A New Dataset and Method for Automatically Grading ESOL Texts}},
year = {2011}
}

This dataset is released for non-commercial research and educational purposes only, please refer to the licence for terms of use. By downloading and using this dataset you agree to this licence.


The Cambridge Learner Corpus
------------------------------

The dataset is part of the Cambridge Learner Corpus (CLC) (http://www.cup.cam.ac.uk/gb/elt/catalogue/subject/custom/item3646603/Cambridge-International-Corpus-Cambridge-Learner-Corpus/?site locale=en GB). The CLC, developed as a collaborative project between Cambridge University Press and Cambridge Assessment, is a large collection of texts produced by English language learners from around the world, sitting Cambridge Assessment's English as a Second or Other Language (ESOL) examinations (http://www.cambridgeesol.org/).


First Certificate in English (FCE) exams
------------------------------------------

There are 11 different exams represented in the CLC. We release texts produced by learners taking the First Certificate in English (FCE) exam, which assesses English at an upper-intermediate level. The texts, which are anonymised, are annotated using XML and linked to meta-data about the question prompts, the candidates' grades, native language and age. We use the FCE writing component which consists of two tasks (eliciting free-text answers) asking learners to write a letter, a report, an article, a composition or a short story, each one between 120 and 180 words. Answers to each of these tasks are annotated with marks, which have been fitted to a RASCH model (Fischer and Molenaar, 1995) to correct for inter-examiner inconsistency and comparability. In addition, an overall mark (in the range 1--40) is assigned to both tasks (which is the one we use in our experiments). 

We release 1244 scripts, where each script contains the answers to the two FCE tasks produced by the learners (1244 distinct learners). 1141 scripts are taken from the examination year 2000 (used for training our system) and 103 scripts from the examination year 2001 (97 used for testing our system + 6 used for the validity tests). 97 scripts from the examination year 2001 were remarked by 4 senior ESOL examiners, and these marks are also provided with the dataset.


FCE mark scheme for the two tasks (Williams, 2008)
---------------------------------------------------

General considerations:
- Complex sentences must be used to gain credit for range of language. Credit will still be given even if there are mistakes, as long as they do not impede communication.
- Spelling and punctuation are not specifically penalised, but may lower the overall
impression mark.

Task 1 - General points:
- Credit is mainly given for efforts at communication, but candidates are penalised for including irrelevant content and for writing too few words. If they write too many words, only the first 180 are marked.
- Candidates must follow the conventions of letter writing, e.g. salutations, paragraphing, closing formulae.
- Aim is to "achieve a positive effect on the reader".
- Marks are awarded for consistent use of appropriate register.
- Every element of the task should be completed.
- A list of questions or statements in simple sentences is not enough; organisation and cohesion, clear layout, appropriate register, control and accuracy of language are all-important features of task achievement. Some evidence of range of language is also required.

Task 2 - General points:
- Candidates must follow every element of the rubric, and write in the appropriate register.
- If they answer on a set text then they must show evidence of having read and understood one of the set texts.
- Candidates should use the appropriate format for the task, e.g. letter, report, article, and make use of task-specific techniques and linguistic features, e.g. direct quotations for an article and wide range of past tenses for a story.


FCE question prompts
----------------------

The prompts eliciting the free text are provided with the dataset. There is no overlap between the prompts used in 2000 and in 2001. A typical prompt taken from the 2000 training dataset is shown below:

Your teacher has asked you to write a story for the school's English language magazine. The story must begin with the following words: "Unfortunately, Pat wasn't very good at keeping secrets".

The 'head sortkey' field is used to uniquely identify each text. The last three components of the sortkey can be used to identify the prompts. In particular, the prompts are identified by year (2000 or 2001), month (01: June, 02: December, 03: March), and examination code (0101, 0102, 0103). The 'dataset' as well as the 'outliers' (see below) folder contains the scripts separated by examcode_year_month. 


FCE error codes
-----------------

Each script has been manually tagged with information about the linguistic errors committed, using a taxonomy of approximately 80 error types (Nicholls, 2003). Below is a list of example error-coded sentences:

1. In the morning, you are <NS type = 'TV'><i>waken</i><c>woken</c></NS> up by a singing puppy.

In this sentence TV denotes an incorrect tense of verb error, where 'waken' can be corrected to 'woken'.

2. It is a very beautiful place and the people there <NS type='AGV'><i>is</i><c>are</c></NS> very kind and generous.
 
In this sentence AGV denotes a verb agreement error.

3. I will give you all <NS type='MD'><c>the</c></NS> information you need.

In this sentence MD denotes a missing determiner error.

4. [...] which caused me <NS type='FN'><i><NS type='RN'><i>trouble</i><c>problem</c></NS></i><c>problems</c></NS>.

In this sentence we have a nested error; RN denotes a 'replace noun' error, while FN denotes a 'wrong noun form' error.


Validity Tests
---------------

6 high-scoring FCE scripts from the examination year 2001 were modified in one of the following ways:

i. Randomly order:
(a) word unigrams within a sentence
(b) word bigrams within a sentence
(c) word trigrams within a sentence
(d) sentences within a script
ii. Swap words that have the same PoS within a sentence

Using the above modifications, we created 30 'outlier' texts, which were given to an ESOL examiner for marking. These 'outlier' texts as well as their marks are also provided with the dataset.


