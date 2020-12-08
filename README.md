# GEC_UD_divergences
* For the use of this code you will likely want to have a manual parse or a parser (e.g. [udpipe][http://ufal.mff.cuni.cz/udpipe]).
* To extract m2 files, one would like to already have m2 files (possibly with non-sense types), those can be extracted manually or automatically (e.g. [errant][https://github.com/chrisjbryant/errant])

## Create a matrix
See Using_m2, currently the file creates both matrices and m2 in the main function, comment out if unneeded (matrix is get_confusion_matrix + extract_matrices)
python using_m2/GEC_UD_divergences_m2.py source.conllu reference.conllu m2file.m2

## Create an m2 file
See Using_m2, currently the file creates both matrices and m2 in the main function, comment out if unneeded (m2 is in syntactic_m2 function)
python using_m2/GEC_UD_divergences_m2.py source.conllu reference.conllu m2file.m2

### Other implementation:
There is also an end to end implementation for English with or without combining with ERRANT (adaptations to support other languages or morphology when POS is not changing are welcome)

## Utility: convert m2 to source and reference text
python using_m2 preprocessing.py m2file.m2


## Cite
If you have found this work useful please cite the CoNLL paper.

```@inproceedings{choshen-etal-2020-classifying,
    title = "Classifying Syntactic Errors in Learner Language",
    author = "Choshen, Leshem  and
      Nikolaev, Dmitry  and
      Berzak, Yevgeni  and
      Abend, Omri",
    booktitle = "Proceedings of the 24th Conference on Computational Natural Language Learning",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.conll-1.7",
    pages = "97--107",
    }
```

