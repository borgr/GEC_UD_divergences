# GEC_UD_divergences
* For the use of this code you will likely want to have a manual parse or a parser (e.g. [udpipe][http://ufal.mff.cuni.cz/udpipe]).
* To extract m2 files, one would like to already have m2 files (possibly with non-sense types), those can be extracted manually or automatically (e.g. [errant][https://github.com/chrisjbryant/errant])

## Create a matrix
python XXX source.conllu reference.conllu m2file.m2

## Create an m2 file
python XXX source.conllu reference.conllu m2file.m2

## Utility: convert m2 to source and reference text
python using_m2 preprocessing.py m2file.m2
