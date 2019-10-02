import sys
import os
import GEC_UD_divergences.GEC_UD_divergences_m2

def preprocess_file(filename, output_directory):
    """
    This preprocesses each file, writing it into sentences in a new output file in the output directory
    :param filename:
    :param output_directory:
    :return:
    """
    partial_name = filename.split('/')[-1]
    data_name = partial_name.split('.')
    output_file_path = output_directory + '/' + data_name[0] +"."+data_name[1]
    with open(filename) as file:
        with open(output_file_path, "w+") as output_file:
            line = file.readline()
            while line:
                if (line[0] == 'S'):
                    output_file.write(line[2:])
                line = file.readline()

def create_corrected_sentences(filename, output_directory):
    """
    creates files with corrected sentences (one perline) especially for running udpipe
    :param filename: m2 file
    :param output_directory: output directory (where output files will be created)
    :return: nothing
    """
    partial_name = filename.split('/')[-1]
    data_name = partial_name.split('.')
    output_file_path = output_directory + '/' + data_name[0] +"."+data_name[1] +".corrected"
    results = GEC_UD_divergences.GEC_UD_divergences_m2.get_annotation_from_m2(filename)
    with open(output_file_path, "w+") as output_file:
        for res in results:
            sentence = ''
            for word in res[2]:
                sentence += word[0]
                sentence += ' '
            if sentence == '' or sentence == ' ':
                output_file.write("-" + '\n')
            else:
                output_file.write(sentence + '\n')


def main():
    """
    This gets a folder with m2 files and creates a preprocessed file (only sentences, one per line)
    :return: nothing
    """
    directory = sys.argv[1]
    output_directory = sys.argv[2]
    for file in os.listdir(directory):
        filename = directory +'/'+os.fsdecode(file)
        # preprocess_file(filename, output_directory)
        create_corrected_sentences(filename, output_directory)


if __name__ == '__main__':
    main()