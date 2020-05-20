'''
    Script to split csv data into train and test dataset
    Note: Considering input csv have no headers in it
    Command to run:
        python generate_train_test_split_from_csv.py
         --csv_file=<PATH TO THE BIG CSV FILE>
         --output_dir=<PATH TO THE OUTPUT DIRECTORY>
         --test_portion=<FLOAT VALUE DENOTING THE RATIO OF TEST SET>
'''

#importing
import csv
import os
import random
from collections import defaultdict

import argparse

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--csv_file", help="path to the csv file",
                    type=str)
parser.add_argument("--output_dir", help="path to the output directory",
                    type=str)
parser.add_argument("--test_portion", help="float value denoting portion of test_set default = 0.2",
                    type=float, default=0.2)
args = vars(parser.parse_args())

def get_csv_data(csv_file_path):
    '''
        script to read csv and return csv data in 2d list and a dictionary containing
        {file_name : total_annotation_count_on_file}
        parms:
            csv_file_path : <PATH TO THE CSV FILE>
    '''
    annotation_count_dict = defaultdict(int)

    # check file exists or not
    if not os.path.exists(csv_file_path):
        raise f'file not found : {csv_file_path}'

    csv_data = []

    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            csv_data.append(row)

    # creating dictionary {filename : total_annotations_for_that_file}
    for csv_row_data in csv_data:
        annotation_count_dict[csv_row_data[0]] += 1

    return (csv_data, annotation_count_dict)

def get_filenames_of_test_set(annotation_count_dict, test_portion):
    '''
        Process annotation data and return name of those files
        which contributes to test_portion of the dataset
        parms:
            annotation_count_dict : dictionary containing filename as a key
                                    and total annotation as value
            test_portion : test_set fraction value
    '''

    total_test_annotations = 0
    expected_annotations = int(sum(annotation_count_dict.values()) * test_portion)

    padding = 0.01

    total_attempt = 0

    while not expected_annotations-padding*expected_annotations \
            < total_test_annotations < \
            expected_annotations+padding*expected_annotations:

        total_attempt += 1

        print(f'{total_attempt} - attempting to find optimal test set')

        test_set = random.sample(annotation_count_dict.keys(),
                                 int(test_portion*len(annotation_count_dict.keys())))

        total_test_annotations = sum([annotation_count_dict[filename] for filename in test_set])

        if total_attempt % 10 == 0:
            padding += 0.01

        if total_attempt % 100 == 0:
            print('Unable to find optimize test_set please check yous csv file')
            break

    print('Test set found successfully')
    return set(test_set)

def generate_train_test_csv_data(csv_data, test_set):
    '''
        return 2d list containing train test data
        params:
            csv_data : input csv data
            test_set :  filenames of files in test set
    '''
    train_csv_data = [['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']]
    test_csv_data = [['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']]

    for row_data in csv_data:
        # converting into int
        row_data[1] = int(row_data[1])
        row_data[2] = int(row_data[2])
        row_data[3] = int(row_data[3])
        row_data[4] = int(row_data[4])

        if row_data[0] in test_set:
            row_data[0] = row_data[0].split('.')[0] + '.jpg'
            test_csv_data.append(row_data)
        else:
            row_data[0] = row_data[0].split('.')[0] + '.jpg'
            train_csv_data.append(row_data)

    return (train_csv_data, test_csv_data)

# entry point
if __name__ == "__main__":
    print('processing_file...')

    csv_data, annotation_count_dict = get_csv_data(args['csv_file'])
    test_set = get_filenames_of_test_set(annotation_count_dict, args['test_portion'])
    train_csv_data, test_csv_data = generate_train_test_csv_data(csv_data, test_set)

    print('Total test images :', len(test_set))
    print('generating train_labels.csv')

    with open(os.path.join(args['output_dir'], 'train_labels.csv'), 'w+') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(train_csv_data)


    print('generating test_labels.csv')

    with open(os.path.join(args['output_dir'], 'test_labels.csv'), 'w+') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(test_csv_data)

    print('process completed successfully')
