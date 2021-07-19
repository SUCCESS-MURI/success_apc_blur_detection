# Author mhatfalv
# This program splits the dataset using the input csv file into correct predictions and incorrect predictions
# TODO can be extended to make formatting corrections
import argparse
import os
import shutil
from _csv import reader

# splits the dataset for correct and incorrect predictions using csv file of results
def split_dataset():
    print("Split Dataset from correct and incorrectly labeled data")
    # Feel free to add more args, or change/remove these.
    parser = argparse.ArgumentParser(description='SUCCESS DATASET PREP')
    # directory output location
    parser.add_argument('--output_dir', type=str)
    # results from test file for prediction
    parser.add_argument('--test_results_file_name', type=str, default=None)
    args = parser.parse_args()
    # now create folders for correct data vs incorectly labeled data
    path_for_correct = os.path.join(args.output_dir, "Correct")
    path_for_incorrect = os.path.join(args.output_dir, "Incorrect")
    os.makedirs(path_for_correct)
    os.makedirs(path_for_incorrect)
    print("Directory for correct data is located in : " + path_for_correct)
    print("Directory for incorrrect data is located in : " + path_for_incorrect)
    # open file in read mode
    # https://thispointer.com/python-read-a-csv-file-line-by-line-with-or-without-header/
    with open(args.test_results_file_name, 'r') as test_results:
        csv_reader = reader(test_results)
        # Iterate over each row in the csv using reader object
        firstrowskip = True
        for row in csv_reader:
            # skip label row
            if firstrowskip:
                firstrowskip = False
                continue
            # check if correct is true or false
            dirName_split = row[0].split("/")
            length = dirName_split.__len__()
            if row[3] == 'TRUE':
                # check if the data we are moving is a directory
                # https://pythonexamples.org/python-check-if-path-is-file-or-directory/
                if os.path.isdir(row[0]):
                    shutil.copytree(row[0], path_for_correct + "/" + dirName_split[length-1])
                else:
                    shutil.copy(row[0], path_for_correct + "/" + dirName_split[length-2] + "_" + dirName_split[length-1])
                print(row[0] + " is correctly predicted")
            else:
                # check if the data we are moving is a directory
                if os.path.isdir(row[0]):
                    shutil.copytree(row[0], path_for_incorrect + "/" + dirName_split[length-1])
                else:
                    shutil.copy(row[0], path_for_incorrect + "/" + dirName_split[length-2] + "_" + dirName_split[length-1])
                print(row[0] + " is incorrectly predicted")
    print("Complete")

if __name__ == "__main__":
    split_dataset()


