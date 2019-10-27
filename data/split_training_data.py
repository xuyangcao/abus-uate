import os
from tqdm import tqdm
import argparse 
import numpy as np 
from numpy.random import shuffle 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labeled_num', '-L', type=int, default=487)
    parser.add_argument('--patient_num', '-S', type=int, default=None)
    parser.add_argument('--file', '-F', type=str, default='./train.list')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    # read all filenames in list files
    with open(args.file, 'r') as f:
        filenames = f.readlines()
    filenames = [item.replace('\n', '') for item in filenames]

    # select patient, if None, all patients are selected
    patient_ID = [filename[:4] for filename in filenames]
    set_patient_ID = set(patient_ID)
    selected_patient_ID = list(set_patient_ID)[:args.patient_num]
    print('select {} pathents'.format(len(selected_patient_ID)))
    print('selected pathent IDs:\n', selected_patient_ID)
    print('-'*100)

    # select slices from the above selected patients 
    selected_filenames = [filename for filename in filenames if filename[:4] in selected_patient_ID] 
    shuffle(selected_filenames)
    labeled_filenames = selected_filenames[:args.labeled_num]
    unlabeled_filenames = selected_filenames[args.labeled_num:]
    print('labeled_slice_no:{}, unlabeled_slice_no:{}'.format(len(labeled_filenames), len(unlabeled_filenames)))
    print('selected slices:\n', labeled_filenames)

    with open('train.'+str(args.labeled_num)+'.labeled', 'w') as f:
        for filename in labeled_filenames:
            f.write(filename+'\n')
    with open('train.'+str(args.labeled_num)+'.unlabeled', 'w') as f:
        for filename in unlabeled_filenames:
            f.write(filename+'\n')
    print('done!')



if __name__ == "__main__":
    main()
