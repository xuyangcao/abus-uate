import os

def gen_list(image_path, save_filename):
    filenames = [filename for filename in os.listdir(image_path) if filename.endswith('png')]
    with open(save_filename, 'w') as f:
        for idx, filename in tqdm(enumerate(filenames)):
            f.write(filename+'\n')

def main():
    #test_data_path = './test_data_2d/'
    #test_list_name = './test.labeled'
    #gen_list(test_data_path, test_list_name)

    train_data_path = './train_data_2d/'
    train_list_name = './train.list'
    gen_list(train_data_path, train_list_name)

if __name__ == "__main__":
    main()

