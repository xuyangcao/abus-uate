import os
import shutil 

if __name__ == "__main__":
    no = 2436 

    train_list = './train.'+str(no)+'.labeled'
    src_image = './image/'
    src_label = './label'
    image_path = './image_' + str(no)
    label_path = './label_' + str(no)

    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    with open(train_list, 'r') as f:
        filenames = f.readlines()
    filenames = [filename.replace('\n', '') for filename in filenames]
    for filename in filenames:
        shutil.copy(os.path.join(src_image, filename), os.path.join(image_path, filename)) 
        shutil.copy(os.path.join(src_label, filename), os.path.join(label_path, filename))

    print('done!')
