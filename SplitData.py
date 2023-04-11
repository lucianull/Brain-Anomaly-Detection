import shutil

if __name__ =='__main__':
    data = Data('data/data/', 'data/train_labels.txt', 'data/validation_labels.txt')
    train_data, train_labels, test_data, test_labels, submit_data = data.LoadData()
    i = 1
    source_path = 'data/data/'
    destination_path_anomaly = 'data/data_for_cnn/train/anomaly/'
    destination_path_normal = 'data/data_for_cnn/train/normal/'
    destination_path_test = 'data/data_for_cnn/test/test/'

    for x in train_labels:
        filename = f'{i:06}'
        if x == 0:
            shutil.copy(source_path + filename + '.png', destination_path_normal + filename + '.png')
        else:
            shutil.copy(source_path + filename + '.png', destination_path_anomaly + filename + '.png')
        i += 1

    for x in test_labels:
        filename = f'{i:06}'
        if x == 0:
            shutil.copy(source_path + filename + '.png', destination_path_normal + filename + '.png')
        else:
            shutil.copy(source_path + filename + '.png', destination_path_anomaly + filename + '.png')
        i += 1

    for i in range(17001, 22150):
        filename = f'{i:06}'
        shutil.copy(source_path + filename + '.png', destination_path_test + filename + '.png')