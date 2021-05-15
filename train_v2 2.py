from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import argparse

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras import optimizers


def load_data_folder(folder, set_index=-1):

    csv_file_name = folder / 'sequence_info_by_frame.csv'
    df = pd.read_csv(csv_file_name)
    img_file_names = sorted(folder.glob('seq???.png'))
    df['set_index'] = set_index
    df['file_name'] = [str(f.resolve()) for f in img_file_names]

    return df


def load_all_data(base_folder, split_by='set', seed=None):

    set_names = sorted(base_folder.glob('set?'))
    columns = ['frame', 'object0_present', 'object1_present',
               'set_index', 'file_name']
    df = pd.DataFrame(columns=columns)

    print()
    print('loading all data into dataframe...')
    print()

    for idx, folder in enumerate(set_names):

        print('loading data from set {} of {}'.format(idx+1, len(set_names)))
        df_partial = load_data_folder(folder, set_index=idx)
        df = pd.concat([df, df_partial])

    df['is_train'] = False
    df['is_val'] = False
    df['is_test'] = False

    if seed:
        np.random.seed(seed)

    print()
    print('loading data into memory...')

    all_data_imgs = np.array([cv2.imread(fname, cv2.IMREAD_GRAYSCALE)[..., np.newaxis]
                              for fname in df.file_name])
    print('imgs shape: ', all_data_imgs.shape)
    all_data_labels = df.loc[:, ['object0_present', 'object1_present']].values
    print('labels shape: ', all_data_labels.shape)
    print('done.')
    print()

    if split_by == 'set':

        set_indices = df['set_index'].unique()
        set_indices.sort()
        n_sets = len(set_indices)

        reordered_sets = np.random.choice(set_indices, n_sets, replace=False)
        test_indices = reordered_sets[-1:]
        val_indices = reordered_sets[-2:-1]
        train_indices = reordered_sets[:-2]

        df['is_train'] = df['set_index'].isin(train_indices)
        df['is_val'] = df['set_index'].isin(val_indices)
        df['is_test'] = df['set_index'].isin(test_indices)

    elif split_by == 'random':

        n_rows = len(df)
        train_prob = 0.6
        val_prob = 0.2
        test_prob = 0.2
        data_group_index = np.random.choice(3, n_rows, p=[train_prob, val_prob, test_prob])

        df['is_train'] = (data_group_index == 0)
        df['is_val'] = (data_group_index == 1)
        df['is_test'] = (data_group_index == 2)

    else:

        print('split by {} not implemented'.format(split_by))

    x_train = all_data_imgs[df['is_train']]
    y_train = all_data_labels[df['is_train']].astype(int)

    x_val = all_data_imgs[df['is_val']]
    y_val = all_data_labels[df['is_val']].astype(int)

    x_test = all_data_imgs[df['is_test']]
    y_test = all_data_labels[df['is_test']].astype(int)

    train_data = (x_train, y_train)
    val_data = (x_val, y_val)
    test_data = (x_test, y_test)

    return train_data, val_data, test_data


def build_model(activation_function, n_features_conv, input_shape, n_features_dense=[128, 64],
                n_output_classes=2):

    model = Sequential()

    block_index = 0
    conv_index = 0
    conv = Conv2D(n_features_conv[block_index], 3, padding='same', strides=(1, 1),
                  name='block{}_conv{}'.format(block_index+1, conv_index+1),
                  activation=activation_function, input_shape=input_shape)
    model.add(conv)
    pool = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same',
                     name='maxpool{}'.format(block_index))
    model.add(pool)

    block_index += 1
    conv_index = 1
    conv = Conv2D(n_features_conv[block_index], 3, padding='same', strides=(1, 1),
                  name='block{}_conv{}'.format(block_index+1, conv_index+1),
                  activation=activation_function)
    model.add(conv)
    pool = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same',
                     name='maxpool{}'.format(block_index))
    model.add(pool)

    block_index += 1
    conv_index = 1
    conv = Conv2D(n_features_conv[block_index], 3, padding='same', strides=(1, 1),
                  name='block{}_conv{}'.format(block_index+1, conv_index+1),
                  activation=activation_function)
    model.add(conv)
    pool = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same',
                     name='maxpool{}'.format(block_index))
    model.add(pool)

    block_index += 1
    conv_index = 1
    conv = Conv2D(n_features_conv[block_index], 3, padding='same', strides=(1, 1),
                  name='block{}_conv{}'.format(block_index+1, conv_index+1),
                  activation=activation_function)
    model.add(conv)
    pool = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same',
                     name='maxpool{}'.format(block_index))
    model.add(pool)


    model.add(Flatten(name='flatten'))

    dense_block_index = 0
    dense = Dense(n_features_dense[dense_block_index],
                  name='dense{}'.format(dense_block_index+1),
                  activation=activation_function)
    model.add(dense)

    dense_block_index += 1
    dense = Dense(n_features_dense[dense_block_index],
                  name='dense{}'.format(dense_block_index+1),
                  activation=activation_function)
    model.add(dense)

    classifier = Dense(n_output_classes,
                       name='classifier',
                       activation='sigmoid')
    model.add(classifier)

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', help='number of training epochs',
                        type=int, default=10)
    parser.add_argument('-b', '--batch_size', help='training batch size',
                        type=int, default=5)
    parser.add_argument('-f', '--folder', help='root data folder',
                        type=str, required=True)
    parser.add_argument('-l', '--learning_rate', help='learning rate',
                        type=float, default=1e-4)  
    parser.add_argument('-a', '--activ_func', help='activation function',
                        type=str, default='relu')

    args = parser.parse_args()
    n_epochs = args.epochs
    batch_size = args.batch_size
    lrate = args.learning_rate
    activation_function=args.activ_func

    base_folder = Path(args.folder)
    if not base_folder.exists():
        print('please include a valid path to data files')

    # load the data
    train_data, val_data, test_data = load_all_data(base_folder)
    x_train, y_train = train_data
    x_val, y_val = val_data
    x_test, y_test = test_data

    INPUT_SHAPE = (150, 128, 1)
    NUMBER_FEATURES_CONV = [16, 32, 64, 128]
    NUMBER_FEATURES_DENSE = [128, 64]

    # load the model
    model = build_model(activation_function, NUMBER_FEATURES_CONV, input_shape=INPUT_SHAPE,
                        n_features_dense=NUMBER_FEATURES_DENSE)

    # set optimizer and loss
    opt = optimizers.SGD(lr=lrate, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    # model info
    model.summary()

    # train
    model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size,
              validation_data=(x_val, y_val))

    # test
    print()
    print()
    print('now evaluating on the test set:')
    final_result = model.evaluate(x_test, y_test)
    print('final_result = ', final_result)
    print()
