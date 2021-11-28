import os
import glob
import random

import pandas as pd
import cv2
import tqdm
import numpy as np


class GetTrainTestCSV:
    def __init__(self, dataset_path_list, csv_name, img_format_list, negative_keep_rate=0.1):
        self.data_path_list = dataset_path_list
        self.img_format_list = img_format_list
        self.negative_keep_rate = negative_keep_rate
        self.save_path_csv = r'generate_dep_info'
        os.makedirs(self.save_path_csv, exist_ok=True)
        self.csv_name = csv_name

    def get_csv(self, pattern):
        def get_data_infos(img_path, img_format):
            data_info = {'img': [], 'label': []}
            img_file_list = glob.glob(img_path + '/*%s' % img_format)
            assert len(img_file_list), 'No data in DATASET_PATH!'
            for img_file in tqdm.tqdm(img_file_list):
                label_file = img_file.replace(img_format, 'png').replace('imgs', 'labels')
                if not os.path.exists(label_file):
                    label_file = 'None'
                # if os.path.getsize(label_file) == 0:
                #     if np.random.random() < self.negative_keep_rate:
                #         data_info['img'].append(img_file)
                #         data_info['label'].append(label_file)
                #     continue
                if pattern == 'test':
                    label_file = 'None'
                data_info['img'].append(img_file)
                data_info['label'].append(label_file)

            return data_info

        data_information = {'img': [], 'label': []}
        for idx, data_dir in enumerate(self.data_path_list):
            if len(self.data_path_list) == len(self.img_format_list):
                img_format = self.img_format_list[idx]
            else:
                img_format = self.img_format_list[0]
            assert os.path.exists(data_dir), 'No dir: ' + data_dir
            img_path_list = glob.glob(data_dir+'/*{0}'.format(img_format))
            # img folder
            if len(img_path_list) == 0:
                img_path_list = glob.glob(data_dir+'/*')
                for img_path in img_path_list:
                    if os.path.isdir(img_path):
                        data_info = get_data_infos(img_path, img_format)
                        data_information['img'].extend(data_info['img'])
                        data_information['label'].extend(data_info['label'])

            else:
                data_info = get_data_infos(data_dir, img_format)
                data_information['img'].extend(data_info['img'])
                data_information['label'].extend(data_info['label'])

        data_annotation = pd.DataFrame(data_information)
        writer_name = self.save_path_csv + '/' + self.csv_name
        data_annotation.to_csv(writer_name, index_label=False)
        print(os.path.basename(writer_name) + ' file saves successfully!')

    def generate_val_data_from_train_data(self, frac=0.1):
        if os.path.exists(self.save_path_csv + '/' + self.csv_name):
            data = pd.read_csv(self.save_path_csv + '/' + self.csv_name)
        else:
            raise Exception('no train data')

        val_data = data.sample(frac=frac, replace=False)
        train_data = data.drop(val_data.index)
        val_data = val_data.reset_index(drop=True)
        train_data = train_data.reset_index(drop=True)
        writer_name = self.save_path_csv + '/' + self.csv_name
        train_data.to_csv(writer_name, index_label=False)
        writer_name = self.save_path_csv + '/' + self.csv_name.replace('train', 'val')
        val_data.to_csv(writer_name, index_label=False)

    def _get_file(self, in_path_list):
        file_list = []
        for file in in_path_list:
            if os.path.isdir(file):
                files = glob.glob(file + '/*')
                file_list.extend(self._get_file(files))
            else:
                file_list += [file]
        return file_list

    def get_csv_file(self, phase):
        phases = ['seg', 'flow', 'od']
        assert phase in phases, f'{phase} should in {phases}!'

        file_list = self._get_file(self.data_path_list)
        file_list = [x for x in file_list if x.split('.')[-1] in self.img_format_list]
        assert len(file_list), 'No data in  data_path_list!'
        random.shuffle(file_list)
        data_information = {}
        if phase == 'seg':
            data_information['img'] = file_list
            data_information['label'] = [x.replace('img', 'label') for x in file_list]
        elif phase == 'flow':
            data_information['img1'] = file_list[:-1]
            data_information['img2'] = file_list[1:]
        elif phase == 'od':
            data_information['img'] = file_list
            data_information['label'] = [x.replace('tiff', 'txt').replace('jpg', 'txt').replace('png', 'txt') for x in file_list]

        data_annotation = pd.DataFrame(data_information)
        writer_name = self.save_path_csv + '/' + self.csv_name
        data_annotation.to_csv(writer_name, index_label=False)
        print(os.path.basename(writer_name) + ' file saves successfully!')


if __name__ == '__main__':
    data_path_list = [
        r'L:\Code\ProjectOnGithub\STT\Data\test_samples\img'
                      ]
    csv_name = 'test_data.csv'
    img_format_list = ['png']

    getTrainTestCSV = GetTrainTestCSV(dataset_path_list=data_path_list, csv_name=csv_name, img_format_list=img_format_list)
    getTrainTestCSV.get_csv_file(phase='seg')


