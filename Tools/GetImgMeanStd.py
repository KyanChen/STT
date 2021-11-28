import os
import pandas as pd
from skimage import io
import numpy as np
import json
import tqdm


# R, G, B
class GetImgMeanStd:
    def __init__(self, data_file):
        assert os.path.exists(data_file), 'train.csv dose not exist!'
        self.data_info = pd.read_csv(data_file, index_col=0)
        self.save_path_mean_std_info = 'generate_dep_info'
        self.mean = None
        self.std = None

    def get_img_mean_std(self):
        means = []
        stds = []
        bar = tqdm.tqdm(total=len(self.data_info))
        for row in self.data_info.iterrows():
            bar.update(1)
            img_name = row[1]['img']
            # print(img_name)
            img = io.imread(img_name)
            img = img / 255.
            assert img is not None, img_name + 'is not valid'
            # height*width*channels, axis=0 is the first dim
            mean = np.mean(np.mean(img, axis=0), axis=0)
            means.append(mean)
            std = np.std(np.std(img, axis=0), axis=0)
            stds.append(std)
        bar.close()
        self.mean = np.mean(np.array(means), axis=0).tolist()
        self.std = np.mean(np.array(stds), axis=0).tolist()
        return {'mean': self.mean, 'std': self.std}

    def write_mean_std_information(self):
        info = self.get_img_mean_std()
        writer = os.path.join(self.save_path_mean_std_info, 'mean_std_info_test.json')
        with open(writer, 'w') as f_writer:
            json.dump(info, f_writer)
        print('\'PRIOR_MEAN\': %s\n\'PRIOR_STD\': %s\n' % (info['mean'], info['std']))


if __name__ == '__main__':
    data_file = r'generate_dep_info/train_data.csv'
    getImgMeanStd = GetImgMeanStd(data_file)
    getImgMeanStd.write_mean_std_information()


