import os
import glob
from skimage import io
import tqdm
img_piece_size = (512, 512)


def get_pieces(img_path, label_path, img_format):
    pieces_folder = os.path.abspath(img_path + '/..')
    if not os.path.exists(pieces_folder + '/img_pieces'):
        os.makedirs(pieces_folder + '/img_pieces')
    if not os.path.exists(pieces_folder + '/label_pieces'):
        os.makedirs(pieces_folder + '/label_pieces')

    img_path_list = glob.glob(img_path+'/austin31.%s' % img_format)
    for idx in tqdm.tqdm(range(len(img_path_list))):
        img = io.imread(img_path_list[idx])
        label = io.imread(label_path + '/' + os.path.basename(img_path_list[idx]).replace(img_format, img_format))
        h, w, c = img.shape
        h_list = list(range(0, h-img_piece_size[1], int(0.9 * img_piece_size[1])))
        h_list = h_list + [h - img_piece_size[1]]
        # h_list[-1] = h - img_piece_size[1]
        w_list = list(range(0, w-img_piece_size[0], int(0.9 * img_piece_size[0])))
        # w_list[-1] = w - img_piece_size[0]
        w_list = w_list + [w - img_piece_size[0]]
        for h_step in h_list:
            for w_step in w_list:
                img_piece = img[h_step:h_step+img_piece_size[1], w_step:w_step+img_piece_size[0]]
                label_piece = label[h_step:h_step + img_piece_size[1], w_step:w_step + img_piece_size[0]]
                assert label_piece.shape[0] == img_piece_size[1] and label_piece.shape[1] == img_piece_size[0], 'shape error'
                io.imsave(pieces_folder + '/img_pieces%s_%d_%d.png' %
                          (img_path_list[idx].replace(img_path, '').replace('.' + img_format, ''), w_step, h_step), img_piece, check_contrast=False)
                io.imsave(pieces_folder + '/label_pieces%s_%d_%d.png' %
                          (img_path_list[idx].replace(img_path, '').replace('.' + img_format, ''), w_step, h_step), label_piece, check_contrast=False)


if __name__ == "__main__":
    parent_path = r'J:\20200923-建筑提取数据集\InriaAerialImageDataset\train'
    for i in ['train', 'val', 'test']:
        img_path = parent_path + '/' + i + '/img'
        label_path = parent_path + '/' + i + '/gt'
        img_format = 'tif'
        get_pieces(img_path, label_path, img_format)

