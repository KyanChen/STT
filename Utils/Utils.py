import yaml
import torch
import random
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from einops import repeat
import cv2
import time
import torch.nn.functional as F


__all__ = ["decode_mask_to_onehot",
           "encode_onehot_to_mask",
           'Logger',
           'get_coords_grid',
           'get_coords_grid_float',
           'draw_bboxes',
           'Infos',
           'inv_normalize_img',
           'make_numpy_img',
           'get_metrics'
           ]


class Infos(object):
    def __init__(self, phase, class_names=None):
        assert phase in ['od'], "Error in Infos"
        self.phase = phase
        self.class_names = class_names
        self.register()
        self.pattern = 'train'
        self.epoch_id = 0
        self.max_epoch = 0
        self.batch_id = 0
        self.batch_num = 0
        self.lr = 0
        self.fps_data_load = 0
        self.fps = 0
        self.val_metric = 0

        # 'running_acc': {'loss': [], 'mIoU': [], 'OA': [], 'F1_score': []},
        # 'epoch_metrics': {'loss': 1e10, 'mIoU': 0, 'OA': 0, 'F1_score': 0},
        # 'best_val_metrics': {'epoch_id': 0, 'loss': 1e10, 'mIoU': 0, 'OA': 0, 'F1_score': 0},
    def set_epoch_training_time(self, data):
        self.epoch_training_time = data

    def set_pattern(self, data):
        self.pattern = data
    def set_epoch_id(self, data):
        self.epoch_id = data
    def set_max_epoch(self, data):
        self.max_epoch = data
    def set_batch_id(self, data):
        self.batch_id = data
    def set_batch_num(self, data):
        self.batch_num = data
    def set_lr(self, data):
        self.lr = data
    def set_fps_data_load(self, data):
        self.fps_data_load = data
    def set_fps(self, data):
        self.fps = data
    def clear_cache(self):
        self.register()

    def get_val_metric(self):
        return self.val_metric

    def cal_metrics(self):
        if self.phase == 'od':
            coco_api_gt = COCO()
            coco_api_gt.dataset['images'] = []
            coco_api_gt.dataset['annotations'] = []
            ann_id = 0
            for i, targets_per_image in enumerate(self.result_all['target_all']):
                for j in range(targets_per_image.shape[0]):
                    coco_api_gt.dataset['images'].append({'id': i})
                    coco_api_gt.dataset['annotations'].append({
                        'image_id': i,
                        "category_id": int(targets_per_image[j, 0]),
                        "bbox": np.hstack([targets_per_image[j, 1:3], targets_per_image[j, 3:5] - targets_per_image[j, 1:3]]),
                        "area": np.prod(targets_per_image[j, 3:5] - targets_per_image[j, 1:3]),
                        "id": ann_id,
                        "iscrowd": 0
                    })
                    ann_id += 1
            coco_api_gt.dataset['categories'] = [{"id": i, "supercategory": c, "name": c} for i, c in
                                                 enumerate(self.class_names)]
            coco_api_gt.createIndex()

            coco_api_pred = COCO()
            coco_api_pred.dataset['images'] = []
            coco_api_pred.dataset['annotations'] = []
            ann_id = 0
            for i, preds_per_image in enumerate(self.result_all['pred_all']):
                for j in range(preds_per_image.shape[0]):
                    coco_api_pred.dataset['images'].append({'id': i})
                    coco_api_pred.dataset['annotations'].append({
                        'image_id': i,
                        "category_id": int(preds_per_image[j, 0]),
                        'score': preds_per_image[j, 1],
                        "bbox": np.hstack(
                            [preds_per_image[j, 2:4], preds_per_image[j, 4:6] - preds_per_image[j, 2:4]]),
                        "area": np.prod(preds_per_image[j, 4:6] - preds_per_image[j, 2:4]),
                        "id": ann_id,
                        "iscrowd": 0
                    })
                    ann_id += 1
            coco_api_pred.dataset['categories'] = [{"id": i, "supercategory": c, "name": c} for i, c in
                                                 enumerate(self.class_names)]
            coco_api_pred.createIndex()

            coco_eval = COCOeval(coco_api_gt, coco_api_pred, "bbox")
            coco_eval.params.imgIds = coco_api_gt.getImgIds()
            coco_eval.evaluate()
            coco_eval.accumulate()
            self.metrics = coco_eval.summarize()
            self.val_metric = self.metrics[1]

    def print_epoch_state_infos(self, logger):
        infos_str = 'Pattern: %s Epoch [%d,%d], time: %d loss: %.4f' % \
                    (self.pattern, self.epoch_id, self.max_epoch, self.epoch_training_time, np.mean(self.loss_all['loss']))
        logger.write(infos_str + '\n')
        time_start = time.time()
        self.cal_metrics()
        time_end = time.time()
        logger.write('Pattern: %s Epoch Eval_time: %d\n' % (self.pattern, (time_end - time_start)))

        if self.phase == 'od':
            titleStr = 6 * ['Average Precision'] + 6 * ['Average Recall']
            typeStr = 6 * ['(AP)'] + 6 * ['(AR)']
            iouStr = 12 * ['0.50:0.95']
            iouStr[1] = '0.50'
            iouStr[2] = '0.75'
            areaRng = 3 * ['all'] + ['small', 'medium', 'large'] + 3 * ['all'] + ['small', 'medium', 'large']
            maxDets = 6 * [100] + [1, 10, 100] + 3 * [100]
            for i in range(12):
                infos_str = '{:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}\n'
                logger.write(infos_str.format(titleStr[i], typeStr[i], iouStr[i], areaRng[i], maxDets[i], self.metrics[i]))


    def save_epoch_state_infos(self, writer):
        iter = self.epoch_id
        keys = [
            'AP_m_all_100',
            'AP_50_all_100',
            'AP_75_all_100',
            'AP_m_small_100',
            'AP_m_medium_100',
            'AP_m_large_100',
            'AR_m_all_1',
            'AR_m_all_10',
            'AR_m_all_100',
            'AR_m_small_100',
            'AR_m_medium_100',
            'AR_m_large_100',
                ]
        for i, key in enumerate(keys):
            writer.add_scalar(f'%s/epoch/%s' % (self.pattern, key), self.metrics[i], iter)

    def print_batch_state_infos(self, logger):
        infos_str = 'Pattern: %s [%d,%d][%d,%d], lr: %5f, fps_data_load: %.2f, fps: %.2f' % \
                    (self.pattern, self.epoch_id, self.max_epoch, self.batch_id,
                     self.batch_num, self.lr, self.fps_data_load, self.fps)
        # add loss
        infos_str += ', loss: %.4f' % self.loss_all['loss'][-1]
        logger.write(infos_str + '\n')

    def save_batch_state_infos(self, writer):
        iter = self.epoch_id * self.batch_num + self.batch_id
        writer.add_scalar('%s/lr' % self.pattern, self.lr, iter)
        for key, value in self.loss_all.items():
            writer.add_scalar(f'%s/%s' % (self.pattern, key), value[-1], iter)

    def save_results(self, img_batch, prior_mean, prior_std, vis_dir, *args, **kwargs):
        batch_size = img_batch.size(0)
        k = np.clip(int(0.3 * batch_size), a_min=1, a_max=batch_size)
        ids = np.random.choice(range(batch_size), k, replace=False)
        for img_id in ids:
            img = img_batch[img_id].detach().cpu()
            pred = self.result_all['pred_all'][img_id - batch_size]
            target = self.result_all['target_all'][img_id - batch_size]

            img = make_numpy_img(inv_normalize_img(img, prior_mean, prior_std))
            pred_draw = draw_bboxes(img, pred, self.class_names, (255, 0, 0))
            target_draw = draw_bboxes(img, target, self.class_names, (0, 255, 0))
            # target = make_numpy_img(encode_onehot_to_mask(target))
            # pred = make_numpy_img(pred_label[img_id])

            vis = np.concatenate([img/255., pred_draw/255., target_draw/255.], axis=0)
            vis = np.clip(vis, a_min=0, a_max=1)
            file_name = os.path.join(vis_dir, self.pattern, f'{self.epoch_id}_{self.batch_id}_{img_id}.png')
            plt.imsave(file_name, vis)

    def register(self):
        self.is_registered_result = False
        self.result_all = {}

        self.is_registered_loss = False
        self.loss_all = {}

    def register_result(self, data: dict):
        for key in data.keys():
            self.result_all[key] = []
        self.is_registered_result = True

    def append_result(self, data: dict):
        if not self.is_registered_result:
            self.register_result(data)
        for key, value in data.items():
            self.result_all[key] += value

    def register_loss(self, data: dict):
        for key in data.keys():
            self.loss_all[key] = []
        self.is_registered_loss = True

    def append_loss(self, data: dict):
        if not self.is_registered_loss:
            self.register_loss(data)
        for key, value in data.items():
            self.loss_all[key].append(value.detach().cpu().numpy())


# draw bboxes on image, bboxes with classID
def draw_bboxes(img, bboxes, color=(255, 0, 0), class_names=None, is_show_score=True):
    '''
    Args:
        img:
        bboxes: [n, 5], class_idx, l, t, r, b
                [n, 6], class_idx, score, l, t, r, b
    Returns:
    '''
    assert img is not None, "In draw_bboxes, img is None"
    if torch.is_tensor(img):
        img = img.cpu().numpy()
    img = img.astype(np.uint8).copy()

    if torch.is_tensor(bboxes):
        bboxes = bboxes.cpu().numpy()
    for bbox in bboxes:
        if class_names:
            class_name = class_names[int(bbox[0])]
        bbox_coordinate = bbox[1:]
        if len(bbox) == 6:
            score = bbox[1]
            bbox_coordinate = bbox[2:]
        bbox_coordinate = bbox_coordinate.astype(np.int)
        if is_show_score:
            cv2.rectangle(img, pt1=tuple(bbox_coordinate[0:2] - np.array([2, 15])),
                          pt2=tuple(bbox_coordinate[0:2] + np.array([15, 1])), color=(0, 0, 255), thickness=-1)
            if len(bbox) == 6:
                cv2.putText(img, text='%s:%.2f' % (class_name, score),
                            org=tuple(bbox_coordinate[0:2] - np.array([1, 7])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.2, color=(255, 255, 255), thickness=1)
            else:
                cv2.putText(img, text='%s' % class_name,
                            org=tuple(bbox_coordinate[0:2] - np.array([1, 7])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.2, color=(255, 255, 255), thickness=1)
        cv2.rectangle(img, pt1=tuple(bbox_coordinate[0:2]), pt2=tuple(bbox_coordinate[2:4]), color=color, thickness=2)
    return img


def get_coords_grid(h_end, w_end, h_start=0, w_start=0, h_steps=None, w_steps=None, is_normalize=False):
    if h_steps is None:
        h_steps = int(h_end - h_start) + 1
    if w_steps is None:
        w_steps = int(w_end - w_start) + 1

    y = torch.linspace(h_start, h_end, h_steps)
    x = torch.linspace(w_start, w_end, w_steps)
    if is_normalize:
        y = y / h_end
        x = x / w_end
    coords = torch.meshgrid(y, x)
    coords = torch.stack(coords[::-1], dim=0)
    return coords


def get_coords_grid_float(ht, wd, scale, is_normalize=False):
    y = torch.linspace(0, scale, ht + 2)
    x = torch.linspace(0, scale, wd + 2)
    if is_normalize:
        y = y/scale
        x = x/scale
    coords = torch.meshgrid(y[1:-1], x[1:-1])
    coords = torch.stack(coords[::-1], dim=0)
    return coords


def get_coords_vector_float(len, scale, is_normalize=False):
    x = torch.linspace(0, scale, len+2)
    if is_normalize:
        x = x/scale
    coords = torch.meshgrid(x[1:-1], torch.tensor([0.]))
    coords = torch.stack(coords[::-1], dim=0)
    return coords


class Logger(object):
    def __init__(self, filename="Default.log", is_terminal_show=True):
        self.is_terminal_show = is_terminal_show
        if self.is_terminal_show:
            self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        if self.is_terminal_show:
            self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        if self.is_terminal_show:
            self.terminal.flush()
        self.log.flush()


class ParamsParser:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_all_dict(dict_infos: dict) -> dict:
    return_dict = {}
    for key, value in dict_infos.items():
        if not isinstance(value, dict):
            return_dict[key] = value
        else:
            return_dict = dict(return_dict.items(), **get_all_dict(value))
    return return_dict


def make_numpy_img(tensor_data):
    if len(tensor_data.shape) == 2:
        tensor_data = tensor_data.unsqueeze(2)
        tensor_data = torch.cat((tensor_data, tensor_data, tensor_data), dim=2)
    elif tensor_data.size(0) == 1:
        tensor_data = tensor_data.permute((1, 2, 0))
        tensor_data = torch.cat((tensor_data, tensor_data, tensor_data), dim=2)
    elif tensor_data.size(0) == 3:
        tensor_data = tensor_data.permute((1, 2, 0))
    elif tensor_data.size(2) == 3:
        pass
    else:
        raise Exception('tensor_data apply to make_numpy_img error')
    vis_img = tensor_data.detach().cpu().numpy()

    return vis_img


def print_infos(logger, writer, infos: dict):
    keys = list(infos.keys())
    values = list(infos.values())
    infos_str = 'Pattern: %s [%d,%d][%d,%d], lr: %5f, fps_data_load: %.2f, fps: %.2f' % tuple(values[:8])
    if len(values) > 8:
        extra_infos = [f', {x}: {y:.4f}' for x, y in zip(keys[8:], values[8:])]
        infos_str = infos_str + ''.join(extra_infos)

    logger.write(infos_str + '\n')

    writer.add_scalar('%s/lr' % infos['pattern'], infos['lr'],
                      infos['epoch_id'] * infos['batch_num'] + infos['batch_id'])
    for key, value in zip(keys[8:], values[8:]):
        writer.add_scalar(f'%s/%s' % (infos['pattern'], key), value,
                          infos['epoch_id'] * infos['batch_num'] + infos['batch_id'])


def invert_affine(origin_imgs, preds, pattern='train'):
    if pattern == 'val':
        for i in range(len(preds)):
            if len(preds[i]['rois']) == 0:
                continue
            else:
                old_h, old_w, _ = origin_imgs[i].shape
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (512 / old_w)
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (512 / old_h)
    return preds


def save_output_infos(input, output, vis_dir, pattern, epoch_id, batch_id):
    flows, pf1s, pf2s = output
    k = np.clip(int(0.2 * len(flows[0])), a_min=2, a_max=len(flows[0]))
    ids = np.random.choice(range(len(flows[0])), k, replace=False)
    for img_id in ids:
        img1, img2 = input['ori_img1'][img_id:img_id+1].to(flows[0].device), input['ori_img2'][img_id:img_id+1].to(flows[0].device)
        # call the network with image pair batches and actions
        flow = flows[0][img_id:img_id+1]
        warps = flow_to_warp(flow)

        warped_img2 = resample(img2, warps)

        ori_img1 = make_numpy_img(img1[0]) / 255.
        ori_img2 = make_numpy_img(img2[0]) / 255.
        warped_img2 = make_numpy_img(warped_img2[0]) / 255.
        flow_amplitude = torch.sqrt(flow[0, 0:1, ...] ** 2 + flow[0, 1:2, ...] ** 2)
        flow_amplitude = make_numpy_img(flow_amplitude)
        flow_amplitude = (flow_amplitude - np.min(flow_amplitude)) / (np.max(flow_amplitude) - np.min(flow_amplitude) + 1e-10)
        u = make_numpy_img(flow[0, 0:1, ...])
        v = make_numpy_img(flow[0, 1:2, ...])

        vis = np.concatenate([ori_img1, ori_img2, warped_img2, flow_amplitude], axis=0)
        vis = np.clip(vis, a_min=0, a_max=1)
        file_name = os.path.join(vis_dir, pattern, str(epoch_id) + '_' + str(batch_id) + '.jpg')
        plt.imsave(file_name, vis)


def inv_normalize_img(img, prior_mean=[0, 0, 0], prior_std=[1, 1, 1]):
    prior_mean = torch.tensor(prior_mean, dtype=torch.float).to(img.device).view(img.size(0), 1, 1)
    prior_std = torch.tensor(prior_std, dtype=torch.float).to(img.device).view(img.size(0), 1, 1)
    img = img * prior_std + prior_mean
    img = img * 255.
    img = torch.clamp(img, min=0, max=255)
    return img


def save_seg_output_infos(input, output, vis_dir, pattern, epoch_id, batch_id, prior_mean, prior_std):
    pred_label = torch.argmax(output, 1)
    k = np.clip(int(0.2 * len(pred_label)), a_min=1, a_max=len(pred_label[0]))
    ids = np.random.choice(range(len(pred_label)), k, replace=False)
    for img_id in ids:
        img = input['img'][img_id].to(pred_label.device)
        target = input['label'][img_id].to(pred_label.device)

        img = make_numpy_img(inv_normalize_img(img, prior_mean, prior_std)) / 255.
        target = make_numpy_img(encode_onehot_to_mask(target))
        pred = make_numpy_img(pred_label[img_id])

        vis = np.concatenate([img, pred, target], axis=0)
        vis = np.clip(vis, a_min=0, a_max=1)
        file_name = os.path.join(vis_dir, pattern, str(epoch_id) + '_' + str(batch_id) + '.jpg')
        plt.imsave(file_name, vis)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def cpt_pxl_cls_acc(pred_idx, target):
    pred_idx = torch.reshape(pred_idx, [-1])
    target = torch.reshape(target, [-1])
    return torch.mean((pred_idx.int() == target.int()).float())


def cpt_batch_psnr(img, img_gt, PIXEL_MAX):
    mse = torch.mean((img - img_gt) ** 2, dim=[1, 2, 3])
    psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
    return torch.mean(psnr)


def cpt_psnr(img, img_gt, PIXEL_MAX):
    mse = np.mean((img - img_gt) ** 2)
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr


def cpt_rgb_ssim(img, img_gt):
    img = clip_01(img)
    img_gt = clip_01(img_gt)
    SSIM = 0
    for i in range(3):
        tmp = img[:, :, i]
        tmp_gt = img_gt[:, :, i]
        ssim = sk_cpt_ssim(tmp, tmp_gt)
        SSIM = SSIM + ssim
    return SSIM / 3.0


def cpt_ssim(img, img_gt):
    img = clip_01(img)
    img_gt = clip_01(img_gt)
    return sk_cpt_ssim(img, img_gt)


def decode_mask_to_onehot(mask, n_class):
    '''
    mask : BxWxH or WxH
    n_class : n
    return : BxnxWxH or nxWxH
    '''
    assert len(mask.shape) in [2, 3], "decode_mask_to_onehot error!"
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(0)
    onehot = torch.zeros((mask.size(0), n_class, mask.size(1), mask.size(2))).to(mask.device)
    for i in range(n_class):
        onehot[:, i, ...] = mask == i
    if len(mask.shape) == 2:
        onehot = onehot.squeeze(0)
    return onehot


def encode_onehot_to_mask(onehot):
    '''
    onehot: tensor, BxnxWxH or nxWxH
    output: tensor, BxWxH or WxH
    '''
    assert len(onehot.shape) in [3, 4], "encode_onehot_to_mask error!"
    mask = torch.argmax(onehot, dim=len(onehot.shape)-3)
    return mask


def decode(pred, target=None, *args, **kwargs):
    """

    Args:
        phase: 'od'
        pred: big_cls_1(0), big_reg_1, small_cls_1(2), small_reg_1, big_cls_2(4), big_reg_2, small_cls_2(6), small_reg_2
        target: [[n,5], [n,5]] list of tensor

    Returns:

    """
    phase = kwargs['phase']
    img_size = kwargs['img_size']
    if phase == 'od':
        prior_box_wh = kwargs['prior_box_wh']
        conf_thres = kwargs['conf_thres']
        iou_thres = kwargs['iou_thres']
        conf_type = kwargs['conf_type']
        pred_conf_32_2 = F.softmax(pred[4], dim=1)[:, 1, ...]  # B H W
        pred_conf_64_2 = F.softmax(pred[6], dim=1)[:, 1, ...]  # B H W
        obj_mask_32_2 = pred_conf_32_2 > conf_thres  # B H W
        obj_mask_64_2 = pred_conf_64_2 > conf_thres  # B H W

        pre_loc_32_2 = pred[1] + pred[5]  # B 4 H W
        pre_loc_32_2[:, 0::2, ...] *= prior_box_wh[0]
        pre_loc_32_2[:, 1::2, ...] *= prior_box_wh[1]
        x_y_grid = get_coords_grid(31, 31, 0, 0)
        x_y_grid *= 8
        x_y_grid = torch.cat([x_y_grid, x_y_grid], dim=0)
        pre_loc_32_2 += x_y_grid.to(pre_loc_32_2.device)

        pre_loc_64_2 = pred[3] + pred[7]  # B 4 H W
        pre_loc_64_2[:, 0::2, ...] *= prior_box_wh[0]
        pre_loc_64_2[:, 1::2, ...] *= prior_box_wh[1]
        x_y_grid_2 = get_coords_grid(63, 63, 0, 0)
        x_y_grid_2 *= 4
        x_y_grid_2 = torch.cat([x_y_grid_2, x_y_grid_2], dim=0)
        pre_loc_64_2 += x_y_grid_2.to(pre_loc_32_2.device)

        pred_all = []
        for i in range(pre_loc_32_2.size(0)):
            score_32 = pred_conf_32_2[i][obj_mask_32_2[i]]  # N
            score_64 = pred_conf_64_2[i][obj_mask_64_2[i]]  # M

            loc_32 = pre_loc_32_2[i].permute((1, 2, 0))[obj_mask_32_2[i]]  # Nx4
            loc_64 = pre_loc_64_2[i].permute((1, 2, 0))[obj_mask_64_2[i]]  # Mx4

            score_list = torch.cat((score_32, score_64), dim=0).detach().cpu().numpy()
            boxes_list = torch.cat((loc_32, loc_64), dim=0).detach().cpu().numpy()
            boxes_list[:, 0::2] /= img_size[0]
            boxes_list[:, 1::2] /= img_size[1]
            label_list = np.ones_like(score_list)
            # 目标预设150
            boxes_list = boxes_list[:150, :]
            score_list = score_list[:150]
            label_list = label_list[:150]
            boxes, scores, labels = weighted_boxes_fusion([boxes_list], [score_list], [label_list], weights=None,
                                                          iou_thr=iou_thres, conf_type=conf_type)
            boxes[:, 0::2] *= img_size[0]
            boxes[:, 1::2] *= img_size[1]
            pred_boxes = np.concatenate((labels.reshape(-1, 1), scores.reshape(-1, 1), boxes), axis=1)
            pred_all.append(pred_boxes)
        if target is not None:
            target_all = [x.cpu().numpy() for x in target]
        else:
            target_all = None
        return {"pred_all": pred_all, "target_all": target_all}



def get_metrics(phase, pred, target):

    '''
    pred: logits, tensor, nBatch*nClass*W*H
    target: labels, tensor, nBatch*nClass*W*H
    '''
    if phase == 'seg':
        pred = torch.argmax(pred.detach(), dim=1)
        pred = decode_mask_to_onehot(pred, target.size(1))
        # positive samples in ground truth
        gt_pos_sum = torch.sum(target == 1, dim=(0, 2, 3))
        # positive prediction in predict mask
        pred_pos_sum = torch.sum(pred == 1, dim=(0, 2, 3))
        # cal true positive sample
        true_pos_sum = torch.sum((target == 1) * (pred == 1), dim=(0, 2, 3))
        # Precision
        precision = true_pos_sum / (pred_pos_sum + 1e-15)
        # Recall
        recall = true_pos_sum / (gt_pos_sum + 1e-15)
        # IoU
        IoU = true_pos_sum / (pred_pos_sum + gt_pos_sum - true_pos_sum + 1e-15)
        # OA
        OA = 1 - (pred_pos_sum + gt_pos_sum - 2 * true_pos_sum) / torch.sum(target >= 0, dim=(0, 2, 3))
        # F1-score
        F1_score = 2 * precision * recall / (precision + recall + 1e-15)
        return IoU, OA, F1_score

def show_feature(f_tensor, img, label, is_normalize=False, is_channel=False):
    # 1 H W
    if not is_normalize:
        # normalize the features / feature maps
        f_tensor = torch.sigmoid(f_tensor)
    f_tensor = f_tensor.detach().cpu().numpy()

    # f_tensor = np.resize(f_tensor, [1, 16])

    # for better visualization, you can normalize the feature heatmap
    f_tensor = 0.8 * (f_tensor - np.min(f_tensor)) / (np.max(f_tensor) - np.min(f_tensor)) + 0.1
    # f_tensor = (f_tensor - np.min(f_tensor)) / (np.max(f_tensor) - np.min(f_tensor))

    # show the img and its segmentation label
    if img is not None and label is not None:
        # img = img.astype(np.uint8)
        # img = cv2.resize(img, f_tensor.shape)

        label = label.astype(np.uint8)
        # label = cv2.resize(label, f_tensor.shape, interpolation=cv2.INTER_NEAREST)
        plt.figure()
        plt.axis('off')
        plt.imshow(img)
        plt.figure()
        plt.axis('off')
        plt.imshow(label, cmap='gray')

    # if show the channel-wise activation feature heatmaps
    if is_channel:
        from matplotlib.pyplot import MultipleLocator
        import seaborn as sns
        # plt.figure(figsize=(64, 1))
        plt.figure()
        f_tensor = f_tensor.reshape(-1)
        x = np.array(range(0, len(f_tensor)))
        # sns.barplot(x=2, y=4, color="salmon")
        palette = []
        t = np.array(sns.color_palette("YlGnBu"))
        # t = np.array(sns.diverging_palette(250, 30, l=65, center="dark", n=200))
        # t = np.array(sns.cubehelix_palette(200, start=1, rot=10, gamma=0.4,  hue=1, dark=0.3, light=0.7))
        for i in range(len(f_tensor)):
            index = int(np.floor(f_tensor[i] / 0.2))
            color = (f_tensor[i] - index*0.2) / 0.2 * (t[index+1]-t[index]) + t[index]
            # color = t[int(f_tensor[i]*200)]
            palette.append(color)

        sns.barplot(x=x, y=f_tensor, palette=palette)
        plt.xlabel("Channel", fontsize=22)
        plt.ylabel("Prob", fontsize=22)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xticks(x, x, rotation=0)
        x_major_locator = MultipleLocator(4)
        ax = plt.gca()
        # ax.spines['top'].set_visible(False)  # 去掉上边框
        # ax.spines['right'].set_visible(False)  # 去掉右边框
        ax.xaxis.set_major_locator(x_major_locator)

        plt.show()
    else:
        plt.figure()
    plt.axis('off')
    sns.heatmap(f_tensor, vmin=0, vmax=1, cmap="jet", center=0.5)
    # plt.imshow(heatmap, cmap='YlGnBu', vmin=0, vmax=1)

