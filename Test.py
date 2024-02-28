import os
# Change the numbers when you want to test with specific gpus
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
import torch
from STTNet import STTNet
import torch.nn.functional as F
from Utils.Datasets import get_data_loader
from Utils.Utils import make_numpy_img, inv_normalize_img, encode_onehot_to_mask, get_metrics, Logger
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

if __name__ == '__main__':
    model_infos = {
        # vgg16_bn, resnet50, resnet18
        'backbone': 'resnet50',
        'pretrained': True,
        'out_keys': ['block4'],
        'in_channel': 3,
        'n_classes': 2,
        'top_k_s': 64,
        'top_k_c': 16,
        'encoder_pos': True,
        'decoder_pos': True,
        'model_pattern': ['X', 'A', 'S', 'C'],

        'log_path': 'Results',
        'NUM_WORKERS': 0,
        # if you need the validation process.
        'IS_VAL': True,
        'VAL_BATCH_SIZE': 4,
        'VAL_DATASET': 'generate_dep_info/val_data.csv',
        # if you need the test process.
        'IS_TEST': True,
        'TEST_DATASET': 'generate_dep_info/test_data.csv',
        'IMG_SIZE': [512, 512],
        'PHASE': 'seg',

        # INRIA Dataset
        'PRIOR_MEAN': [0.40672500537632994, 0.42829032416229895, 0.39331840468605667],
        'PRIOR_STD': [0.029498464618176873, 0.027740088491668233, 0.028246722411879095],
        # # # WHU Dataset
        # 'PRIOR_MEAN': [0.4352682576428411, 0.44523221318154493, 0.41307610541534784],
        # 'PRIOR_STD': [0.026973196780331585, 0.026424642808887323, 0.02791246590291434],

        # load state dict path
        'load_checkpoint_path': 'Checkpoints/ckpt_latest.pt',
    }
    if model_infos['IS_VAL']:
        os.makedirs(model_infos['log_path']+'/val', exist_ok=True)
    if model_infos['IS_TEST']:
        os.makedirs(model_infos['log_path']+'/test', exist_ok=True)
    logger = Logger(model_infos['log_path'] + '/log.log')

    data_loaders = get_data_loader(model_infos, test_mode=True)
    loss_weight = 0.1
    model = STTNet(**model_infos)

    logger.write(f'load checkpoint from {model_infos["load_checkpoint_path"]}\n')
    state_dict = torch.load(model_infos['load_checkpoint_path'], map_location='cpu')
    model_dict = state_dict['model_state_dict']
    try:
        model_dict = OrderedDict({k.replace('module.', ''): v for k, v in model_dict.items()})
        model.load_state_dict(model_dict)
    except Exception as e:
        model.load_state_dict(model_dict)
    model = model.cuda()
    device_ids = range(torch.cuda.device_count())
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        logger.write(f'Use GPUs: {device_ids}\n')
    else:
        logger.write(f'Use GPUs: 1\n')

    patterns = ['val', 'test']
    for pattern_id, is_pattern in enumerate([model_infos['IS_VAL'], model_infos['IS_TEST']]):
        if is_pattern:
            # pred: logits, tensor, nBatch * nClass * W * H
            # target: labels, tensor, nBatch * nClass * W * H
            # output, batch['label']
            collect_result = {'pred': [], 'target': []}
            pattern = patterns[pattern_id]
            model.eval()
            for batch_id, batch in enumerate(data_loaders[pattern]):
                # Get data
                img_batch = batch['img'].cuda()
                label_batch = batch['label'].cuda()
                img_names = batch['img_name']
                collect_result['target'].append(label_batch.data.cpu())

                # inference
                with torch.no_grad():
                    logits, att_branch_output = model(img_batch)

                collect_result['pred'].append(logits.data.cpu())
                # get segmentation result, when the phase is test.
                pred_label = torch.argmax(logits, 1)
                pred_label *= 255

                # output the segmentation result
                if pattern == 'test' or batch_id % 5 == 1:
                    batch_size = pred_label.size(0)
                    # k = np.clip(int(0.3 * batch_size), a_min=1, a_max=batch_size)
                    # ids = np.random.choice(range(batch_size), k, replace=False)
                    ids = range(batch_size)
                    for img_id in ids:
                        img = img_batch[img_id].detach().cpu()
                        target = label_batch[img_id].detach().cpu()
                        pred = pred_label[img_id].detach().cpu()
                        img_name = img_names[img_id]

                        img = make_numpy_img(
                            inv_normalize_img(img, model_infos['PRIOR_MEAN'], model_infos['PRIOR_STD']))
                        target = make_numpy_img(encode_onehot_to_mask(target)) * 255
                        pred = make_numpy_img(pred)

                        vis = np.concatenate([img / 255., target / 255., pred / 255.], axis=0)
                        vis = np.clip(vis, a_min=0, a_max=1)
                        file_name = os.path.join(model_infos['log_path'], pattern, f'{img_name.split(".")[0]}.png')
                        plt.imsave(file_name, vis)

            collect_result['pred'] = torch.cat(collect_result['pred'], dim=0)
            collect_result['target'] = torch.cat(collect_result['target'], dim=0)
            IoU, OA, F1_score = get_metrics('seg', **collect_result)
            logger.write(f'{pattern}: Iou:{IoU[-1]:.4f} OA:{OA[-1]:.4f} F1:{F1_score[-1]:.4f}\n')

