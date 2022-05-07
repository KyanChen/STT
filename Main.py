import torch
from STTNet import STTNet
import torch.nn.functional as F
from Utils.Datasets import get_data_loader
from Utils.Utils import make_numpy_img, inv_normalize_img, encode_onehot_to_mask, get_metrics
import os
import matplotlib.pyplot as plt
import numpy as np


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

        'BATCH_SIZE': 8,
        'IS_SHUFFLE': True,
        'NUM_WORKERS': 0,
        'DATASET': 'Tools/generate_dep_info/train_data.csv',
        'model_path': 'Checkpoints',
        'log_path': 'Results',
        # if you need the validation process.
        'IS_VAL': True,
        'VAL_BATCH_SIZE': 4,
        'VAL_DATASET': 'Tools/generate_dep_info/val_data.csv',
        # if you need the test process.
        'IS_TEST': True,
        'TEST_DATASET': 'Tools/generate_dep_info/test_data.csv',
        'IMG_SIZE': [512, 512],
        'PHASE': 'seg',
        'PRIOR_MEAN': [0.46278404739026296, 0.469763416147487, 0.44496931596235817],
        'PRIOR_STD': [0.03600466440229604, 0.036798446555721516, 0.038701379834091894],
        # if you want to load state dict
        'load_checkpoint_path': 'D:/Code/ProjectOnGithub/STT/Checkpoints/ckpt_181.pt'
    }
    os.makedirs(model_infos['model_path'], exist_ok=True)
    if model_infos['IS_VAL']:
        os.makedirs(model_infos['log_path']+'/val', exist_ok=True)
    if model_infos['IS_TEST']:
        os.makedirs(model_infos['log_path']+'/test', exist_ok=True)

    data_loaders = get_data_loader(model_infos)
    loss_weight = 0.1
    model = STTNet(**model_infos)

    epoch_start = 0
    if os.path.exists(model_infos['load_checkpoint_path']):
        print(f'load checkpoint from {model_infos["load_checkpoint_path"]}')
        state_dict = torch.load(model_infos['load_checkpoint_path'], map_location='cpu')
        epoch_start = state_dict['epoch_id']
        model_dict = state_dict['model_state_dict']
        model.load_state_dict(model_dict)

    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch_id in range(epoch_start, 200):
        pattern = 'train'
        model.train()  # Set model to training mode
        for batch_id, batch in enumerate(data_loaders[pattern]):
            # Get data
            img_batch = batch['img'].cuda()
            label_batch = batch['label'].cuda()

            # inference
            optimizer.zero_grad()
            logits, att_branch_output = model(img_batch)

            # compute loss
            label_downs = F.interpolate(label_batch, att_branch_output.size()[2:], mode='nearest')
            loss_branch = F.binary_cross_entropy_with_logits(att_branch_output, label_downs)
            loss_master = F.binary_cross_entropy_with_logits(logits, label_batch)
            loss = loss_master + loss_weight * loss_branch
            # loss backward
            loss.backward()
            optimizer.step()

            if batch_id % 3 == 1:
                print(f'{pattern}: {epoch_id}/200 {batch_id}/{len(data_loaders[pattern])} loss: {loss.item():.4f}')

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
                            file_name = os.path.join(model_infos['log_path'], pattern, f'Epoch_{epoch_id}_{img_name.split(".")[0]}.png')
                            plt.imsave(file_name, vis)

                collect_result['pred'] = torch.cat(collect_result['pred'], dim=0)
                collect_result['target'] = torch.cat(collect_result['target'], dim=0)
                IoU, OA, F1_score = get_metrics('seg', **collect_result)
                print(f'{pattern}: {epoch_id}/200 Iou:{IoU[-1]:.4f} OA:{OA[-1]:.4f} F1:{F1_score[-1]:.4f}')
        if epoch_id % 20 == 1:
            torch.save({
                'epoch_id': epoch_id,
                'model_state_dict': model.state_dict()
            }, os.path.join(model_infos['model_path'], f'ckpt_{epoch_id}.pt'))
        torch.save({
            'epoch_id': epoch_id,
            'model_state_dict': model.state_dict()
        }, os.path.join(model_infos['model_path'], f'ckpt_latest.pt'))

