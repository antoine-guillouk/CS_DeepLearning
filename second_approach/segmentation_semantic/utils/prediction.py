##Librairies

import os, cv2, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encoding import reverse_one_hot, colour_code_segmentation, confidence, color_confidence
from augmentation import get_preprocessing, get_validation_augmentation, crop_image
from utils import get_class_dict, load_config
from mask_dataset import MaskDataset



def predict(model=None, fold='test', preprocessing_fn=None):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_dict = get_class_dict()
    config = load_config()
    model_path = config['MODEL_PATH']

    if preprocessing_fn is None:
        ENCODER = config['ENCODER']
        ENCODER_WEIGHTS = config['ENCODER_WEIGHTS']
        preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    if model is None:
        # load best saved model checkpoint from previous commit (if present)
        if os.path.exists(model_path):
            model = torch.load(model_path, map_location=DEVICE)
            print('Loaded DeepLabV3+')

    # test dataset (with preprocessing transformations)
    test_dataset = MaskDataset(
        fold='test', 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=class_dict['color']
    )
    # test dataset for visualization (without preprocessing transformations)
    test_dataset_vis = MaskDataset(
    fold=fold, 
        augmentation=get_validation_augmentation(),
        class_rgb_values=class_dict['color']
    )

    sample_preds_folder = config['PREDICTIONS_DIR']
    if not os.path.exists(sample_preds_folder):
        os.makedirs(sample_preds_folder)

    for idx in range(len(test_dataset)):

        image, gt_mask = test_dataset[idx]
        image_vis = crop_image(test_dataset_vis[idx][0].astype('uint8'))
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        # Predict test image
        pred_mask = model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        # Convert pred_mask from `CHW` format to `HWC` format
        pred_mask = np.transpose(pred_mask,(1,2,0))
        pred_mask_colored = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), class_dict['color']))
        confidence_map = crop_image(color_confidence(confidence(pred_mask)))
        # Convert gt_mask from `CHW` format to `HWC` format
        gt_mask = np.transpose(gt_mask,(1,2,0))
        gt_mask = crop_image(colour_code_segmentation(reverse_one_hot(gt_mask), class_dict['color']))
        cv2.imwrite(os.path.join(sample_preds_folder, f"sample_pred_{idx}.png"), np.hstack([image_vis, gt_mask, pred_mask_colored, confidence_map])[:,:,::-1])

def test(model, preprocessing_fn, train_logs_list, valid_logs_list):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_dict = get_class_dict()

    # create test dataloader (with preprocessing operation: to_tensor(...))
    test_dataset = MaskDataset(
        fold='test', 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=class_dict['color'],
    )
    test_dataloader = DataLoader(test_dataset)
    loss = smp.utils.losses.DiceLoss(ignore_channels=[0,0,0])
    metrics = [
        smp.utils.metrics.IoU(ignore_channels=[0,0,0]),
    ]

    ## Model Evaluation on Test Dataset
    test_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    config = load_config()
    sample_preds_folder = config['PREDICTIONS_DIR']

    valid_logs = test_epoch.run(test_dataloader)
    print("Evaluation on Test Data: ")
    print(f"Mean IoU Score: {valid_logs['iou_score']:.4f}")
    print(f"Mean Dice Loss: {valid_logs['dice_loss']:.4f}")

    # Plot Dice Loss & IoU Metric for Train vs. Val
    train_logs_df = pd.DataFrame(train_logs_list)
    valid_logs_df = pd.DataFrame(valid_logs_list)
    train_logs_df.T

    plt.figure(figsize=(20,8))
    plt.plot(train_logs_df.index.tolist(), train_logs_df.iou_score.tolist(), lw=3, label = 'Train')
    plt.plot(valid_logs_df.index.tolist(), valid_logs_df.iou_score.tolist(), lw=3, label = 'Valid')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('IoU Score', fontsize=20)
    plt.title('IoU Score Plot', fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig(os.path.join(sample_preds_folder, 'iou_score_plot.png'))
    plt.show()

    plt.figure(figsize=(20,8))
    plt.plot(train_logs_df.index.tolist(), train_logs_df.dice_loss.tolist(), lw=3, label = 'Train')
    plt.plot(valid_logs_df.index.tolist(), valid_logs_df.dice_loss.tolist(), lw=3, label = 'Valid')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Dice Loss', fontsize=20)
    plt.title('Dice Loss Plot', fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig(os.path.join(sample_preds_folder, 'dice_loss_plot.png'))
    plt.show()


def main():
    ## Model Definition
    config = load_config()
    encoder = config['ENCODER']
    encoder_weights = config['ENCODER_WEIGHTS']

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    predict(fold='test', preprocessing_fn=preprocessing_fn)

if __name__ == "__main__":
    main()

predict()