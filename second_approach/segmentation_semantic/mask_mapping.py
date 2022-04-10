##Librairies
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from utils.augmentation import get_training_augmentation, get_preprocessing, get_validation_augmentation
from utils.utils import get_class_dict, load_config
from utils.mask_dataset import MaskDataset
from utils.prediction import predict, test

def train(encoder, encoder_weights, epochs):
    config = load_config()
    model_path = config['MODEL_PATH']

    ## Model Definition
    ENCODER = encoder
    ENCODER_WEIGHTS = encoder_weights
    EPOCHS = epochs

    class_dict = get_class_dict()
    CLASSES = class_dict['name']
    ACTIVATION = 'softmax2d' # 'softmax2d' for multiclass segmentation

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    # load best saved model checkpoint from previous commit (if present)
    if os.path.exists(model_path):
        model = torch.load(model_path, map_location=DEVICE)
        print('Loaded DeepLabV3+')
        print(model)
    else:
        # create segmentation model with pretrained encoder
        model = smp.DeepLabV3Plus(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Get train and val dataset instances
    train_dataset = MaskDataset(
        fold='train', 
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=class_dict['color'],
    )

    valid_dataset = MaskDataset(
        fold='valid', 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=class_dict['color'],
    )



    # Get train and val data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True) #add multiprocessing : 'num_workers=4'
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False) #add multiprocessing : 'num_workers=2'

    # define loss function
    loss = smp.utils.losses.DiceLoss()

    # define metrics
    metrics = [
        smp.utils.metrics.IoU(),
    ]

    # define optimizer
    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=0.0001),
    ])

    # define learning rate scheduler (not used in this NB)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=2, eta_min=5e-5,
    )

    # load best saved model checkpoint from previous commit (if present)
    # if os.path.exists(f'{model_to_be_trained}'):
    #     model = torch.load(f'{model_to_be_trained}', map_location=DEVICE)

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )

    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []

    for i in range(0, EPOCHS):
        # Perform training & validation
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        # Save model if a better val IoU score is obtained
        if best_iou_score < valid_logs['iou_score']:
            best_iou_score = valid_logs['iou_score']
            torch.save(model, model_path)
            print('Model saved!')


    return model, train_logs_list, valid_logs_list

def main():
    ## Model Definition
    config = load_config()
    encoder = config['ENCODER']
    encoder_weights = config['ENCODER_WEIGHTS']
    epochs = config['EPOCHS']

    model, train_logs_list, valid_logs_list = train(encoder, encoder_weights, epochs)
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    predict(model=model, fold='test', preprocessing_fn=preprocessing_fn)
    test(model, preprocessing_fn, train_logs_list, valid_logs_list)

if __name__ == "__main__":
    main()

