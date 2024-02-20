import argparse
import sys
import numpy as np
from dataset_loader import DatasetFolder
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from network import UNet
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import datetime
import time
import matplotlib.pyplot as plt
import logging


def calculate_iou(pred, gt):
    intersection = torch.logical_and(pred, gt)
    union = torch.logical_or(pred, gt)
    iou = torch.sum(intersection, [1, 2]) / torch.sum(union, [1, 2])
    return iou.mean()


def calculate_f1(pred, gt):
    intersection = torch.logical_and(pred, gt)
    f1 = 2*torch.sum(intersection, [1,2]) / (torch.sum(pred, [1,2]) + torch.sum(gt, [1,2])) 
    return f1.mean()


def calculate_precision(pred, gt):
    TP = torch.logical_and(pred == 1, gt == 1)
    FP = torch.logical_and(pred == 1, gt == 0)
    precision = torch.sum(TP, [1,2]) / (torch.sum(TP, [1,2]) + torch.sum(FP, [1,2]))
    return precision.mean()


def calculate_recall(pred, gt):
    TP = torch.logical_and(pred == 1, gt == 1)
    FN = torch.logical_and(pred == 0, gt == 1)
    recall = torch.sum(TP, [1,2]) / (torch.sum(TP, [1,2]) + torch.sum(FN, [1,2]))    
    return recall.mean()


# Base training function
def train(trainloader, model, device, optimizer, loss_function):

    model.train()
    avg_loss = []

    for images, masks, img_paths, mask_paths in tqdm(trainloader):

        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        predictions = model(images)
        
        loss = loss_function(predictions, masks)
        avg_loss.append(loss.cpu().detach().numpy())

        loss.backward()
        optimizer.step()

    avg_loss = np.mean(avg_loss)
    return avg_loss


# Base validation/testing function
def test(testloader, model, device, loss_function, save_results=False):
    
    model.eval()
    iou = []
    precision = []
    recall = []
    f1 = []
    avg_val_loss = []

    with torch.no_grad():
        for images, masks, img_paths, mask_paths in tqdm(testloader):
            images, masks = images.to(device), masks.to(device)

            predictions = model(images)
            out = torch.sigmoid(predictions)
            
            predicted_masks_bin = torch.sigmoid(out) > 0.5

            # Calculate performance
            avg_val_loss.append(loss_function(predictions, masks).cpu().detach().numpy())
            iou.append(calculate_iou(predicted_masks_bin, masks).cpu().numpy())
            precision.append(calculate_precision(predicted_masks_bin, masks).cpu().numpy())
            recall.append(calculate_recall(predicted_masks_bin, masks).cpu().numpy())
            f1.append(calculate_f1(predicted_masks_bin, masks).cpu().numpy())

            if save_results:
                if not os.path.exists('./output/test_output'):
                    os.mkdir('./output/test_output')
                for i in range(predicted_masks_bin.shape[0]):
                    filename = os.path.basename(mask_paths[i])
                    img = torch.repeat_interleave(predicted_masks_bin[i, :, :, :], 3, dim=2).permute([2, 0, 1]) # Convert to 3 channels?
                    save_image(img.float(), os.path.join('./output/test_output', filename))

    # Calculate total performance of the model
    iou = np.mean(iou)
    precision = np.mean(precision)
    recall = np.mean(recall)
    f1 = np.mean(f1)
    avg_val_loss = np.mean(avg_val_loss)

    return iou, precision, recall, f1, avg_val_loss


# Log any unhandled exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    logging.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))


# MAIN
if __name__ == "__main__":

    if not os.path.exists('./output'):
        os.mkdir('./output')
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler('./output/training.log', mode='a'),
                              logging.StreamHandler()])
    
    # Log any unhandled exceptions
    sys.excepthook = handle_exception

    # arguments that can be defined upon execution of the script
    options = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    options.add_argument('--train', action='store_true', default=False, help='Train the model.')
    options.add_argument('--test', action='store_true', default=False, help='Test the model.')    
    options.add_argument('--traincsv', default='dataset/train.csv', help='directory of the train CSV')
    options.add_argument('--testcsv', default='dataset/test.csv', help='directory of the test CSV')
    options.add_argument('--valcsv', default='dataset/val.csv', help='directory of the validation CSV')
    options.add_argument('--batchsize', type=int, default=1, help='batch size')
    options.add_argument('--epochs', type=int, default=40, help='number of training epochs')
    options.add_argument('--imagesize', type=int, default=(528, 960), help='size of the image (height, width)') # Needs to be divisible by 16
    # options.add_argument('--imagesize', type=int, default=(624, 1104), help='size of the image (height, width)') # Needs to be divisible by 16
    
    # TODO: REMOVE AFTER TESTING
    # options.add_argument('--traincsv', default='dataset/val.csv', help='directory of the train CSV')
    # options.add_argument('--testcsv', default='dataset/val.csv', help='directory of the test CSV')
    # options.add_argument('--valcsv', default='dataset/val.csv', help='directory of the validation CSV')
    # options.add_argument('--imagesize', type=int, default=(256, 256), help='size of the image (height, width)') # Needs to be divisible by 16

    opt = options.parse_args()

    PRE__MEAN = [0.5, 0.5, 0.5]
    PRE__STD = [0.5, 0.5, 0.5]
    
    # Define transforms for dataset augmentation
    image_and_mask_transform_train=A.Compose([A.Resize(opt.imagesize[0], opt.imagesize[1]),
                                            A.HorizontalFlip(p=0.5),
                                            A.SafeRotate (limit=5, border_mode=4, always_apply=False, p=0.5),   # TODO: Experiment with the limit
                                            ToTensorV2()])
    
    image_only_transform_train=A.Compose([
                                        # TODO: Experiment with enabling this
                                        # A.GaussNoise(var_limit=(1.0, 10.0), mean=0, per_channel=True, always_apply=False, p=0.5), 
                                        # A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, always_apply=False, p=0.5),
                                        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
                                        A.Normalize(PRE__MEAN, PRE__STD),
                                        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                        A.Resize(621, 1104), # Convert the image to original mask dimensions to avoid errors
                                        ])
    
    image_and_mask_transform_test=A.Compose([A.Resize(opt.imagesize[0], opt.imagesize[1]),
                                            A.HorizontalFlip(p=0.5),
                                            ToTensorV2()])
    
    image_only_transform_test=A.Compose([A.Normalize(PRE__MEAN, PRE__STD), 
                                         A.Resize(621, 1104) # Convert the image to original mask dimensions to avoid errors
                                         ]) 
    
    # Define dataloaders
    train_data = DatasetFolder(csv=opt.traincsv, image_only_transform=image_only_transform_train, transform=image_and_mask_transform_train)
    val_data = DatasetFolder(csv=opt.valcsv, image_only_transform=image_only_transform_test, transform=image_and_mask_transform_test)
    test_data = DatasetFolder(csv=opt.testcsv, image_only_transform=image_only_transform_test, transform=image_and_mask_transform_test)

    trainloader = DataLoader(train_data, opt.batchsize, shuffle=True)
    valloader = DataLoader(val_data, opt.batchsize, shuffle=False)
    testloader = DataLoader(test_data, opt.batchsize, shuffle=False)

    # Load the CNN model
    device = "cuda:0"
    model = UNet().to(device)

    # Initialize loss function
    l_bce = torch.nn.BCEWithLogitsLoss()

    # Print out some messages
    logging.info('=========== New session ===========')
    logging.info('--------- Stats and config --------')
    logging.info(f"Train dataset stats: number of images: {len(train_data)}")
    logging.info(f"Validation dataset stats: number of images: {len(val_data)}")
    logging.info(f"Test dataset stats: number of images: {len(test_data)}")
    logging.info(f"Batch size: {opt.batchsize}")
    logging.info(f"Image size: {opt.imagesize}")
    logging.info(f"Number of epochs: {opt.epochs}")


    if opt.train:

        # Initialize optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
        scheduler = ReduceLROnPlateau(optimizer, mode='min', threshold_mode='rel', factor=0.1, patience=3, threshold=0.01, cooldown=0, eps=1e-5, verbose=True) 

        train_loss_over_time = []
        val_loss_over_time = []
        val_iou_over_time = []
        val_precision_over_time = []
        val_recall_over_time = []
        val_f1_over_time = []
        
        best_iou = 0


        start_time = time.time()

        logging.info(f'--------- Begin training ---------')

        for epoch in range(opt.epochs):

            try:
                # Train
                avg_loss = train(trainloader, model, device, optimizer, l_bce)
                train_loss_over_time.append(avg_loss)

                # Validation            
                iou, precision, recall, f1, avg_val_loss = test(valloader, model, device, l_bce)
                val_loss_over_time.append(avg_val_loss)
                val_iou_over_time.append(iou)
                val_precision_over_time.append(precision)
                val_recall_over_time.append(recall)
                val_f1_over_time.append(f1)

                # Save network weights if better than previous best
                if iou > best_iou:
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, './output/weights.pth')
                    best_iou = iou
                    logging.info(f'New best, saving weights')

                curr_lr = optimizer.state_dict()['param_groups'][0]['lr']
                logging.info(f'Epoch {epoch+1}/{opt.epochs}: Train loss:{avg_loss:.8f},curr.lr:{curr_lr},v.loss:{avg_val_loss:.8f},v.IOU:{iou:.8f},v.precision:{precision:.8f},v.recall:{recall:.8f},v.f1:{f1:.8f},best v.IOU:{best_iou:.8f}')
            
                scheduler.step(avg_val_loss)

                # Early stop: If scheduler reached the lr limit and there are too many bad epochs, early stop
                if (scheduler.num_bad_epochs >= scheduler.patience) and (optimizer.state_dict()['param_groups'][0]['lr'] * scheduler.factor < scheduler.eps):
                    logging.warning(f'Stopping due to too many bad epochs')
                    break

            except KeyboardInterrupt:
                logging.warning("Stopping due to keyboard interrupt")
                break
        
        logging.info('----------- End training -----------')
    
    if opt.test:
        logging.info(f'--------- Begin testing ---------')
        iou, precision, recall, f1, avg_val_loss = test(testloader, model, device, l_bce, save_results=True)
        logging.info(f'Test results: loss:{avg_val_loss:.8f},IOU:{iou:.8f},precision:{precision:.8f},recall:{recall:.8f},f1:{f1:.8f}')
    

    # Print final info
    end_time = time.time()
    elapsed_time = datetime.timedelta(seconds=(end_time - start_time))
    logging.info(f'Total time: {elapsed_time}')

    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot loss over time
    plt.figure(figsize=(15, 10))
    plt.plot(range(len(train_loss_over_time[1:])), train_loss_over_time[1:], c="dodgerblue")
    plt.plot(range(len(val_loss_over_time[1:])), val_loss_over_time[1:], c="r")
    plt.title("Loss per epoch", fontsize=18)
    plt.xlabel("epoch", fontsize=18)
    plt.ylabel("loss", fontsize=18)
    plt.legend(['Training loss', 'Validation loss'], fontsize=18)
    filename = f'loss-{current_datetime}.svg'
    plt.savefig(os.path.join('output', filename))
    
    # Plot IOU over time
    plt.figure(figsize=(15, 10))
    plt.plot(range(len(val_iou_over_time[1:])), val_iou_over_time[1:], c="dodgerblue")
    plt.title("IoU per epoch", fontsize=18)
    plt.xlabel("epoch", fontsize=18)
    # plt.ylabel("", fontsize=18)
    plt.legend(['IoU'], fontsize=18)
    filename = f'iou-{current_datetime}.svg'
    plt.savefig( os.path.join('output', filename))
    
    # Plot F1 over time
    plt.figure(figsize=(15, 10))
    plt.plot(range(len(val_f1_over_time[1:])), val_f1_over_time[1:], c="dodgerblue")
    plt.title("F1 per epoch", fontsize=18)
    plt.xlabel("epoch", fontsize=18)
    # plt.ylabel("", fontsize=18)
    plt.legend(['F1'], fontsize=18)
    filename = f'f1-{current_datetime}.svg'
    plt.savefig( os.path.join('output', filename))

    plt.show()