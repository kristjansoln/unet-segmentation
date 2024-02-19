import argparse
import sys
import numpy as np
from dataset_loader import DatasetFolder
from torch.utils.data import DataLoader
from network import UNet
import torch
from tqdm import tqdm
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import datetime
import time
import matplotlib.pyplot as plt
import logging


def calculate_iou(predictions, masks):
    # pred and gt are binary tensors with shape [batch_size, ...]

    pred = torch.squeeze(predictions)
    gt = torch.squeeze(masks)

    intersection = torch.logical_and(pred, gt)
    union = torch.logical_or(pred, gt)
    iou = torch.sum(intersection) / torch.sum(union)
    return iou.mean()


# Base training function
def train(trainloader, model, device, optimizer, loss_function):

    model.train()
    avg_loss = []

    for images, masks, img_paths, mask_paths in tqdm(trainloader):

        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad() # Set all gradients to 0

        predictions = model(images) # Feedforward
        # out = softmax(predictions)
        # out = sigmoid(predictions)
        
        loss = loss_function(predictions, masks) # Calculate the error of the current batch
        avg_loss.append(loss.cpu().detach().numpy())

        loss.backward() # Calculate gradients with backpropagation
        optimizer.step() # optimize weights for the next batch

    avg_loss = np.mean(avg_loss)
    return avg_loss


# Base validation/testing function
def test(testloader, model, device, loss_function):
    
    model.eval()
    iou = []
    avg_val_loss = []

    with torch.no_grad():
        # for images, masks, img_paths, mask_paths in tqdm(testloader):
        for images, masks, img_paths, mask_paths in tqdm(testloader):
            images, masks = images.to(device), masks.to(device)

            predictions = model(images)
            out = torch.sigmoid(predictions)
            
            # Calculate IoU for training data
            predicted_masks_bin = torch.sigmoid(out) > 0.5
            # predicted_masks_bin = Sigmoid(out) > 0.5
            iou.append(calculate_iou(predicted_masks_bin, masks).cpu().numpy())

            avg_val_loss.append(loss_function(predictions, masks).cpu().detach().numpy())

            # plt.imshow(masks[0].cpu())
            # plt.show()
            # plt.imshow(predicted_masks_bin[0].cpu())
            # plt.show()

    iou = np.mean(iou)  # Calculate the total accuracy of the model
    avg_val_loss = np.mean(avg_val_loss)

    return iou, avg_val_loss

# TODO: Add inference which stores the resulting masks (and scores/loss?)


def handle_exception(exc_type, exc_value, exc_traceback):
    logging.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))


# MAIN
    
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler('./output/training.log', mode='a'),
                              logging.StreamHandler()])
    
    # Log any unhandled exceptions
    sys.excepthook = handle_exception

    # arguments that can be defined upon execution of the script
    options = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # options.add_argument('--train', action='store_true', help='Train the model.')
    # options.add_argument('--test', action='store_true', help='Test the model.')
    
    # options.add_argument('--traincsv', default='dataset/train.csv', help='directory of the train CSV')
    # options.add_argument('--testcsv', default='dataset/test.csv', help='directory of the test CSV')
    # options.add_argument('--valcsv', default='dataset/val.csv', help='directory of the validation CSV')
    options.add_argument('--batchsize', type=int, default=1, help='batch size')
    options.add_argument('--epochs', type=int, default=2, help='number of training epochs')

    # TODO: REMOVE AFTER TESTING
    options.add_argument('--traincsv', default='dataset/val.csv', help='directory of the train CSV')
    options.add_argument('--testcsv', default='dataset/val.csv', help='directory of the test CSV')
    options.add_argument('--valcsv', default='dataset/val.csv', help='directory of the validation CSV')
    options.add_argument('--imagesize', type=int, default=(624, 1104), help='size of the image (height, width)') # Needs to be divisible by 16
    # options.add_argument('--imagesize', type=int, default=(256, 256), help='size of the image (height, width)') # Needs to be divisible by 16
    # options.add_argument('--imagesize', type=int, default=(512,384), help='size of the image (height, width)')
    # options.add_argument('--imagesize', type=int, default=(622, 1104), help='size of the image (height, width)')

    opt = options.parse_args()

    PRE__MEAN = [0.5, 0.5, 0.5]
    PRE__STD = [0.5, 0.5, 0.5]
    
    # Define transforms for dataset augmentation
    # TODO: Add transformations
    image_and_mask_transform_train=A.Compose([A.Resize(opt.imagesize[0], opt.imagesize[1]),
                                              A.HorizontalFlip(p=0.5),
                                              A.VerticalFlip(p=0.5),
                                                ToTensorV2()])
    
    image_only_transform_train=A.Compose([A.Normalize(PRE__MEAN, PRE__STD),
                                        A.Resize(621, 1104), # Convert the image to mask dimensions
                                        A.RandomBrightnessContrast()])
    
    image_and_mask_transform_test=A.Compose([A.Resize(opt.imagesize[0], opt.imagesize[1]),
                                              A.HorizontalFlip(p=0.5),
                                              A.VerticalFlip(p=0.5),
                                                ToTensorV2()])
    
    image_only_transform_test=A.Compose([A.Normalize(PRE__MEAN, PRE__STD), 
                                         A.Resize(621, 1104) # Convert the image to mask dimensions
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

    # Initialize the loss function and iou
    # l_ce = torch.nn.CrossEntropyLoss()
    l_bce = torch.nn.BCEWithLogitsLoss()
    # jaccard = BinaryJaccardIndex(threshold=0.5).to(device)

    # Init softmax layer
    # softmax = torch.nn.Softmax(dim=1)
    # sigmoid = torch.nn.Sigmoid()

    # Conversion from BGR to single channel grayscale images - for masks
    # to_gray = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
    # gray_kernel = torch.FloatTensor([[[[0.114]], [[0.587]], [[0.299]]]])
    # to_gray.weight = torch.nn.Parameter(gray_kernel, requires_grad=False)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))

    train_loss_over_time = []
    val_loss_over_time = []
    val_iou_over_time = []

    best_iou = 0

    if not os.path.exists('./output'):
        os.mkdir('./output')
    
    logfile = os.path.join('./output/log.txt')

    # Print out some messages
    logging.info('======= New training session ======')
    logging.info('--------- Stats and config --------')
    logging.info(f"Train dataset stats: number of images: {len(train_data)}")
    logging.info(f"Validation dataset stats: number of images: {len(val_data)}")
    logging.info(f"Test dataset stats: number of images: {len(test_data)}")
    logging.info(f"Batch size: {opt.batchsize}")
    logging.info(f"Image size: {opt.imagesize}")
    logging.info(f"Number of epochs: {opt.epochs}")

    start_time = time.time()

    logging.info(f'--------- Begin training ---------')

    for epoch in range(opt.epochs):

        try:
            # Train
            avg_loss = train(trainloader, model, device, optimizer, l_bce)
            train_loss_over_time.append(avg_loss)

            # Validation
            # TODO: Add better performance measures
            iou, avg_val_loss = test(testloader, model, device, l_bce)
            val_iou_over_time.append(iou)
            val_loss_over_time.append(avg_val_loss)

            # Save network weights if better than previous best
            if iou > best_iou:
                torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, './output/CNN_weights.pth')
                best_iou = iou
                logging.info(f'New best, saving weights')

            logging.info(f'Epoch {epoch+1}/{opt.epochs}: Train loss: {avg_loss:.8f}, val. loss: {avg_val_loss:.8f}, val. IOU: {iou:.8f}, best val. IOU: {best_iou:.8f}')
           
            # TODO: Implement early stop
            # TODO: Implement scheduler

        except KeyboardInterrupt:
            logging.warning("Stopping due to keyboard interrupt")
            break
    
    logging.info('----------- End training -----------')
    
    # Print final info
    end_time = time.time()
    elapsed_time = datetime.timedelta(seconds=(end_time - start_time))
    logging.info(f'Total training time: {elapsed_time}')

    # Plot loss over time
    plt.figure(figsize=(15, 10))
    plt.plot(range(len(train_loss_over_time[1:])), train_loss_over_time[1:], c="dodgerblue")
    plt.plot(range(len(val_loss_over_time[1:])), val_loss_over_time[1:], c="r")
    plt.title("Loss per epoch", fontsize=18)
    plt.xlabel("epoch", fontsize=18)
    plt.ylabel("loss", fontsize=18)
    plt.legend(['Training loss', 'Validation loss'], fontsize=18)
    # TODO: Add datetime to filename
    filename = f'loss.svg'
    plt.savefig(os.path.join('output', filename))
    
    # Plot equal error rate over time
    plt.figure(figsize=(15, 10))
    plt.plot(range(len(val_iou_over_time[1:])), val_iou_over_time[1:], c="dodgerblue")
    plt.title("IoU per epoch", fontsize=18)
    plt.xlabel("epoch", fontsize=18)
    # plt.ylabel("EER, AUC", fontsize=18)
    plt.legend(['IoU'], fontsize=18)
    # TODO: Add datetime to filename
    filename = f'iou.svg'
    plt.savefig( os.path.join('output', filename))

    plt.show()