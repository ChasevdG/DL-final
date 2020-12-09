from .on_screen_classifier import On_Screen_Classifier, ResNet, ResidualBlock, save_model
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_on_screen_data
from torchvision import transforms, models
from .utils import accuracy


def train(args):
    from os import path
    # Initialize the logger
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    """
    Your code here, modify your HW4 code
    Hint: Use the log function below to debug and visualize your model
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load the model
    if args.model == 'cnn':
        model = On_Screen_Classifier().to(device)
    elif args.model == 'resnet18':
        model = models.resnet18(pretrained=True)
        # Froze the gradient of all the parameters except for the last layer
        for param in model.parameters():
            param.requires_grad = False
        # Add a new linear layer for classification
        fc_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
            nn.LogSoftmax(dim=1)
        )
        # Put the model on to GPU (if available)
        model.to(device)
    elif args.model == 'resnet':
        model = ResNet(ResidualBlock).to(device)
    else:
        print('Currently not support model: {}'.format(args.model))
        raise ValueError
    # Load previous trained model, if needed
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 50], gamma=0.2)

    # Load data from dataloader and perform data preprocessing (with data augmentation)
    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(transforms) if inspect.isclass(v)})
    train_data = load_on_screen_data('data', num_workers=1, transform=transform, batch_size=args.batch)
    # Set the loss function (criterion)
    if args.model == 'cnn':
        aim_loss = torch.nn.CrossEntropyLoss()
    elif args.model == 'resnet18':
        aim_loss = torch.nn.NLLLoss()
    elif args.model == 'resnet':
        aim_loss = torch.nn.CrossEntropyLoss()

    global_step = 0
    for epoch in range(args.num_epoch):
        print('Epoch: ', epoch)
        model.train()
        # Set seom statistics variables
        acc = 0
        tot_acc = 0
        num_zeros = 0
        cnt_batch = 0
        running_loss = 0.0
        # Loop for Training
        for idx, data in enumerate(train_data):
            # aim = torch.tensor(np.asarray(aim))
            img, label = data[0].to(device), data[1].to(device)
            # img, aim = img.to(device), aim.to(device)
            pred = model(img)
            # pred = pred[:,0]
            # Continuous version of focal loss
            # pred, aim = pred.type(torch.FloatTensor).to(device), aim.type(torch.LongTensor).to(device)
            # loss_val = (aim_loss(pred, aim)).mean()
            # By setting the reduction attribute of the CrossEntropyLoss(), we don't need mean() here
            loss_val = aim_loss(pred, label)
            running_loss += loss_val.item()
            # Logger, show some statistics
            if train_logger is not None and global_step % 100 == 0:
                log(train_logger, img, label, pred, global_step)

            if (idx + 1) % 2 == 0:
                print(
                    '[%d, %5d] loss: %.3f'
                    % (epoch + 1, idx + 1, running_loss / 2)
                )
                if train_logger is not None:
                    # Add writer to record the traning loss
                    train_logger.add_scalar('loss', running_loss / 2, global_step=global_step)
                running_loss = 0.0

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
            cnt_batch += 1
            zer = torch.zeros(len(pred), 2)
            zer[:, 0] = 1
            out = pred.max(1)[1].type_as(label)
            num_zeros += accuracy(zer, out)
            tot_acc += accuracy(pred, label)
        # Calculate the total accuracy and zeros ratio per epoch
        acc = tot_acc/cnt_batch
        num_zeros = num_zeros/cnt_batch
        print("Accuracy: ", acc)
        print("Zero ratio: ", num_zeros)

        # TODO: Add Validation (need some data for valiation)
        # if valid_logger is None or train_logger is None:
        #     print('epoch %-3d' % (epoch))
        # print(epoch, loss_val)
        # print(pred[0], label[0])

        # Save model periodly
        if (epoch + 1) % args.save_model_freq == 0:
            # model_state = {
            #     'state_dict': model.state_dict()
            # }
            # torch.save(
            #     model_state, '{0}/train_done_{1}.pth'
            #     .format(args.save_dir, epoch + 1)
            # )
            save_model(model, 'train_done_{0}'.format(epoch + 1))

        # schedule the lr
        scheduler.step()

    save_model(model)


def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth label (0: No Ball; 1: Has Ball)
    pred: predicted label
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    # WH2 = np.array([img.size(-1), img.size(-2)])/2
    if label[0].detach().cpu().item() == 0:
        ax.text(0, 0, r'Label: No Ball', fontsize=15)
    else:
        ax.text(0, 0, r'Label: Has Ball', fontsize=15)
    if pred[0].detach().cpu().max(0)[1].item() == 0:
        ax.text(0, 6, r'Pred: No Ball', fontsize=15)
    else:
        ax.text(0, 6, r'Pred: Has Ball', fontsize=15)
    logger.add_figure('viz', fig, global_step)
    del ax, fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=100)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([Resize((128,128)), ColorJitter(0.9, 0.9, 0.9, 0.1),RandomHorizontalFlip(), ToTensor()])')
    parser.add_argument('-w', '--size-weight', type=float, default=0.01)
    parser.add_argument('-m', '--model', type=str, default='cnn')
    parser.add_argument('-b', '--batch', type=int, default=64, help="Batchsize, default: 64")
    parser.add_argument('-sf', '--save_model_freq', type=int, default=4, help='Frequency of saving model, per epoch')

    args = parser.parse_args()
    train(args)
