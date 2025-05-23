import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sentry_sdk.utils import epoch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
import timm
import torch.nn.functional as F
import warnings
import tqdm
import wandb
# from vit_pytorch import ViT
from vit import ViT
import warmup_scheduler

warnings.filterwarnings("ignore")

wandb.init(
    entity = "hails",
    project = "vit_cnn_cifar100"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model' , type = str , required=False , default = "vit" , help = "resnet or vit")
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)
    parser.add_argument("--run_name" , type = str , required=False , default = "vit_small_patch16_224" , help = "model_size")
    args = parser.parse_args()

    wandb.run.name = f"{args.run_name}"
    wandb.save()

    prams = {
        "epoch" : args.epoch,
        "batch_size" : args.batch_size,
        "lr" : args.lr
    }

    wandb.config.update(prams)

    num_epochs = args.epoch

    transform_cnn = transforms.Compose([

        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5 , 0.5 , 0.5], [0.2 , 0.2 , 0.2])
    ])

    transform_vit = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # change the color
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]),
        transforms.RandomErasing(p=0.25)
    ])

    if args.model == "resnet":

        dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_cnn)
        dataset_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_cnn)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=8)
        test_loader = DataLoader(dataset_test, batch_size=args.batch_size, drop_last=True, shuffle=False, num_workers=8)

    elif args.model == "vit":

        dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_vit)
        dataset_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_vit)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=8)
        test_loader = DataLoader(dataset_test, batch_size=args.batch_size, drop_last=True, shuffle=False, num_workers=8)


    print("Data load complete, start training")

    if args.model == "resnet":
        model = models.resnet34(pretrained = False)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, 100)
        model = model.to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.opt_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    elif args.model == "vit":
        model = ViT(img_size = 224 , patch = 16)
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                      betas=(0.9, 0.999),
                                      weight_decay=5e-5)

        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200,
                                                                    eta_min=1e-5)

        scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1.,
                                                            total_epoch=5,
                                                            after_scheduler=base_scheduler)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):

        prev_param = next(iter(model.named_parameters()))[1].detach().clone()

        model.train()
        i = 0
        total_loss, total_correct, total_samples = 0, 0, 0

        for images, labels in tqdm.tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct = (torch.argmax(preds, dim=1) == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)
            i += 1

        curr_param = next(iter((model.named_parameters())))[1].detach()

        if not torch.equal(prev_param , curr_param):
            print("Training")
        else:
            print("Not training")

        acc_epoch = total_correct / total_samples * 100
        avg_loss = total_loss / total_samples
        wandb.log({"Train_loss" : avg_loss})
        wandb.log({"Train_acc" : acc_epoch})
        print(f'Epoch [{epoch + 1}/{num_epochs}], Acc: {acc_epoch:.3f}%, Loss: {avg_loss:.4f}')

        scheduler.step()
        model.eval()

        with torch.no_grad():
            correct, total = 0, 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                class_output = model(images)
                preds = torch.softmax(class_output, dim=1)

                _, predicted = torch.max(preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total * 100
            print(f'Test Acc: {accuracy:.3f}%')
            wandb.log({"Test_acc": accuracy})
    torch.save(model.state_dict() , f"./{args.run_name}")

if __name__ == '__main__':
    main()

