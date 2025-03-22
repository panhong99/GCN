import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
import timm
import torch.nn.functional as F
import warnings
import tqdm

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resize_pos_embed(posemb , grid_size , new_grid_size , num_extra_tokens = 1):
    #Todo split [CLS] , grid tokens
    posemb_tok , posemb_grid = posemb[: , : num_extra_tokens] , posemb[: , num_extra_tokens :]
    dim = posemb.shape[-1]

    posemb_grid = posemb_grid.reshape(1 , grid_size , grid_size , dim).permute(0 , 3 , 1 , 2) # 1 , dim , H , W
    posemb_grid = F.interpolate(posemb_grid , size = new_grid_size , mode = "bicubic" , align_corners=False)

    posemb_grid = posemb_grid.permute(0 , 2 , 3 , 1).reshape(1 , new_grid_size * new_grid_size , dim)

    #Todo reshape by image size # 32 x 32
    return torch.cat([posemb_tok , posemb_grid] , dim = 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model' , type = str , required=False , default = "vit_ti_8_32" , help = "resnet_18 or vit_ti_8_32")
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--opt_decay', type=float, default=1e-6)
    args = parser.parse_args()

    num_epochs = args.epoch

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    dataset_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, drop_last=True, shuffle=False, num_workers=8)

    print("Data load complete, start training")

    if args.model == "resnet_18":
        model = models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, 100)
        model = model.to(device)

    elif args.model == "vit_ti_8_32":
        # model = timm.create_model("vit_small_patch16_224" , pretrained = False)
        model = timm.create_model("vit_tiny_patch16_224" , pretrained = False)
        model.patch_embed.img_size = [32 , 32]
        model.patch_embed.proj = nn.Conv2d(3 , 192 , kernel_size = 8 , stride = 8)
        model.head = nn.Linear(in_features=192 , out_features=100)

        #Todo origin input_size = 192 , new input_size = 32
        resized_posemb = resize_pos_embed(model.pos_embed , 14 , 4)
        model.pos_embed = torch.nn.Parameter(resized_posemb)
        model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.opt_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(num_epochs):
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

        acc_epoch = total_correct / total_samples * 100
        avg_loss = total_loss / total_samples

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

    torch.save(model.state_dict() , f"./{args.model}")

if __name__ == '__main__':
    main()
