import sys
from math import floor

import numpy as np
import torch
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, Dropout, AdaptiveAvgPool2d, Sequential, Linear, Module
from torchvision import models
from torchvision.models import ResNet18_Weights

from dataset import device

print("第一次初始化时，可能需要下载一些文件，请耐心等候...")
resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
print("完成!\n")


class MyModel(Module):
    use_angle = False
    angle_in_pic = False

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = ReLU(inplace=True)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = resnet.layer1
        # self.dropout1 = nn.Dropout(p=0.1)
        self.layer2 = resnet.layer2
        # self.dropout2 = nn.Dropout(p=0.1)
        self.layer3 = resnet.layer3
        # self.dropout3 = nn.Dropout(p=0.2)
        self.layer4 = resnet.layer4
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512, 10)
        self.dropout_fc = Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        # x = self.dropout1(x)
        x = self.layer2(x)
        # x = self.dropout2(x)
        x = self.layer3(x)
        # x = self.dropout3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dropout_fc(x)
        return x


class MyModelWA(MyModel):
    use_angle = True

    def __init__(self):
        super().__init__()
        self.fc = Linear(512 + 4, 10)
        self.angle_process = Sequential(
            Linear(4, 4),
            ReLU()
        )

    def forward(self, x, ag, eg):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        # x = self.dropout1(x)
        x = self.layer2(x)
        # x = self.dropout2(x)
        x = self.layer3(x)
        # x = self.dropout3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(torch.concatenate((x, ag.min(2)[0], ag.max(2)[0], eg.min(2)[0], eg.max(2)[0]), dim=1))
        x = self.dropout_fc(x)
        return x


class MyModel2(Module):
    use_angle = False
    angle_in_pic = False

    def __init__(self):
        super(MyModel2, self).__init__()
        self.res1 = Sequential(
            Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            resnet.layer1,
            # Dropout(p=0.1),
            resnet.layer2,
            # Dropout(p=0.1),
            resnet.layer3,
            # Dropout(p=0.2),
            resnet.layer4,
            AdaptiveAvgPool2d((1, 1)),
        )
        self.res2 = self.res1 = Sequential(
            Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            resnet.layer1,
            # Dropout(p=0.1),
            resnet.layer2,
            # Dropout(p=0.1),
            resnet.layer3,
            # Dropout(p=0.2),
            resnet.layer4,
            AdaptiveAvgPool2d((1, 1))
        )
        self.res3 = self.res1 = Sequential(
            Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            resnet.layer1,
            # Dropout(p=0.1),
            resnet.layer2,
            # Dropout(p=0.1),
            resnet.layer3,
            # Dropout(p=0.2),
            resnet.layer4,
            AdaptiveAvgPool2d((1, 1))
        )
        self.res4 = self.res1 = Sequential(
            Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            resnet.layer1,
            # Dropout(p=0.1),
            resnet.layer2,
            # Dropout(p=0.1),
            resnet.layer3,
            # Dropout(p=0.2),
            resnet.layer4,
            AdaptiveAvgPool2d((1, 1))
        )
        self.total_se = Sequential(
            # WeightLayer(),
            Linear(512 * 4, 512),
            Dropout(p=0.3),
            ReLU(),
            Linear(512, 10),
            Dropout(0.5)
        )

    def forward(self, x):
        if x.shape[1] == 4:
            a = self.res1(x[:, 0].unsqueeze(1)).squeeze()
            b = self.res2(x[:, 1].unsqueeze(1)).squeeze()
            c = self.res3(x[:, 2].unsqueeze(1)).squeeze()
            d = self.res4(x[:, 3].unsqueeze(1)).squeeze()
            x = self.total_se(torch.concatenate((a, b, c, d), 1))

        else:
            a = self.res1(x[:, 0].unsqueeze(1)).squeeze()
            b = self.res2(x[:, 1].unsqueeze(1)).squeeze()
            x = self.total_se(torch.concatenate((a, b), 1))

        return x


class ModelWithAngle(MyModel2):
    use_angle = True
    angle_in_pic = False

    def __init__(self):
        super().__init__()
        self.ag_net = Sequential(
            Conv2d(1, 32, (2, 7)),
            BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            Conv2d(32, 64, (1, 7)),
            BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ReLU(inplace=True),
            #             MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            AdaptiveAvgPool2d((1, 1))
        )
        self.total_se[0] = Linear(2112, 512)

    def forward(self, x, ag, eg):
        angle = torch.concatenate((ag, eg), dim=1)
        angle = angle.unsqueeze(1)
        a = self.res1(x[:, 0].unsqueeze(1)).squeeze()
        b = self.res2(x[:, 1].unsqueeze(1)).squeeze()
        c = self.res3(x[:, 2].unsqueeze(1)).squeeze()
        d = self.res4(x[:, 3].unsqueeze(1)).squeeze()
        e = self.ag_net(angle).view(x.shape[0], -1)
        total = torch.concatenate((a, b, c, d, e), dim=1)
        return self.total_se(total)


class ModelWithAngleInPic(MyModel2):
    use_angle = True
    angle_in_pic = True

    def forward(self, x, ag, eg):
        return super().forward(x)


class ModelWA3(MyModel2):
    use_angle = True
    angle_in_pic = False

    def __init__(self):
        super().__init__()
        self.angle_process = Sequential(
            Linear(4, 4),
            ReLU()
        )
        self.total_se[0] = Linear(2048 + 4, 512)

    def forward(self, x, ag, eg):
        a = self.res1(x[:, 0].unsqueeze(1)).squeeze()
        b = self.res2(x[:, 1].unsqueeze(1)).squeeze()
        c = self.res3(x[:, 2].unsqueeze(1)).squeeze()
        d = self.res4(x[:, 3].unsqueeze(1)).squeeze()
        e = self.angle_process(torch.concatenate((ag.min(2)[0], ag.max(2)[0], eg.min(2)[0], eg.max(2)[0]), dim=1))
        total = torch.concatenate((a, b, c, d, e), dim=1)
        return self.total_se(total)
        # print(ag.shape)


class ModelWAAve(MyModel2):
    use_angle = True
    angle_in_pic = False

    def __init__(self):
        super().__init__()
        self.total_se = Sequential(
            MaxPool2d((4, 1), stride=1),
            Linear(512, 128),
            ReLU(),
            Dropout(p=0.3),
            Linear(128, 10),
            ReLU(),
            Dropout()
        )

    def forward(self, x, ag, eg):
        a = self.res1((x[:, 0] / ag).unsqueeze(1)).squeeze().unsqueeze(1)
        b = self.res2((x[:, 1] / ag).unsqueeze(1)).squeeze().unsqueeze(1)
        c = self.res3((x[:, 2] / eg).unsqueeze(1)).squeeze().unsqueeze(1)
        d = self.res4((x[:, 3] / eg).unsqueeze(1)).squeeze().unsqueeze(1)

        total = torch.concatenate((a, b, c, d), dim=1)
        return self.total_se(total)


def train(dataloader, model, loss_fn, optimizer, label='训练'):
    bar_length = 20
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    dataloader.dataset.include_angle = model.angle_in_pic
    for batch, (cls, uni, ag, eg) in enumerate(dataloader):
        cls, uni = cls.to(device), uni.to(device)
        ag, eg = ag.to(device).float(), eg.to(device).float()

        pred = model(uni).squeeze() if not model.use_angle else model(uni, ag, eg).squeeze()
        loss = loss_fn(pred, cls)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), (batch + 1) * len(cls)

        progress = floor(current / size * bar_length)
        sys.stdout.write(f"{label}: [{progress * '='}{(bar_length - progress) * ' '}] [{current:>5d}/{size:>5d}]\r")
        sys.stdout.flush()
        total_loss += loss
    print(f"{label}: [{bar_length * '='}] 完成  总误差: {total_loss:>3f}{' ' * 30}")
    return total_loss


def train_all(epochs, train_loader, test_loader, model, loss_fn, optimizer, self_test_count=0, file_name='Model',
              save_interval=-1):
    losses = []
    self_losses = []
    currs = []
    self_currs = []

    for i in range(epochs):
        print(f"{'-' * 30}\nEpoch {i + 1}\n{'-' * 30}")
        loss = train(train_loader, model, loss_fn, optimizer)
        losses.append(loss)
        if self_test_count > 0:
            self_curr, self_loss = test(train_loader, model, loss_fn, '自测', self_test_count)
            self_currs.append(self_curr)
            self_losses.append(self_loss)
        curr = test(test_loader, model, loss_fn)[0]
        currs.append(curr)
        if (save_interval != -1 and (i + 1) % save_interval == 0) or curr > 90:
            torch.save(model, f'{file_name}_epoch_{i + 1}.pt')
            print("模型已保存!")
        print()
    return losses, self_losses, currs, self_currs


def test(dataloader, model, loss_fn, label='测试', cnt=-1):
    model.eval()

    batch = dataloader.batch_size
    test_loss, correct = 0, 0
    count = 0
    size = len(dataloader.dataset) if cnt == -1 else cnt
    bar_length = 20

    dataloader.dataset.include_angle = model.angle_in_pic

    with torch.no_grad():
        for cls, uni, ag, eg in dataloader:
            cls, uni, ag, eg = cls.to(device), uni.to(device), ag.to(device).float(), eg.to(device).float()

            pred = model(uni).squeeze() if not model.use_angle else model(uni, ag, eg).squeeze()
            test_loss += loss_fn(pred, cls).item()
            correct += (pred.argmax(1) == cls.argmax(1)).type(torch.float).sum().item()
            count += 1

            progress = floor(count * batch / size * bar_length)
            sys.stdout.write(
                f"{label}: [{progress * '='}{(bar_length - progress) * ' '}] [{count * batch:>5d}/{size:>5d}]\r")
            sys.stdout.flush()

            if count * batch >= size:
                break
    test_loss /= count * batch  # 实际测试的数量 = cnt * 一批的数量（16）
    correct /= count * batch
    print(f"{label}: [{bar_length * '='}] 完成  准确率: {(100 * correct):>0.1f}%,平均误差: {test_loss:>8f}")
    return correct * 100, test_loss


def validate(dataloader, model, postprocess=None, refusion=0.7):
    model.eval()
    results = []

    dataloader.dataset.include_angle = model.angle_in_pic

    for file, uni, ag, eg in dataloader:
        uni, ag, eg = uni.to(device), ag.to(device).float(), eg.to(device).float()

        pred = model(uni).squeeze() if not model.use_angle else model(uni, ag, eg).squeeze()
        if callable(postprocess):
            pred = postprocess(pred)

        pred = pred.cpu().detach().numpy()
        for i in range(pred.shape[0]):
            pred_max = np.max(pred[i])
            pred_cls = np.argmax(pred[i]) + 1 if pred_max > refusion else 0
            results.append((file[i], pred_cls, pred_max))
    print("验证完成！")

    return results
