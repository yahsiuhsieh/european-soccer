import torch
import torch.nn as nn


class Soccernet(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_acc = []
        self.test_acc = []

        self.fc1 = nn.Linear(770, 500)
        self.fc2 = nn.Linear(500, 400)
        self.fc3 = nn.Linear(400, 200)
        self.fc4 = nn.Linear(200, 50)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(50, 3)

    def forward(self, xb):
        xb = self.relu(self.fc1(xb))
        xb = self.relu(self.fc2(xb))
        xb = self.relu(self.fc3(xb))
        xb = self.relu(self.fc4(xb))
        xb = self.fc(xb)
        return xb

    def accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def train(self, train_loader, criterion, optimizer, epoch):
        losses = Average()
        top1 = Average()

        # # switch to train mode
        # model.train()

        for i, (input, target) in enumerate(train_loader):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = self.forward(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1 = self.accuracy(output.data, target)
            losses.update(loss.data, input.size(0))
            top1.update(prec1[0][0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            curr_lr = optimizer.param_groups[0]["lr"]
            if i % 50 == 0:
                print(
                    "Epoch: [{}/50][{}/{}]\t"
                    "LR: {}\t"
                    "Avg Loss: {loss.avg:.2f}\t"
                    "Avg Prec: {top1.avg:.2f}".format(
                        epoch, i, len(train_loader), curr_lr, loss=losses, top1=top1
                    )
                )

        print(" * Training Prec {top1.avg:.2f}".format(top1=top1))
        self.train_acc.append(top1.avg.cpu().data.numpy())

    def validate(self, val_loader, criterion):
        losses = Average()
        top1 = Average()

        # # switch to evaluate mode
        # model.eval()

        for i, (input, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            with torch.no_grad():
                output = self.forward(input)
                loss = criterion(output, target)

                # measure accuracy and record loss
                prec1 = self.accuracy(output.data, target)
                losses.update(loss.data, input.size(0))
                top1.update(prec1[0][0], input.size(0))

        print(" * Testing Prec {top1.avg:.2f}".format(top1=top1))
        self.test_acc.append(top1.avg.cpu().data.numpy())


class Average(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count