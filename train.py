from config import optimisation

def train_minib(args, device, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (features, label) in enumerate(train_loader):
        features, label = features.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(features)
        loss = optimisation.loss1(output, label)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # the following printing needs to be moved to config.io
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, 
                batch_idx * len(features), 
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader), 
                loss.item()))
            if args.dry_run:
                break
