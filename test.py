from config import optimisation, modes

def test_minib(device, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with modes.silent():
        for features, label in test_loader:
            features, label = features.to(device), label.to(device)
            output = model(features)
            test_loss += optimisation.loss1(
                output, label, reduction='sum'
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of max log-probab as predicted class
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    # the following printing needs to be moved into config.io
    print('\nTest: loss: {:.4f}, accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, 
        correct, 
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
