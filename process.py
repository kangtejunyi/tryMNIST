from config import io, paths

def preprocess(args, use_cuda):
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                        'pin_memory': True,
                        'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=io.cleaner.Compose([
        io.cleaner.ToTensor(),
        io.cleaner.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = io.data1(paths.data_dir, train=True, download=True,
                        transform=transform)
    dataset2 = io.data1(paths.data_dir, train=False, download=False,
                        transform=transform)
    train_loader = io.dataloader(dataset1, **train_kwargs)
    test_loader = io.dataloader(dataset2, **test_kwargs)
    return train_loader, test_loader
