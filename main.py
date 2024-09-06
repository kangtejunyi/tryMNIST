from config import io, hardware, seed, optimisation, tunnings, paths

import process
import architecture
import train
import test

def main():
    args = io.parser()
    use_cuda = hardware.graphical1() and not args.no_cuda
    use_mps = hardware.graphical2() and not args.no_mps
    dispatcher = hardware.deploy(
        use_cuda, use_mps
    )
    seed.seed1(
        args.seed
    )
    modeler = architecture.NN().to(
        dispatcher
    )
    optimizer = optimisation.solver1(
        modeler.parameters(), tunnings.lr
    )
    scheduler = optimisation.scheduler1(
        optimizer, step_size=1, gamma=args.gamma
    )
    train_loader, test_loader = process.preprocess(args, use_cuda)
    # the following code needs to be moved into process.py
    for epoch in range(1, args.epochs + 1):
        train.train_minib(
            args,
            dispatcher,
            modeler,
            train_loader,
            optimizer,
            epoch
        )
        test.test_minib(
            dispatcher,
            modeler,  
            test_loader
        )
        scheduler.step()

    if args.save_model:
        io.writter(modeler.state_dict(), \
                   '/'.join([paths.model_dir, "mnist_cnn.pt"]))


if __name__ == '__main__':
    main()