import argparse

parser = argparse.ArgumentParser(description="Tensorflow implementation of ECO")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51'])
parser.add_argument('dataset_path', type=str, default=None)
parser.add_argument('modality', type=str, choices=['RGB'])
parser.add_argument('train_list', type=str)
parser.add_argument('val_list', type=str)

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default='ECOfull', choices=['ECOfull', 'ECOlite'])
parser.add_argument('--num_segments', type=int, default=4)
parser.add_argument('--pretrained_parts', type=str, default='both',
                    choices=['scratch', '2D', '3D', 'both','finetune'])

parser.add_argument('--net2d_dropout', default=0.6, type=float)
parser.add_argument('--net3d_dropout', default=0.5, type=float)

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=16, type=int,
                    metavar='N', help='mini batch_size (default: 16)')
parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[100], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by lr_decay')
parser.add_argument('--lr_decay', default=2, type=float,
                    metavar='LRdecay', help='learning rate decay value')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
        metavar='W', help='weight decay (default: 5e-4)')

# ========================= Monitor Configs ==========================
parser.add_argument('--print_freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--save_freq', '-sf', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--resume_path', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
# TODO multiple GPUs support
parser.add_argument('--gpus', nargs='+', type=int, default=None)


if __name__ == "__main__":
    args = parser.parse_args()
    print(type(args.lr_steps))
    init_lr = args.lr
    lr_values = [init_lr / (args.lr_decay**i) for i in range(len(args.lr_steps)+1)]
    print(lr_values)
