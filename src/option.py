import argparse

parser = argparse.ArgumentParser(description='Depth AutoEncoder training on KITTI',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset-format', default='sequential', metavar='STR',
                    help='dataset format, stacked: stacked frames (from original TensorFlow code) \
                    sequential: sequential folders (easier to convert to with a non KITTI/Cityscape dataset')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('--batch_size', default=24, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', default=0.00002, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-i', '--img_test', dest='img_test', action='store_true',
                    help='img test on validation set')
parser.add_argument('-r', '--real_test', dest='real_test', action='store_true',
                    help='test on Eigen test split')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                    help='csv where to save per-gradient descent train stats')
parser.add_argument('--result_dir', type=str, default='./AE_results')
parser.add_argument('--model_dir',type=str, default = './AE_trained_model_lr0000')
parser.add_argument('--RtoD_model_dir',type=str, default = './AE_RtoD_trained_model_lr0004_color_nonMulti/epoch_18_AE_depth_loss_0.2561.pkl')
parser.add_argument('--gpu_num', type=str, default = "2")
parser.add_argument('--norm', type=str, default = "Batch")
parser.add_argument('--mode', type=str, default = "DtoD")
parser.add_argument('--height', type=int, default = 128)
parser.add_argument('--width', type=int, default = 416)
parser.add_argument('--dataset', type=str, default = "KITTI")
parser.add_argument('--img_save', action='store_true', help='result image save')

args = parser.parse_args()