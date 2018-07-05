import argparse


###################################### ARGUMENTS

parser = argparse.ArgumentParser(description='Specify some hyper parametres:')

# Training
parser.add_argument("-b",   help="Batch size",             default=64,     type=int)
parser.add_argument("-lr",  help="Learning rate",          default=0.01,   type=float)
parser.add_argument("-mo",  help="Momentum",               default=0.9,    type=float)
parser.add_argument("-wd",  help="Weight decay",           default=0.0005, type=float)
parser.add_argument("-e",   help="Number of total epochs", default=5,      type=int)
parser.add_argument("-dlb", help="Drop last batch",        action="store_true")
parser.add_argument('-r',   help='path to latest checkpoint' default='',   type=str,   metavar='PATH')

# Hardware
parser.add_argument("-cpu", help="Do not use cuda",        action="store_true")
parser.add_argument('-gpus',help='Use multiple GPUs',      action='store_true')

# Data
parser.add_argument('data', metavar='DIR', help='path to dataset') # [imagenet-folder with train and val folders]

parser.add_argument("-vp",  help="Validation percentage",  default=0.05,  type=float)
parser.add_argument('-nw',  help="Number of workers",      default=4,     type=int)

# Model
parser.add_argument('-a',   help='Model architecture: ',   default='resnet18', type=str)
parser.add_argument("-pre", help="Load pre-trained model", action="store_true")
parser.add_argument("-cp",  help="Use check_point",        action="store_true")


# Debug
parser.add_argument('-e', '--evaluate',        dest='evaluate',   action='store_true',  help='evaluate model on validation set')
parser.add_argument('--print-freq', '-p',      default=10,   type=int,   metavar='N',   help='print frequency (default: 10)')
parser.add_argument('-v',   help="Pring debug info",       action="store_true")
parser.add_argument('-d',   help="Remote debug",           action="store_true")

args = parser.parse_args()


epochs          = args.e
batch_size      = args.b
learning_rate   = args.lr
momentum        = args.mo
weight_decay    = args.wd
resume          = args.r

val_percent     = args.vp
gpu             = not args.cpu
multiple_gpus   = args.gpus
num_workers     = args.nw
drop_last_batch = args.dlb

architecture    = args.a
pretrained      = args.pre
check_point     = args.cp
verbose         = args.v

# TODO: Poner como argumentos
gpu_count       = 1
images_per_gpu  = 1

#shuffle_data    = True
CROP_SIZE       = 256