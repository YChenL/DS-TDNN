import argparse, glob, os, torch, warnings, time
from loss.tools import *
from dataLoader import train_loader
from MyModel import MyModel


parser = argparse.ArgumentParser(description = "ECAPA_trainer")
## Training Settings
parser.add_argument('--num_frames',     type=int,   default=200,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch',      type=int,   default=100,     help='Maximum number of epochs')
parser.add_argument('--batch_size',     type=int,   default=256,     help='Batch size')
parser.add_argument('--n_cpu',          type=int,   default=16,      help='Number of loader threads')
parser.add_argument('--test_step',      type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',             type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",       type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')

## Training and evaluation path/lists, save path
parser.add_argument('--train_list',     type=str,   default="/root/autodl-tmp/Vox2/train_list.txt",                help='Train list')
parser.add_argument('--train_path',     type=str,   default="/root/autodl-tmp/Vox2/train",                         help='Absolute path to the train set')
parser.add_argument('--eval_list',      type=str,   default="/root/autodl-tmp/Vox1/test_list.txt",                 help='Evaluation list')
parser.add_argument('--eval_path',      type=str,   default="/root/autodl-tmp/Vox1/test",                          help='Absolute path to the test set')
parser.add_argument('--musan_path',     type=str,   default="/root/autodl-tmp/musan",                              help='Absolute path to the MUSAN set')
parser.add_argument('--rir_path',       type=str,   default="/root/autodl-tmp/RIRS_NOISES/simulated_rirs",         help='Absolute path to the RIR set')
parser.add_argument('--save_path',      type=str,   default="exps/Dual_TDNN_SDGF1536",                             help='Path to save the score.txt and models')
parser.add_argument('--initial_model',  type=str,   default="exps/Dual_TDNN_SDGF1536/model/model_0075.model",      help='Path of the initial_model') 

## Model and Loss settings
parser.add_argument('--C',              type=int,   default=1536,   help='Channel size for the speaker encoder')
parser.add_argument('--mixedprec',      type=bool,  default=False,  help='Only do evaluation')
parser.add_argument('--m',              type=float, default=0.2,    help='Loss margin in AAM softmax')
parser.add_argument('--s',              type=float, default=30,     help='Loss scale in AAM softmax')
parser.add_argument('--k',              type=int,   default=3,      help='subcenter nums in DAM softmax')
parser.add_argument('--n_class',        type=int,   default=5994,   help='Number of speakers')

## Command
parser.add_argument('--eval',   dest='eval',  action='store_false', help='Only do evaluation')

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)



## Define the data loader
trainloader = train_loader(**vars(args))
trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, \
                                          pin_memory=True, num_workers = args.n_cpu, drop_last = True, prefetch_factor = 10)

# Search for the exist models
modelfiles = glob.glob('%s/model_0*.model'%args.model_save_path)
modelfiles.sort()

#Only do evaluation, the initial_model is necessary
if args.eval == True:
	s = ECAPAModel(**vars(args))
	print("Model %s loaded from previous state!"%args.initial_model)
	s.load_parameters(args.initial_model)
	EER, minDCF = s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)
	print("EER %2.2f%%, minDCF %.4f%%"%(EER, minDCF))
	quit()

If initial_model is exist, system will train from the initial_model
if args.initial_model != "":
	print("Model %s loaded from previous state!"%args.initial_model)
	s = ECAPAModel(**vars(args))
	s.load_parameters(args.initial_model)
	epoch = 1

## Otherwise, system will try to start from the saved model&epoch
elif len(modelfiles) >= 1:
	print("Model %s loaded from previous state!"%modelfiles[-1])
	epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
	s = ECAPAModel(**vars(args))
	s.load_parameters(modelfiles[-1])
# Otherwise, system will train from scratch
else:
    epoch = 1
    s = MyModel(**vars(args))


EERs = []
score_file = open(args.score_save_path, "a+")



while(1):
	## Training for one epoch
	loss, lr, acc = s.train_network(epoch = epoch, loader = trainLoader)

	## Evaluation every [test_step] epochs
	if epoch % args.test_step == 0:
		s.save_parameters(args.model_save_path + "/model_%04d.model"%epoch)
		EERs.append(s.eval_network(eval_list = args.eval_list, eval_path = args.eval_path)[0])
		print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%"%(epoch, acc, EERs[-1], min(EERs)))
		score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n"%(epoch, lr, loss, acc, EERs[-1], min(EERs)))
		score_file.flush()

	if epoch >= args.max_epoch:
		quit()

	epoch += 1