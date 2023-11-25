import torch 
import argparse

from attack import Attack
from utils import *
from data.dataset import data_loader

""" 
python generate_attacks.py --epsilons [perturbation size:float] --attack_list [attack #1] [attack #2] --vit_path [path to the downloaded ViT model]  
--attack_images_dir [path to create folder and save attack samples]
"""

parser = argparse.ArgumentParser(description='Generate Attack from ViT')

parser.add_argument('--epsilons', type=float , 
                    help='Perturbations Size')
parser.add_argument('--attack_list', type=str , nargs='+',
                    help='Attack List to Generate')
parser.add_argument('--vit_path', type=str ,
                    help='pass the path for the downloaded MLPs folder')
parser.add_argument('--attack_images_dir', type=str ,
                    help='Directory to save the generated attacks')
parser.add_argument('--root_dir', type=str ,
                    help='Directory to clean data')

args = parser.parse_args()

loader_, dataset_ = data_loader(root_dir=args.root_dir)

model = torch.load(args.vit_path).cuda()
model.eval()


#Generate and save attacks
generate_save_attacks(
    attack_names= args.attack_list,
    model= model,
    samples= loader_['test'], 
    classes= ['Normal', 'Tuberculosis'],
    attack_image_dir= args.attack_images_dir,
    epsilon=args.epsilons,
)
