import numpy as np
import yaml
import os

config = yaml.load('configs/pre_phenix.yaml')
input_dir = config['output_dir']
filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                                if f.startswith('event') and not f.endswith('_ID.npz')])
ip_list = []
for filename in filenames:
    with np.load(filename) as f:
        ip_list.append(f['ip'])

np.savez('normal_ip', ip=np.max(np.array(ip_list), axis=0))
