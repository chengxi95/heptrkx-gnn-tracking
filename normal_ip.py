import numpy as np
import yaml
import os

with open('configs/pre_phenix.yaml') as f:
    config = yaml.load(f)
input_dir = config['output_dir']
filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                                if f.startswith('event') and not f.endswith('_ID.npz')])
ip_list = []
for filename in filenames:
    with np.load(filename) as f:
        #print(f['ip'].shape)
        ip_list.append(f['ip'][0])
print(np.array(ip_list).shape)
np.savez('normal_ip', ip=np.max(np.array(ip_list), axis=0))
