import os
import glob
import itertools
import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

sacred.SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
sacred.SETTINGS.CAPTURE_MODE = 'no'

ex = Experiment('FSS_Weld_Thesis')
ex.captured_out_filter = apply_backspaces_and_linefeeds

source_folders = ['.', './dataloaders', './util']
sources_to_save = list(itertools.chain.from_iterable(
    [glob.glob(f'{folder}/*.py') for folder in source_folders]
))
for f in ['dpanet_pml.py', 'spm.py', 'pml_loss.py', 'resnet_cbam.py']:
    if f not in sources_to_save:
        sources_to_save.append(f)
for source_file in sources_to_save:
    if os.path.exists(source_file):
        ex.add_source_file(source_file)


@ex.config
def cfg():
    seed = 1234
    gpu_id = 0
    mode = 'train'
    exp_str = 'chapter4_tcc_head_plus_ldr_voc'

    dataset = 'VOC'
    label_sets = 0
    input_size = (256, 256)
    ignore_label = 255
    batch_size = 2
    n_steps = 50000
    start_step = 0
    num_workers = 4
    val_interval = 2000
    ckpt_path = ''
    save_best_only = True

    task = dict(
        n_ways=1,
        n_shots=1,
        n_queries=1,
    )

    model = dict(
        backbone='plain',
        use_tcc_head=True,
        tcc_fuse_skel=0.25,
        tcc_fuse_edge=0.15,
        n_parts=6,
        alpha=1.0,
        tau=0.2,
        temperature=20.0,
    )

    loss = dict(
        tcc_skel_weight=0.1,
        tcc_edge_weight=0.1,
        tcc_skel_iters=25,
        tcc_skel_radius=1,
        tcc_edge_radius=2,
        ldr_weight=0.0,
        ce_weight=0.7,
        epml_weight=0.03,
        edge_radius=3,
        edge_weight=4.0,
    )

    optim = dict(
        lr=1.0e-4,
        weight_decay=1e-4,
    )
    use_amp = False
    use_ldr = False

    path = {
        'log_dir': './runs',
        'init_path': './initmodel/resnet50-19c8e357.pth',
        'VOC': {
            'data_dir': r'/mnt/d/USST/Code/PSCNet++/Pascal/VOCdevkit/VOC2012',
            'data_split': 'trainaug',
            'val_split': 'val',
        },
        'COCO': {
            'data_dir': './data/COCO/',
            'data_split': 'train2014',
            'val_split': 'val2014',
        },
        'HANFEG': {
            'data_dir': './data/Hanfeg/',
            'data_split': 'train',
            'val_split': 'val',
        },
    }


@ex.config_hook
def add_observer(config, command_name, logger):
    exp_name = f'{ex.path}_{config["exp_str"]}'
    observer = FileStorageObserver.create(os.path.join(config['path']['log_dir'], exp_name))
    ex.observers.append(observer)
    return config
