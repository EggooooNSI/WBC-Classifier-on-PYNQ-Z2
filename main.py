#%%
from ideal_classfication_trainer import IdealClassificationTrainer
from tester import ClassificationTester
from cascaded_tester import CascadedClassificationTester
from wbcdataset import dataio
from config import *
from resnet_torch import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import load_ckpt_for_eval

mode = 'train' # from ['train', 'test', 'grad_cam']

data_param = {
    'type-str': 'm-g-(b-t)',  # ['b-t', 'm-g-(b-t)', 'm-g-b-t']
    'val_num_per_type': 100,
    'shuffle_data': False,
    'data_folder': '/home/ye/classification-master/BTMG/',
    'input_size': (16, 16),   # 16×16 for FPGA (dataio target_size)
}

train_param = {
    'optimizer':'adam', # ['sgd', 'adam']
    'lr': 0.00008, # adam 0.00008 for three types aug
    'batch_size': 16,
    'net': 'unet10',
    'training_epochs': 1000,
    'early_stop_epochs': 15,
    'early_stop_metrics': 'val_loss',
    'initial_channels': 16,  # 32 ch for FPGA (or 16 for smaller)
    'checkpoint': {
        'save_checkpoint': True,
        'clean_prev_ckpt_flag': False,
        'dir_name_suffix': 'unet10-16-8e-05-m-g-(b-t)_16x16',  # 16×16 input for FPGA
        'metrics': 'val_loss',
    },
    'pretrained': {
        'load_pretrained': False,
        'ckpt_dir': 'checkpoint/unet10-16-8e-05-m-g-(b-t)_16x16',
        'ckpt_num': None,
    }
    }

eval_param = {
    'dataset_for_test': ['test'],
    'ckpt_dir': 'checkpoint/unet10-16-8e-05-m-g-(b-t)_16x16',
    'ckpt_num': None,
    'cascade_param': {
        'if_cascade': True,
        'ckpt_dir': 'checkpoint/unet10-16-8e-05-m-g-(b-t)_16x16',
        'further_classify_which_type_in_first_model': 2,
    },
    'tsne_param':  # only take effect when 'if_cascade' = False
        {
            'cal_tsne': True,
            'path_to_save_data': 'tsne_data/test_tsne_and_targets.csv',
            'draw_figure': True,
        },
    'draw_prc': True,
    }

# Load data
train_loader, val_loader, test_loader, type_count = dataio(
    data_param['data_folder'],
    train_param['batch_size'],
    shuffle_data=data_param['shuffle_data'],
    val_num_per_type=data_param['val_num_per_type'],
    type_str=data_param['type-str'],
    target_size=data_param.get('input_size', None),
)

# Build model
if train_param['net'] == 'unet10': 
    if mode=='train':
        print(f"Device: {device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        initial_ch = train_param.get('initial_channels', 16)
        
        # Create model with configurable channels
        model = ResNet10(1, type_count, initial_channels=initial_ch).to(device)
        
        params = sum(p.numel() for p in model.parameters())
        print(f"\nTraining Standard ResNet10 for Tensil AI:")
        print(f"  - Float weights (Tensil will do PTQ)")
        print(f"  - {initial_ch} initial channels")
        print(f"  - Input size: {data_param.get('input_size', 'original')}")
        print(f"  - Output classes: {type_count}")
        print(f"  - Total parameters: {params:,}")
        print(f"  - Model size: ~{params * 4 / 1024:.1f} KB (FP32)")
        print(f"               ~{params * 2 / 1024:.1f} KB (FP16 after Tensil PTQ)")
    elif mode == 'test' and eval_param['cascade_param']['if_cascade'] == False:
        initial_ch = train_param.get('initial_channels', 16)
        model = ResNet10(1, type_count, initial_channels=initial_ch).to(device)
        model = load_ckpt_for_eval(eval_param['ckpt_dir'], eval_param['ckpt_num'], model)
    else:
        model1 = ResNet10(1, 3, initial_channels=train_param.get('initial_channels', 16)).to(device)
        model1 = load_ckpt_for_eval(eval_param['ckpt_dir'], eval_param['ckpt_num'], model1)
        model2 = ResNet10(1, 2, initial_channels=train_param.get('initial_channels', 16)).to(device)
        model2 = load_ckpt_for_eval(eval_param['cascade_param']['ckpt_dir'], None, model2)

# Execute training or testing
if mode == 'train':
    trainer = IdealClassificationTrainer(model, train_param, data_param)
    trainer.fit(train_loader, val_loader)
elif mode == 'test' and eval_param['cascade_param']['if_cascade'] == False:
    tester = ClassificationTester(model, eval_param['dataset_for_test'], eval_param, data_param['type-str'])
    tester.fit(train_loader, val_loader, test_loader)
elif mode == 'test' and eval_param['cascade_param']['if_cascade'] == True:
    tester = CascadedClassificationTester(model1, model2, eval_param['dataset_for_test'],
                                          eval_param['cascade_param']['further_classify_which_type_in_first_model'])
    tester.fit(train_loader, val_loader, test_loader)
elif mode == 'grad_cam':
    tester = ClassificationTester(model, eval_param['dataset_for_test'], eval_param, data_param['type-str'])
    tester.grad_cam(val_loader)
