## Overview
Use ultrasound signal to regress the mean arterial blood pressure. 
The inputs to the models include the two vectors: Arterial Area Waveform 
and Blood Flow Velocity Waveform, and anthropometric features: age, height, weight, heartrate, sex. The goal is to regress the Mean 
Arterial Pressure (MAP) value.

## Pre-processing
Place the raw .mat files in `data/measured_mit_v1/raw` and run:
```bash
python preprocess_part[dataset part].py
```

## Training - Single Fold/Run
```bash
python measured_mit_v1_train.py configs/train/[train_config] --fold=[fold num] \
--seed=[random seed num] --run_eval=[true or false, whether or not to run testing code automatically]
```

## Testing - Single Fold/Run
```bash
python measured_mit_v1_eval.py configs/eval/[eval_config] --run-dir=runs/[run_dir] \
--experiment_key=[comet experiment key] --part=[dataset part (1 or 2)] --fold=[fold num] --seed=[random seed num]
```

## Training and Testing - Multiple Folds
```bash
python train_folds_wrapper.py configs/train/[train_config].yml --seed=[random seed num]
```

## Add new modules to the repo
### New dataset
Add new files to the `core/datasets` directory. For each dataset, need 
implement the `__getitem__` and `__len__` methods.

Here is the example for the PWDB dataset:

```python
def __getitem__(self, index: int):
    data_this = {}
    for feat_name in (self.scalar_feat_name + ['a', 'map', 'id']):
        if feat_name == 'a':
            a_wave = self.raw['a'][index]
            v_wave = self.raw['v'][index]

            data_this['a'] = a_wave.astype(np.float32)
            data_this['v'] = v_wave.astype(np.float32)
        else:
            data_this[feat_name] = self.raw[feat_name][index].astype(
                np.float32)

    return data_this

def __len__(self) -> int:
    return self.instance_num
```

After defining the model, add it to the `core/builder.py` file as below:

```python
def make_dataset():
    if configs.dataset.name == 'pwdb':
        from core.datasets import Pwdb
        dataset = Pwdb(root=configs.dataset.root,
                       split_ratio=configs.dataset.split_ratio,
                       location=configs.dataset.location,
                       resample_len=configs.dataset.resample_len,
                       augment_setting=configs.dataset.augment_setting)
```

### New model
To add new model, you can write a model inherited from `nn.Module` and put 
the file in `core/models` directory.

Here is the example for one FC model:

```python
class FC(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 layer_num: int,
                 ):
        super().__init__()
        self.input_layer = nn.Linear(in_ch, out_ch)

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            self.layers.append(nn.Linear(out_ch, out_ch))

        self.regress = nn.Linear(out_ch, 1)

    def forward(self, x):
        x = torch.stack((x['pwv'],
                         x['comp'],
                         x['z0'],
                         x['deltat'],
                         x['pp'],
                         torch.mean(x['a'], dim=-1),
                         torch.mean(x['v'], dim=-1)), dim=-1)

        x = self.input_layer(x)  # N, C

        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x))

        x = self.regress(x)

        return x.squeeze(dim=-1)
```

After defining the model, add it to the `core/builder.py` file as below:

```python
def make_model() -> nn.Module:
    if configs.model.name == 'fc':
        from core.models import FC
        model = FC(in_ch=configs.model.in_ch,
                   out_ch=configs.model.out_ch,
                   layer_num=configs.model.layer_num)
```

### Using CometML Logger
To enable proper usage of the CometML logger:
1. Set the environment variable $COMET_API_KEY to your own Comet API key.
2. Set the project name as desired in the CometWriter class in `callbacks.py`



### Results
The table below shows the standard deviation of the prediction error (in mmHg) for 6 models: Baseline (FC), LSTM (waveforms only), Transformer (waveforms only), CNN (waveforms only), Transformer + anthropometric data, CNN + anthropometric data. The results are computed in the bucketed dataset split setting, and the error std is shown per seed and on average, along with the number of parameters and FLOPs of each model.

| **Model**                | **Error Std Seed 1** | **Error Std Seed 2** | **Error Std Seed 3** | **Avg Error Std**  | **\#Params** | **MFLOPs** |
|--------------------------|----------------------|----------------------|----------------------|--------------------|--------------|------------|
| **Baseline**             | 9.08                 | 9.58                 | 9.88                 | **9.51**           | 12.9K        | 0.026      |
| **LSTM**                 | 10.67                | 9.66                 | 9.12                 | **9.81**           | 1.58M        | 317.5      |
| **Transformer**          | 8.76                 | 8.61                 | 9.18                 | **8.85**           | 4.73M        | 943.9      |
| **CNN**                  | 8.86                 | 8.92                 | 8.64                 | **8.81**           | 37.3K        | 7.44       |
| **Transformer + anthro** | 8.60                 | 8.98                 | 8.40                 | **8.7**            | 4.73M        | 944        |
| **CNN + anthro**         | 7.56                 | 8.29                 | 7.92                 | **7.92**           | 37.4K        | 7.48       |
