
from src.partition_tools import pipeline_stratified_kfold
import pandas as pd
from src.utils import pkl_load, pkl_dump
from src.train_eval import train_loop, nested_kcv_train
from src.models import *
from src.data_processing import get_tensor_dataset
from torch.utils.data import DataLoader
from torch import optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Load data
cedar = pd.read_csv('./data/cedar_neoepitope_220701_scored.csv')
dataset_5fold = pd.read_csv('./data/cedar_neoepitopes_partition_5fold.csv')
dataset_10fold = pipeline_stratified_kfold('./out/cedar_peps_hobohm_0.925.pep', cedar, k=10, shuffle=True)
ics_kl = pkl_load('./output/ics_kl.pkl')
ics_shannon = pkl_load('./output/ics_shannon.pkl')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Set up hyperparams
n_epochs = 300
lr = 1e-3
batch_size = 128
# init objects
model = Net(n_filters=16, n_hidden=32, act_cnn=nn.SELU(), act_lin=nn.SELU())
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

test = dataset_5fold.query('fold==0')
valid = dataset_5fold.query('fold==1')
train = dataset_5fold.query('fold != 0 and fold !=1')

test_dataset = get_tensor_dataset(test, ics_kl, device)

x_test, y_test = test_dataset.tensors[0], test_dataset.tensors[1]

train_loader = DataLoader(get_tensor_dataset(train, ics_kl, device), batch_size=batch_size)
valid_loader = DataLoader(get_tensor_dataset(valid, ics_kl, device), batch_size=batch_size)

model, result_metrics = train_loop(model, criterion, optimizer, train_loader, valid_loader,
                                   device, n_epochs, early_stopping=True, patience=20, delta=1e-6, filename='xd')
