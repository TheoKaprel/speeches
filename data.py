import torch
import torch.utils.data as data_utils
from functions import create_dicts, encode, create_data

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

# dev = "cpu"
device = torch.device(dev)
print(device)

# Path to the training data file containing speechs
PATH = 'candidats.txt'
# Text-cutting params
SEQ_LEN = 50
SEQ_SKIP = 3
BATCH_SIZE = 64


TEXT = open(PATH).read().lower()[0:1500000]
ALPHABET = sorted(list(set(TEXT)))
NB_CHARS = len(ALPHABET)
CHAR_INDEX, INDEX_CHAR = create_dicts(ALPHABET)
ENCODED_TEXT = encode(TEXT, CHAR_INDEX, NB_CHARS, bool)
INPUTS, TARGETS = create_data(ENCODED_TEXT, SEQ_LEN, SEQ_SKIP)
inputs = torch.from_numpy(INPUTS).float().to(device)
targets = torch.from_numpy(TARGETS).float().to(device)

print('loaded!')


prct_train = 80
id_train = int(inputs.shape[0]*prct_train/100)



#Data
train = data_utils.TensorDataset(inputs[:id_train,:,:], targets[:id_train,:])
train_loader = data_utils.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

test = data_utils.TensorDataset(inputs[id_train:,:,:], targets[id_train:,:])
test_loader = data_utils.DataLoader(test,batch_size = BATCH_SIZE, shuffle = True)

# torch.save(train_loader, 'train_loader.pt')
# torch.save(test_loader, 'test_loader.pt')

print('saved!')

