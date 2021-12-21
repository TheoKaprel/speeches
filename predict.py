import numpy as np
import torch
from network import NeuralNetwork
from functions import encode
from data import SEQ_LEN,SEQ_SKIP,ALPHABET, NB_CHARS, CHAR_INDEX, INDEX_CHAR

# Sampling dans la distribution de probabilités prédite
def sample(preds, temperature=1.0, do_sample=True):
    preds = preds.detach().numpy()
    if do_sample:
        # Avec sampling
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        index = np.argmax(probas)
        
    else:
        # Sans sampling
        preds = np.reshape(preds, (1, preds.shape[0]))
        index = np.argmax(preds)
    
    z = torch.zeros(NB_CHARS)
    z[index] = 1
    return z

def predict_single_input(model, sentence):
    x = encode(sentence, CHAR_INDEX, NB_CHARS, float)
    x = torch.from_numpy(x).view((1,x.shape[0],x.shape[1])).float()
    preds = model(x)
    if len(preds.shape) > 1:
        return preds[-1]
    else:
        return preds


device = torch.device('cpu')

model = torch.load('./model_epoch_49.pt', map_location=device)

model.eval()
FIRST_SENTENCE = "mes chers compatriotes nous abordons en ce moment "
PARA = FIRST_SENTENCE
sent = FIRST_SENTENCE
length_para = 1000

for c in range(length_para):
    pred = predict_single_input(model, sent)
    vect = sample(pred, do_sample =True)
    carac = INDEX_CHAR[int(torch.argmax(vect))]

    PARA+=carac
    sent= PARA[-SEQ_LEN:]

print(PARA)

