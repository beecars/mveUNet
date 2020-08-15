import torch
from utils import readBinImage

def predict(model, img_fname):
    
    model.eval()

    img = readBinImage(img_fname, as_tensor=True)
    img = img.unsqueeze(0).unsqueeze(0).float()
    img = img.to('cuda')
    prediction = model(img)
    
    return prediction
###