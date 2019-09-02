import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import string

import models.crnn as crnn


model_path = './data/crnn.pth'
img_path = './data/demo.png'
alphabet = '0123456789,.:(%$!^&-/);<~|`>?+=_[]{}"\'@#*ABCDEFGHIJKLMNOPQRSTUVWXYZ\ '
imgH = 32 # should be 32
nclass = len(alphabet) + 1
nhiddenstate = 256

model = crnn.CRNN(imgH, 1, nclass, nhiddenstate)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((200, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

model.eval()
preds = model(image)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-30s => %-30s' % (raw_pred, sim_pred))
