import argparse
import torch
from my_data import MyDataset, VOCAB
from my_models import MyModel0
from my_utils import pred_to_dict
import json


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-i", "--hidden-size", type=int, default=256)

    args = parser.parse_args()
    args.device = torch.device(args.device)

    model = MyModel0(len(VOCAB), 16, args.hidden_size).to(args.device)
    dataset = MyDataset(None, args.device, test_path="data/test_dict.pth")

    model.load_state_dict(torch.load("model.pth"))

    model.eval()
    with torch.no_grad():
        for key in dataset.test_dict.keys():
            text_tensor = dataset.get_test_data(key)

            oupt = model(text_tensor)
            prob = torch.nn.functional.softmax(oupt, dim=2)
            prob, pred = torch.max(prob, dim=2)

            prob = prob.squeeze().cpu().numpy()
            pred = pred.squeeze().cpu().numpy()

            real_text = dataset.test_dict[key]
            result = pred_to_dict(real_text, pred, prob)

            with open("results/" + key + ".json", "w", encoding="utf-8") as json_opened:
                json.dump(result, json_opened, indent=4)

            print(key)


if __name__ == "__main__":
    test()
