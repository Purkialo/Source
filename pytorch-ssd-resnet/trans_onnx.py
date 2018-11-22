import torch
from vision.ssd.resnet_ssd import create_resnet_ssd, create_resnet_ssd_predictor
from vision.utils.misc import Timer
import argparse
import torch.onnx
from torch.autograd import Variable

parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument("--trained_model", type=str)
parser.add_argument("--label_file", type=str, help="The label file path.")
parser.add_argument("--nms_method", type=str, default="hard")
args = parser.parse_args()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    timer = Timer()
    class_names = [name.strip() for name in open(args.label_file).readlines()]
    net = create_resnet_ssd(len(class_names), is_test=True)
    timer.start("Load Model")
    net.load(args.trained_model)
    net = net.to(DEVICE)
    print(f'It took {timer.end("Load Model")} seconds to load the model.')
    net.eval()

    dummy_input = Variable(torch.randn(1, 3, 400, 400)).cuda()
    #confidence, locations = net(dummy_input)
    #print(confidence)
    #print(locations)
    with torch.no_grad():
        torch.onnx.export(net, dummy_input, "resssd.onnx", verbose=True)
    print("onnx done!")