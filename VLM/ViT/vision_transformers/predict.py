import os
import json
import argparse

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification


def main():
    parser = argparse.ArgumentParser(description='ViT Prediction')
    parser.add_argument('--image_dir', type=str, default='../tulip.jpg',
                        help='path to image file or directory containing images')
    parser.add_argument('--model_path', type=str, default='./weights/model-9.pth',
                        help='path to model weights')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='number of classes')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # check if input is a directory or single file
    if os.path.isdir(args.image_dir):
        # get all image files in directory
        img_paths = []
        for file in os.listdir(args.image_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_paths.append(os.path.join(args.image_dir, file))
        img_paths.sort()
    else:
        img_paths = [args.image_dir]

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model using Hugging Face ViTForImageClassification
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=args.num_classes,
        ignore_mismatched_sizes=True
    ).to(device)

    # load model weights
    assert os.path.exists(args.model_path), "weights file: '{}' not exist.".format(args.model_path)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # create output directory if not exists
    output_dir = './prediction_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # process each image
    with torch.no_grad():
        for img_path in img_paths:
            # load image
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path)
            plt.figure()
            plt.imshow(img)
            # [N, C, H, W]
            img_tensor = data_transform(img)
            # expand batch dimension
            img_tensor = torch.unsqueeze(img_tensor, dim=0)

            # predict class
            outputs = model(img_tensor.to(device))
            logits = outputs.logits
            predict = torch.softmax(logits, dim=1)
            predict_cla = torch.argmax(predict, dim=1).item()

            print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                         predict[0][predict_cla].item())
            plt.title(print_res)
            print(f"\nImage: {os.path.basename(img_path)}")
            for i in range(predict.shape[1]):
                print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                          predict[0][i].item()))

            # save figure
            output_filename = os.path.join(output_dir, os.path.basename(img_path))
            plt.savefig(output_filename, bbox_inches='tight', dpi=150)
            plt.close()

    print(f"\nPrediction results saved to: {output_dir}")


if __name__ == '__main__':
    main()
