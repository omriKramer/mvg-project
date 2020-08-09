import torchvision.transforms as transforms

# Dataset configuration
dataset_path = r"C:\Users\Dror\Desktop\tmp_data"
imgs_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])  # Copy from https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198
