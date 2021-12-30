from torchvision import transforms


def create_transform(width, height):
    if width < 224 or height < 224:
        raise Exception('Width and height need to be at least 224')

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((width, height)),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])
