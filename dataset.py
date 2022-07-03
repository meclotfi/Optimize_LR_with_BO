import torchvision.datasets as datasets
from torchvision import transforms
import torch



def get_data_loaders(batch_size = 256):
    transform = transforms.Compose([transforms.ToTensor()]) 
    

    # download dataset
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    print(len(mnist_train), len(mnist_test))

    # Load dataset
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size,
        shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size,
        shuffle=True, num_workers=0)
    dataloaders = {'train': train_loader, 'val': test_loader}
    dataset_sizes= {'train': len(mnist_train), 'val': len(mnist_test)}
    return dataloaders,dataset_sizes