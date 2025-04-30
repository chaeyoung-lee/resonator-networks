import numpy as np
import os
import json
import pickle
from PIL import Image, ImageOps
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_mnist_images(num_samples=5000):
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root='./data', train=True,
                           download=True, transform=transform)
    loader = DataLoader(mnist, batch_size=num_samples, shuffle=True)
    images, labels = next(iter(loader))
    return images.numpy(), labels.numpy()


def colorize_digit(img_array, rgb_color, contrast=2.5, brightness=1.5):
    """
    Colorizes a grayscale MNIST digit for a black background using alpha blending.
    The digit is brightened, contrasted, and painted with the given color.
    """
    from PIL import ImageEnhance

    # Convert to uint8 grayscale image
    img = Image.fromarray((img_array * 255).astype(np.uint8)).convert("L")

    # Enhance contrast and brightness
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Brightness(img).enhance(brightness)

    # Use the digit image as alpha
    color_layer = Image.new("RGBA", img.size, rgb_color + (0,))
    alpha_mask = img.point(lambda p: min(p, 255))  # keep in 0–255 range
    color_layer.putalpha(alpha_mask)

    return color_layer


def generate_synthetic_overlay_dataset(root_dir='synthetic_dataset', num_samples=1000, dim=1000, num_obj=1):
    os.makedirs(os.path.join(root_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'vectors'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'labels'), exist_ok=True)

    mnist_images, mnist_labels = load_mnist_images(num_samples * 3)

    color_map = {
        'cyan': (0, 255, 255),
        'pink': (255, 105, 180),
        'red': (255, 0, 0),
        'green': (0, 128, 0)
    }
    positions = {
        'top': 5, 'middle': 18, 'bottom': 35,
        'left': 5, 'center': 18, 'right': 35
    }

    factors = {
        'Color': list(color_map.keys()),
        'Digit': [str(i) for i in range(10)],
        'VLoc': ['top', 'middle', 'bottom'],
        'HLoc': ['left', 'center', 'right']
    }

    # Generate codebooks
    codebooks = {}
    for factor, values in factors.items():
        codebooks[factor] = {
            val: (2 * np.random.randint(0, 2, dim) - 1).astype(np.int8)
            for val in values
        }

    for i in tqdm(range(1, num_samples+1)):
        canvas = Image.new('RGB', (64, 64), color='black')
        encoded = np.zeros(dim, dtype=np.int8)
        label_dict = {}

        # Define position grid
        vlocs = ['top', 'middle', 'bottom']
        hlocs = ['left', 'center', 'right']
        all_positions = [(v, h) for v in vlocs for h in hlocs]

        # Randomly choose x non-overlapping positions
        chosen_positions = np.random.choice(
            len(all_positions), size=num_obj, replace=False)
        digit_positions = [all_positions[i] for i in chosen_positions]

        for j in range(num_obj):  # place 3 digits
            idx = (i-1)*num_obj + j
            digit = str(mnist_labels[idx])
            digit_img = mnist_images[idx][0]  # shape (28,28)

            color = np.random.choice(list(color_map.keys()))
            vloc, hloc = digit_positions[j]

            # Colorize
            colorized_digit = colorize_digit(digit_img, color_map[color])
            colorized_digit = colorized_digit.resize(
                (20, 20), resample=Image.BILINEAR)

            # Position
            x = positions[hloc]
            y = positions[vloc]
            # canvas.paste(colorized_digit, (x, y), colorized_digit.convert("L"))
            canvas.paste(colorized_digit, (x, y),
                         colorized_digit.split()[-1])  # use alpha channel

            # Update composite vector
            composite = np.ones(dim, dtype=np.int8)
            composite *= codebooks['Digit'][digit]
            composite *= codebooks['Color'][color]
            composite *= codebooks['VLoc'][vloc]
            composite *= codebooks['HLoc'][hloc]
            encoded += composite

            label_dict[f'obj{j+1}'] = {
                'Color': color,
                'Digit': digit,
                'VLoc': vloc,
                'HLoc': hloc
            }

        # Save files
        canvas.save(os.path.join(root_dir, 'images', f'{i}.jpg'))
        encoded = np.sign(encoded)
        np.save(os.path.join(root_dir, 'vectors', f'{i}.npy'), encoded)
        with open(os.path.join(root_dir, 'labels', f'{i}.json'), 'w') as f:
            json.dump(label_dict, f, indent=2)

    # Save codebooks
    with open(os.path.join(root_dir, 'codebook.pkl'), 'wb') as f:
        pickle.dump(codebooks, f)

    print(
        f'Dataset with {num_samples} multi-digit images saved to {root_dir}/')
    return codebooks


if __name__ == "__main__":
    # Generate synthetic dataset
    codebooks = generate_synthetic_overlay_dataset(
        root_dir='synthetic_dataset_1', num_samples=10000, dim=10000, num_obj=1)
