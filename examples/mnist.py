# Re-defining the fixed implementation after kernel reset

import os
import json
import pickle
import numpy as np
from time import time

import _set_the_path
from dynamics import rn_numpy
from utils.encoding_decoding import cosine_sim, best_guess


def run_scene_with_deflation(composite_vec, codebooks, factors, num_objects, max_iters=200):
    decoded_objects = []
    residual_vec = composite_vec.copy()
    time_total = 0

    for i in range(num_objects):
        start_time = time()
        factor_states, _, _ = rn_numpy.run(
            residual_vec,
            codebooks,
            synapse_type='OP',
            state_init=None,
            max_num_iters=max_iters,
            lim_cycle_detection_len=0,
            live_plotting_obj=None,
            silent=True
        )
        time_taken = time() - start_time
        time_total += time_taken

        pred_indices = best_guess(factor_states, codebooks)
        decoded_objects.append(pred_indices)

        pred_vec = np.ones_like(residual_vec)
        for factor in pred_indices:
            idx = pred_indices[factor]
            pred_vec *= codebooks[factor][:, idx]

        residual_vec -= pred_vec
        residual_vec = np.sign(residual_vec)

    return decoded_objects, time_total


def load_sample_and_run(sample_idx="1", dataset_dir="synthetic_dataset", num_objects=3, printSample=True, encoder=None):
    with open(os.path.join(dataset_dir, "codebook.pkl"), "rb") as f:
        codebooks = pickle.load(f)

    factors = {factor_name: list(codebooks[factor_name].keys())
               for factor_name in codebooks}

    codebooks_np = {
        factor: np.column_stack([codebooks[factor][val]
                                for val in factors[factor]])
        for factor in factors
    }

    if encoder is None:
        vec = np.load(os.path.join(
            dataset_dir, "vectors", f"{sample_idx}.npy"))
    elif encoder == "nn":
        vec = inference("vsa_encoder.pth", dataset_dir, sample_idx)
        vec = np.sign(vec).astype(np.int8)
    elif encoder == "vsa":
        # vec = vsa_encoder(labels)
        pass

    with open(os.path.join(dataset_dir, "labels", f"{sample_idx}.json")) as f:
        labels = json.load(f)

    decoded_objects, time_total = run_scene_with_deflation(
        vec, codebooks_np, factors, num_objects=num_objects)

    inverse_factors = {factor: {i: val for i, val in enumerate(factors[factor])}
                       for factor in factors}
    if printSample:
        # Print decoded results
        print(f"\n=== Sample {sample_idx} Decoding ===")
        print("Decoded objects:")
        for i, obj in enumerate(decoded_objects):
            for factor, idx in obj.items():
                print(f"{inverse_factors[factor][idx]}", end=" ")
            print()

        print("\nLabels:")
        for i in range(num_objects):
            for factor, idx in labels[f"obj{i+1}"].items():
                print(f"{labels[f'obj{i+1}'][factor]}", end=" ")
            print()

    # Match decoded objects to ground truth (permutation-invariant)
    used_gt = list(labels.values())  # e.g. [obj1_dict, obj2_dict, obj3_dict]
    total = 0
    correct = 0

    for pred in decoded_objects:
        best_match = None
        best_score = -1

        for gt in used_gt:
            score = sum(
                inverse_factors[factor][pred[factor]] == gt[factor]
                for factor in pred
            )
            if score > best_score:
                best_match = gt
                best_score = score

        correct += best_score
        total += len(pred)
        used_gt.remove(best_match)

    accuracy = correct / total if total > 0 else 0.0

    return accuracy, time_total


def inference(encoder, dataset_dir, sample_idx):
    import torch
    from PIL import Image
    import torchvision.transforms as transforms
    from ff_network import VSAEncoder

    model = VSAEncoder()
    model.load_state_dict(torch.load(encoder))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    image_path = os.path.join(dataset_dir, "images", f"{sample_idx}.jpg")
    image = Image.open(image_path).convert("L")
    # Add batch dimension => shape [1, 1, 28, 28]
    image = transform(image).unsqueeze(0)

    # 3. Run inference
    with torch.no_grad():
        vsa_vector = model(image)  # shape [1, 1000]
        vsa_vector = vsa_vector.squeeze(0)  # shape [1000]

    return vsa_vector.numpy()


def test_mnist(num_samples, root_dir, num_obj, encoder=None):
    accs = []
    times = []
    for i in range(1, num_samples + 1):
        accuracy, time_total = load_sample_and_run(
            str(i), dataset_dir=root_dir, num_objects=num_obj, printSample=False, encoder=encoder)
        accs.append(accuracy)
        times.append(time_total)

    print(
        f"Average accuracy over {num_samples} samples: {np.mean(accs)*100:.2f}")
    print(
        f"Average time over {num_samples} samples: {np.mean(times)*1000:.2f} ms")


# You would call this function with a valid sample index and dataset directory
if __name__ == "__main__":
    test_mnist(1000, "synthetic_dataset_1", 1, encoder=None)
