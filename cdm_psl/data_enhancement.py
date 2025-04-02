import numpy as np


def random_perturbation(data, perturb_scale=0.05):
    perturb = (2 * np.random.rand(*data.shape) - 1) * perturb_scale
    return data + perturb


def interpolation(data, num_samples):
    interpolated_samples = []
    n = data.shape[0]
    for _ in range(num_samples):
        idx1, idx2 = np.random.choice(n, 2, replace=False)
        alpha = np.random.rand()
        interpolated_sample = alpha * data[idx1] + (1 - alpha) * data[idx2]
        interpolated_samples.append(interpolated_sample)
    return np.array(interpolated_samples)


def gaussian_noise(data, noise_scale=0.05):
    noise = np.random.normal(0, noise_scale, data.shape)
    return data + noise


def data_enhancement(offspringA, augmentation_factor=2):
    original_size = offspringA.shape[0]
    augmented_samples_num = original_size * (
        augmentation_factor - 1
    )

    perturbed_samples = random_perturbation(offspringA)
    interpolated_samples = interpolation(offspringA, augmented_samples_num // 3)
    noisy_samples = gaussian_noise(offspringA)

    augmented_data = np.vstack([perturbed_samples, interpolated_samples, noisy_samples])

    np.random.shuffle(augmented_data)
    augmented_data = augmented_data[:augmented_samples_num, :]

    enhanced_data = np.vstack([offspringA, augmented_data])

    return enhanced_data
