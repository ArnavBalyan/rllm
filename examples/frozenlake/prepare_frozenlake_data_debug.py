import numpy as np

from rllm.data.dataset import DatasetRegistry


def prepare_frozenlake_data(train_size=10000, test_size=100):
    """
    Prepare and register FrozenLake DEBUG datasets with identical observations.
    All examples use: seed=42, size=4, p=0.8
    """
    # Hardcoded debug values - all environments will be identical
    train_seeds = np.full(train_size, 42)
    test_seeds = np.full(test_size, 42)
    train_sizes = np.full(train_size, 4)
    test_sizes = np.full(test_size, 4)
    train_ps = np.full(train_size, 0.8)
    test_ps = np.full(test_size, 0.8)

    def frozenlake_process_fn(seed, size, p, idx):
        """Process function to create FrozenLake task instances."""
        return {"seed": seed, "size": size, "p": p, "index": idx, "uid": f"{seed}_{size}_{p}"}

    # Create train and test data
    train_data = [frozenlake_process_fn(seed, train_sizes[idx], train_ps[idx], idx) for idx, seed in enumerate(train_seeds)]
    test_data = [frozenlake_process_fn(seed, test_sizes[idx], test_ps[idx], idx) for idx, seed in enumerate(test_seeds)]

    # Register the datasets with the DatasetRegistry
    train_dataset = DatasetRegistry.register_dataset("frozenlake_debug", train_data, "train")
    test_dataset = DatasetRegistry.register_dataset("frozenlake_debug", test_data, "test")

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_frozenlake_data()
    print(f"DEBUG Train dataset: {len(train_dataset.get_data())} examples (all identical)")
    print(f"DEBUG Test dataset: {len(test_dataset.get_data())} examples (all identical)")
    print("Sample train example:", train_dataset.get_data()[0])
    print("Sample test example:", test_dataset.get_data()[0])
    print("\nDataset saved as 'frozenlake_debug'")

