import torch
import numpy as np
import h5py
from typing import Iterator, Tuple
from datasets.distributed import split_dataset_by_node

from torchtitan.tools.logging import logger
from torchtitan.config_manager import JobConfig
from torchtitan.components.dataloader import ParallelAwareDataloader



class LLM_Dataset:
    def __init__(self, data_path: str, context_length: int, device: str = "cpu", batch_size: int = 1):
        """
        Initializes the LLM_Dataset with the path to the HDF5 file and context length.

        Args:
            data_path (str): Path to the HDF5 file containing tokenized data.
            context_length (int): Length of each sequence.
        """
        self.data_path = data_path
        self.context_length = context_length
        self.device = device
        self.batch_size = batch_size
    
    def __len__(self) -> int:
        """
        Returns the number of batches in the dataset.

        Returns:
            int: Number of batches in the dataset.
        # """
        with h5py.File(self.data_path, 'r') as hdf5_file:
            dataset = hdf5_file['tokens']
            dataset_size = dataset.shape[0]
            n_examples = (dataset_size - 1) // self.context_length
            return n_examples

    def get_batch_iterator(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Creates an iterator for generating batches of data from an HDF5 file.

        Args:
            data_path (str): Path to the HDF5 file containing tokenized data.
            batch_size (int): Number of sequences in each batch.
            context_length (int): Length of each sequence.
            device (str, optional): Device to load the data onto ('cpu' or 'cuda'). Defaults to "cpu".

        Yields:
            tuple: A tuple containing input sequences (xb) and target sequences (yb).
        """
        data_path = self.data_path
        context_length = self.context_length
        device = self.device
        batch_size = self.batch_size

        # Open the HDF5 file in read mode
        with h5py.File(data_path, 'r') as hdf5_file:

            # Extract the dataset of tokenized sequences
            dataset = hdf5_file['tokens']

            # Get the total size of the dataset
            dataset_size = dataset.shape[0]

            # Calculate the number of examples (sequences) that can be made from the data
            n_examples = (dataset_size - 1) // context_length

            # Create an array of indices for examples and shuffle them for randomness
            example_idxs = np.arange(n_examples)
            np.random.shuffle(example_idxs)

            # Initialize epoch counter and example counter
            epochs = 0
            counter = 0

            while True:
                # Check if the current batch exceeds the number of available examples
                if counter + batch_size > n_examples:
                    # Shuffle the indices again and reset the counter to 0
                    np.random.shuffle(example_idxs)
                    counter = 0
                    epochs += 1  # Increment the epoch counter

                # Select a batch of random indices to generate sequences
                random_indices = example_idxs[counter:counter+batch_size] * context_length 

                # Retrieve sequences from the dataset based on the random indices
                random_samples = torch.tensor(np.array([dataset[idx:idx+context_length+1] for idx in random_indices]), dtype=torch.long)

                # Separate the input sequences (xb) and target sequences (yb)
                xb = random_samples[:, :context_length].to(device)  # Input sequence (first half of the random sample)
                yb = random_samples[:, 1:context_length+1].to(device)  # Target sequence (second half of the random sample)

                # Increment the counter to move to the next batch
                counter += batch_size

                # Yield the input and target sequences as a tuple for the current batch
                yield xb, yb


    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Allows the dataset to be iterable.

        Yields:
            tuple: A tuple containing input sequences (xb) and target sequences (yb).
        """
        return self.get_batch_iterator()

def build_llm_dataloader(
    dp_world_size: int,
    dp_rank: int,
    job_config: JobConfig,
    tokenizer: None = None,
    infinite: bool = True,

) -> ParallelAwareDataloader:
    
    dataset = LLM_Dataset(
        data_path=job_config.training.dataset,
        context_length=job_config.training.seq_len,
        device=dp_rank,
        batch_size=job_config.training.batch_size
    )
    return ParallelAwareDataloader(
        dataset=dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=job_config.training.batch_size,
    )

if __name__ == '__main__':
    # Example Usage (requires a dummy HDF5 file for testing)
    # Create a dummy HDF5 file
    import os
    dummy_data_path = "dummy_data.h5"
    if not os.path.exists(dummy_data_path):
        with h5py.File(dummy_data_path, 'w') as f:
            f.create_dataset('tokens', data=np.arange(1000))

    batch_size = 4
    context_length = 10
    for xb, yb in get_batch_iterator(dummy_data_path, batch_size, context_length):
        print("Input Batch Shape:", xb.shape)
        print("Target Batch Shape:", yb.shape)
        break