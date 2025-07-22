import pytest
import torch
from flashinfer.page import get_batch_indices_positions

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for flashinfer kernels")
def test_get_batch_indices_positions_example_from_doc():
    """
    Test get_batch_indices_positions() using the example from the flashinfer documentation.
    """
    # Example data from the docstring
    nnz_kv = 10
    device = torch.device("cuda:0")
    append_indptr = torch.tensor([0, 1, 3, 6, 10], dtype=torch.int32, device=device)
    seq_lens     = torch.tensor([5, 5, 5, 5], dtype=torch.int32, device=device)

    # Call the function under test
    batch_indices, positions = get_batch_indices_positions(append_indptr, seq_lens, nnz_kv)
    print(batch_indices)
    # Expected outputs (from the documentation example)
    expected_batch_indices = torch.tensor(
        [0, 1, 1, 2, 2, 2, 3, 3, 3, 3],
        dtype=torch.int32, device=device
    )
    expected_positions = torch.tensor(
        [4, 3, 4, 2, 3, 4, 1, 2, 3, 4],
        dtype=torch.int32, device=device
    )

    # Assertions
    assert torch.equal(batch_indices, expected_batch_indices), (
        f"batch_indices mismatch: got {batch_indices}, expected {expected_batch_indices}"
    )
    assert torch.equal(positions, expected_positions), (
        f"positions mismatch: got {positions}, expected {expected_positions}"
    )
