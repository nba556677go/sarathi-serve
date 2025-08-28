import unittest
import torch
from flashinfer import get_batch_indices_positions, get_seq_lens

class TestGetBatchIndicesPositions(unittest.TestCase):
    def test_get_batch_indices_positions(self):
        # Create test data
        batch_size = 2
        block_size = 16
        
        # Create append_indptr tensor
        append_indptr = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
        
        # Create kv_indptr tensor
        kv_indptr = torch.tensor([0, 2, 3], dtype=torch.int32, device="cuda")
        
        # Create kv_last_page_len tensor
        kv_last_page_len = torch.tensor([8, 4], dtype=torch.int32, device="cuda")
        
        # Get sequence lengths
        seq_lens = get_seq_lens(kv_indptr, kv_last_page_len, block_size)
        print(seq_lens)
        # Total number of tokens
        nnz = append_indptr[-1].item()
        
        # Call the function
        batch_indices, positions = get_batch_indices_positions(
            append_indptr, 
            seq_lens, 
            nnz
        )
        
        # If we reach here without error, the test passes
        self.assertEqual(batch_indices.size(0), nnz)
        self.assertEqual(positions.size(0), nnz)

if __name__ == "__main__":
    unittest.main()
