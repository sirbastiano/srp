"""
Standalone dataloader for the sarSSM model

The dataloader follows roughly these steps:
    1. One RC and one AC column get read in from the zarr array
    2. The data (originally [10000, 1]) columns get split into real and imaginary numbers in the same array
    so their size each becomes [10000, 2]
    3. The data then gets normalized to between 0 and 1 using some preset values which I chose through plotting histograms of the size of
    the I and Q values for the different data and hand picking an appropriate range.
    4. Finally, a position embedding column gets added to the rc array. This goes from [10000, 2] to [10000, 3]. This position embedding
    column is all the same number, which is the number of the column (i.e. how far down the range line it is) divided by 20000. The reasoning
    for doing it like this is that it means all the range embeddings fall between 0 and 1 for a scan with 20000 columns.
"""

# -- File info -- #
__author__ = 'Sebastian Fieldhouse'
__contact__ = 'sebastianfieldhouse.2@gmail.com'
__date__ = '2025-07-20'

from torch.utils.data import Dataset
import torch
import numpy as np
import zarr

class azimuthColumnDataset(Dataset):
    def __init__(self, samples: list[tuple[str, str]]):
        '''
        samples:
            list of tuples where the first string is the address of the rc sample and the second string is the address of the corresponding ac sample
        '''  
        # data min and max values - I got these from manually testing the data (for real/im values the max and min are around the same)
        
        self.rc_min = -3000
        self.rc_max = 3000
        
        self.gt_min = -12000
        self.gt_max = 12000 
                 
        
        # calculate length of dataset - putting it here because multiple uses
        self.length = 0
        self.samples = []
        
        print(f"Opening zarr files...", flush=True)
        for (rc_address, ac_address) in samples:       
            zarr_array = zarr.open(rc_address, mode='r')
            shape = zarr_array.shape
            chunks = zarr_array.chunks
            num_chunks = np.prod([int(np.ceil(s / c)) for s, c in zip(shape, chunks)])
            
            # increment the total num_chunks counter
            self.length += num_chunks
            
            # create a new tuple (rc_addr, ac_addr, num_of_columns)
            self.samples.append((rc_address, ac_address, num_chunks))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):       
        # search for which array the sample is in
        cumulative_length = 0
        for i, (rc_address, ac_address, num_chunks) in enumerate(self.samples):
            #print(f" for i = {i} cumulative_length = {cumulative_length} & num_chunks = {num_chunks}")
            if idx < cumulative_length + num_chunks:
                # the index is in this list
                item_index = idx - cumulative_length
                break
            cumulative_length += num_chunks
            
        rc_array = zarr.open(rc_address, mode='r')
        rc_chunk = rc_array[:, item_index]
        
        ac_array = zarr.open(ac_address, mode='r')
        ac_chunk = ac_array[:, item_index]
        
        inp = self._prepare_sample(rc_chunk, self.rc_min, self.rc_max)
        target = self._prepare_sample(ac_chunk, self.gt_min, self.gt_max)
        
        # add position embedding to the input
        position_embedding = torch.full((10000, 1), (item_index + 1) / 20000)
        inp = torch.cat((inp, position_embedding), dim=1)
        
        # send everything to float32 so it matches the datatype that the outputs will be in 
        return inp.to(torch.float32), target.to(torch.float32)
        
    def _prepare_sample(self, array, array_min, array_max):
        array = self._iq_split(array, array.shape)
        array = torch.tensor(array)
        array = self._normalize(array, array_min, array_max)
        return array

        
    def _normalize(self, tensor, tensor_min, tensor_max):
        # normalize the data to the range between 0 and 1
        # note that because data is normalized to this range the center of the data ends up at 0.5
        normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
        normalized_tensor = torch.clamp(normalized_tensor, min=tensor_min, max=tensor_max)         
            
        return normalized_tensor
        
        
    def _iq_split(self, array, array_shape):      
        # if the shape of the array is (10000,) then this tuple needs to be expanded for the code further down to work
        if len(array_shape) == 1:
            array_shape = array_shape + (1,)

        # calculate double the shape of the input
        combined_array_shape = (array_shape[0], array_shape[1]*2)        
        combined_array = np.empty(combined_array_shape)
        combined_array[:, 0] = array.real[:]
        combined_array[:, 1] = array.imag[:]
        return combined_array             

