from dataloader import SARDataloader, SARTransform, get_sar_dataloader, SampleFilter
from utils import GT_MAX, GT_MIN, RC_MAX, RC_MIN
import zarr
import numpy as np

filters = SampleFilter(years=[2023], polarizations=["hh"], stripmap_modes=[1, 2, 3], parts=["PT1", "PT3"])


def test_concat_axes_vertical_multirow_visualization(transforms: SARTransform):
    patch_size = (1000, 100)
    buffer = (1000, 1000)
    stride = (1000, 100)
    batch_size = 16
    max_base_sample_size = (10000, 1000)
    dataloader = get_sar_dataloader(
        data_dir="/Data_large/marine/PythonProjects/SAR/sarpyx/data",
        level_from="rcmc",
        level_to="az",
        batch_size=batch_size,
        num_workers=0,
        patch_mode="rectangular", 
        patch_size = patch_size,#(1000, 100),
        buffer = buffer, # (1000, 1000),
        stride = stride, #(1000, 100),
        max_base_sample_size= max_base_sample_size, #(10000, 1000),
        #transform=transforms,
        shuffle_files = False,
        patch_order="col", 
        complex_valued = True,
        save_samples = False, 
        backend="zarr", 
        verbose=True, 
        samples_per_prod = 100,
        cache_size = 1000, 
        online = False, 
        max_products=1, 
        concatenate_patches = True, 
        concat_axis=0, 
        positional_encoding=True,
        filters=filters
    )
    from pathlib import Path
    
    start_x, start_y = buffer
    end_x, end_y = buffer[0] + max_base_sample_size[0], buffer[1] + patch_size[1]
    file = dataloader.dataset._files["full_name"].loc[0]

    for i, (x_batch, y_batch) in enumerate(dataloader):
        for sample_idx in range(len(x_batch)):
            sample_from, sample_to = x_batch[sample_idx], y_batch[sample_idx]
            print(f"Sample of index {sample_idx}")
            #print(dataloader.dataset._samples_by_file[Path("/Data_large/marine/PythonProjects/SAR/sarpyx/data/s1a-s1-raw-s-hh-20230508t121142-20230508t121213-048442-05d3c0.zarr")])
            #sample_from, sample_to = x_batch[0], y_batch[0]
            # sample_from, sample_to = dataloader.dataset[("/Data_large/marine/PythonProjects/SAR/sarpyx/data/s1a-s1-raw-s-hh-20230508t121142-20230508t121213-048442-05d3c0.zarr", 1000, 1000)]
            restored_column = dataloader.dataset.get_patch_visualization(patch=sample_from, level=dataloader.dataset.level_from, restore_complex=True, prepare_for_plotting=False)#.flatten()
            actual_column = zarr.open(file, mode='r')[dataloader.dataset.level_from]
            #print(f"Original column shape: {actual_column.shape}")
            if i != 0 or sample_idx != 0:
                start_x = start_x + max_base_sample_size[0]
                if start_x + max_base_sample_size[0] > (actual_column.shape[0] - 2*buffer[0]):
                    start_x = buffer[0]
                    start_y = start_y + patch_size[1]
                    end_y = start_y + patch_size[1]
                
                end_x = start_x + max_base_sample_size[0]
            # start_x = buffer[0] + (max_base_sample_size[0] * (i * batch_size + sample_idx)) % (actual_column.shape[0] - 2*buffer[0] + 1)
            # end_x = buffer[0] + (max_base_sample_size[0] * (i * batch_size + sample_idx + 1)) % (actual_column.shape[0] - 2*buffer[0] + 1)
            # start_y = buffer[1] + (max_base_sample_size[0] * (i * batch_size + sample_idx)) // (actual_column.shape[0] - 2*buffer[0] + 1) #(stride[1] * (i * batch_size + sample_idx)) % (actual_column.shape[1] - 2*buffer[1] + 1)
            # end_y = buffer[1] + (max_base_sample_size[0] * (i * batch_size + sample_idx)) // (actual_column.shape[0] - 2*buffer[0] + 1) + patch_size[1] #(stride[1] * (i * batch_size + sample_idx + 1)) % (actual_column.shape[1] - 2*buffer[1] + 1)
            print(f"Sampling actual patch from ({start_x},{start_y}) to ({end_x},{end_y})")
            actual_column = actual_column[start_x:end_x, start_y:end_y]
            
            #print(f"Actual column: {actual_column} vs restored column shape: {restored_column}")
            #print(f"Original column: {actual_column}")
            #print(f"Concatenatactual_column[ 1000: -1000, 1000 + i*100: 1000 + (i+1)*100]ed patch from: {sample_from}")
            #print(f"Restored sample from: {restored_column}")
            #print(f"Original column shape: {actual_column.shape}, Restored column shape: {restored_column.shape}")
            max_index = 5000# min(restored_column.shape[0], actual_column.shape[0])
            # actual_column_shape = actual_column.shape[0] - actual_column.shape[0]%300
            #assert restored_column.shape[0] == (actual_column_shape), f"Shape mismatch at level {dataloader.dataset.level_from} at batch {i}: {restored_column.shape} vs {actual_column_shape}"
            #print(restored_column[:max_index-1])
            #print(actual_column[:max_index-1])
            assert np.allclose(restored_column[:max_index-1, :], actual_column[:max_index-1, :], rtol=1e-10, atol=1e-10), f"Restored column is different than original column at level {dataloader.dataset.level_from}. Difference: {restored_column[:max_index-1] - actual_column[:max_index-1]}"
            print(f"✅ First test passed!!")
            max_index = 1000 #min(restored_column.shape[0], actual_column.shape[0])
            restored_column = dataloader.dataset.get_patch_visualization(patch=sample_to, level=dataloader.dataset.level_to, restore_complex=True, prepare_for_plotting=False)#.flatten()
            actual_column = zarr.open(file, mode='r')[dataloader.dataset.level_to]
            #print(f"Original column shape: {actual_column.shape}")
            actual_column = actual_column[start_x:end_x, start_y:end_y]
            #assert restored_column.shape[0] == actual_column_shape, f"Shape mismatch at level {dataloader.dataset.level_to} at batch {i}: {restored_column.shape} vs {actual_column_shape}"
            assert np.allclose(restored_column[:max_index-1, :], actual_column[:max_index-1, :], rtol=1e-10, atol=1e-10), f"Restored column is different than original column at level {dataloader.dataset.level_to}. Difference: {restored_column[:max_index-1] - actual_column[:max_index-1]}"
            #    print(f"Batch {i}: x {x_batch.shape}, y {y_batch.shape}")
            print(f"✅ Second test passed!!")

def test_concat_axes_vertical_visualization(transforms: SARTransform):
    patch_size = (1000, 1)
    buffer = (1000, 1000)
    stride = (300, 1)
    batch_size = 16
    dataloader = get_sar_dataloader(
        data_dir="/Data_large/marine/PythonProjects/SAR/sarpyx/data",
        level_from="rcmc",
        level_to="az",
        batch_size=batch_size,
        num_workers=0,
        patch_mode="rectangular", 
        patch_size = patch_size,#(1000, 1),
        buffer = buffer,#(1000, 1000),
        stride = stride,
        #transform=transforms,
        shuffle_files = False,
        patch_order="row", 
        complex_valued = True,
        save_samples = False, 
        backend="zarr", 
        verbose=True, 
        samples_per_prod = 1000,
        cache_size = 100, 
        online = True, 
        max_products=1, 
        concatenate_patches = True, #True, 
        concat_axis=0, 
        positional_encoding= False, 
        filters=filters
    )
    file = dataloader.dataset._files["full_name"].loc[0]

    for i, (x_batch, y_batch) in enumerate(dataloader):
        for sample_idx, (sample_from, sample_to) in enumerate(zip(x_batch, y_batch)):
            #sample_from, sample_to = x_batch[0], y_batch[0]
            # sample_from, sample_to = dataloader.dataset[("/Data_large/marine/PythonProjects/SAR/sarpyx/data/s1a-s1-raw-s-hh-20230508t121142-20230508t121213-048442-05d3c0.zarr", 1000, 1000)]
            restored_column = dataloader.dataset.get_patch_visualization(patch=sample_from, zfile=file, level=dataloader.dataset.level_from, restore_complex=True, prepare_for_plotting=False).flatten()
            actual_column = zarr.open(file, mode='r')[dataloader.dataset.level_from]
            #print(f"Original column shape: {actual_column.shape}")
            actual_column = actual_column[1000:-1000, 1000+ i*batch_size + sample_idx]
            #print(f"Original column: {actual_column}")
            #print(f"Concatenated patch from: {sample_from}")
            #print(f"Restored sample from: {restored_column}")
            #print(f"Original column shape: {actual_column.shape}, Restored column shape: {restored_column.shape}")
            max_index = 1000# min(restored_column.shape[0], actual_column.shape[0])
            actual_column_shape = actual_column.shape[0] - actual_column.shape[0]%300
            #assert restored_column.shape[0] == (actual_column_shape), f"Shape mismatch at level {dataloader.dataset.level_from} at batch {i}: {restored_column.shape} vs {actual_column_shape}"
            #print(restored_column[:max_index-1])
            #print(actual_column[:max_index-1])
            assert np.allclose(restored_column[:max_index-1], actual_column[:max_index-1], rtol=1e-10, atol=1e-10), f"Restored column is different than original column at level {dataloader.dataset.level_from}. Difference: {restored_column[:max_index-1] - actual_column[:max_index-1]}"
            print(f"✅ First test passed!!")
            max_index = 1000 #min(restored_column.shape[0], actual_column.shape[0])
            restored_column = dataloader.dataset.get_patch_visualization(patch=sample_to, level=dataloader.dataset.level_to, restore_complex=True, prepare_for_plotting=False).flatten()
            actual_column = zarr.open(file, mode='r')[dataloader.dataset.level_to]
            #print(f"Original column shape: {actual_column.shape}")
            actual_column = actual_column[1000:-1000, 1000 + i*batch_size + sample_idx]
            #assert restored_column.shape[0] == actual_column_shape, f"Shape mismatch at level {dataloader.dataset.level_to} at batch {i}: {restored_column.shape} vs {actual_column_shape}"
            assert np.allclose(restored_column[:max_index-1], actual_column[:max_index-1], rtol=1e-10, atol=1e-10), f"Restored column is different than original column at level {dataloader.dataset.level_to}. Difference: {restored_column[:max_index-1] - actual_column[:max_index-1]}"
            #    print(f"Batch {i}: x {x_batch.shape}, y {y_batch.shape}")
            print(f"✅ Second test passed!!")
        
def test_original_axes_horizontal(transforms: SARTransform):
    patch_size = (1, -1)
    buffer = (0, 0)
    stride = (1, 300)
    dataloader = get_sar_dataloader(
        data_dir="/Data_large/marine/PythonProjects/SAR/sarpyx/data",
        level_from="rcmc",
        level_to="az",
        batch_size=16,
        num_workers=0,
        patch_mode="rectangular", 
        patch_size = patch_size,#(1000, 1),
        buffer = buffer,#(1000, 1000),
        stride = stride,
        transform=transforms,
        shuffle_files = False,
        patch_order="row", 
        complex_valued = True,
        save_samples = False, 
        backend="zarr", 
        verbose=True, 
        samples_per_prod = 1000,
        cache_size = 100, 
        online = True, 
        max_products=1, 
        concatenate_patches = False, #True, 
        concat_axis=0, 
        positional_encoding= True, 
        filters=filters
    )
    import numpy as np
    file = dataloader.dataset._files["full_name"].loc[0]
    # ===== FIRST TEST: level_from =====
    for i in range (3):

        sample_from, sample_to = dataloader.dataset[(file, i, 0)]
        
        restored_column = dataloader.dataset.get_patch_visualization(
            patch=sample_from, 
            level=dataloader.dataset.level_from, 
            restore_complex=True, 
            prepare_for_plotting=False
        ).squeeze(0)
        restored_column = restored_column.flatten()

        actual_column_from = zarr.open(file, mode='r')[dataloader.dataset.level_from]
        actual_column_from = actual_column_from[i, :]
        
        print(f"FIRST TEST - Original column shape: {actual_column_from.shape}, Restored column shape: {restored_column.shape}")
        
        max_index = 100
        actual_column_shape_from = actual_column_from.shape[0]  # Renamed for clarity
        
        assert restored_column.shape[0] == actual_column_shape_from, f"Shape mismatch at level {dataloader.dataset.level_from}: {restored_column.shape} vs {actual_column_shape_from}"
        assert np.allclose(restored_column[:max_index-1], actual_column_from[:max_index-1], rtol=1e-10, atol=1e-10), f"Restored column is different than original column at level {dataloader.dataset.level_from}. Difference: {restored_column[:max_index-1] - actual_column_from[:max_index-1]}"
        
        print(f"✅ First test passed!!")
        
        # ===== SECOND TEST: level_to =====
        actual_column_to = zarr.open(file, mode='r')[dataloader.dataset.level_to]
        actual_column_to = actual_column_to[i, :]
        
        restored_column_to = dataloader.dataset.get_patch_visualization(
            patch=sample_to, 
            level=dataloader.dataset.level_to, 
            restore_complex=True, 
            prepare_for_plotting=False
        ).squeeze(0)
        restored_column_to = restored_column_to.flatten()

        print(f"SECOND TEST - Original column shape: {actual_column_to.shape}, Restored column shape: {restored_column_to.shape}")

        max_index = min(restored_column_to.shape[0], actual_column_to.shape[0])
        actual_column_shape_to = actual_column_to.shape[0]  # NEW: Calculate for level_to
        
        # FIX: Use actual_column_shape_to instead of actual_column_shape_from
        assert restored_column_to.shape[0] == actual_column_shape_to, f"Shape mismatch at level {dataloader.dataset.level_to}: {restored_column_to.shape} vs {actual_column_shape_to}"
        assert np.allclose(restored_column_to[:max_index-1], actual_column_to[:max_index-1], rtol=1e-10, atol=1e-10), f"Restored column is different than original column at level {dataloader.dataset.level_to}. Difference: {restored_column_to[:max_index-1] - actual_column_to[:max_index-1]}"

        print(f"✅ Second test passed!! Both levels validated successfully.")
def test_original_axes_vertical(transforms: SARTransform):
    patch_size = (-1, 1)
    buffer = (0, 0)
    stride = (300, 1)
    dataloader = get_sar_dataloader(
        data_dir="/Data_large/marine/PythonProjects/SAR/sarpyx/data",
        level_from="rcmc",
        level_to="az",
        batch_size=16,
        num_workers=0,
        patch_mode="rectangular", 
        patch_size = patch_size,#(1000, 1),
        buffer = buffer,#(1000, 1000),
        stride = stride,
        transform=transforms,
        shuffle_files = False,
        patch_order="row", 
        complex_valued = True,
        save_samples = False, 
        backend="zarr", 
        verbose=True, 
        samples_per_prod = 1000,
        cache_size = 100, 
        online = True, 
        max_products=1, 
        concatenate_patches = False, #True, 
        concat_axis=0, 
        positional_encoding= True, 
        filters=filters
    )
    import numpy as np
    file = dataloader.dataset._files["full_name"].loc[0]

    # ===== FIRST TEST: level_from =====
    for i in range (3):
        sample_from, sample_to = dataloader.dataset[(file, 0, i)]
        
        restored_column = dataloader.dataset.get_patch_visualization(
            patch=sample_from, 
            level=dataloader.dataset.level_from, 
            restore_complex=True, 
            prepare_for_plotting=False
        ).squeeze(1)
        restored_column = restored_column.flatten()
        
        actual_column_from = zarr.open(file, mode='r')[dataloader.dataset.level_from]
        actual_column_from = actual_column_from[:, i]
        
        print(f"FIRST TEST - Original column shape: {actual_column_from.shape}, Restored column shape: {restored_column.shape}")
        
        max_index = 100
        actual_column_shape_from = actual_column_from.shape[0]  # Renamed for clarity
        
        assert restored_column.shape[0] == actual_column_shape_from, f"Shape mismatch at level {dataloader.dataset.level_from}: {restored_column.shape} vs {actual_column_shape_from}"
        assert np.allclose(restored_column[:max_index-1], actual_column_from[:max_index-1], rtol=1e-10, atol=1e-10), f"Restored column is different than original column at level {dataloader.dataset.level_from}. Difference: {restored_column[:max_index-1] - actual_column_from[:max_index-1]}"
        
        print(f"✅ First test passed!!")
        
        # ===== SECOND TEST: level_to =====
        actual_column_to = zarr.open(file, mode='r')[dataloader.dataset.level_to]
        actual_column_to = actual_column_to[:, i]
        
        restored_column_to = dataloader.dataset.get_patch_visualization(
            patch=sample_to, 
            level=dataloader.dataset.level_to, 
            restore_complex=True, 
            prepare_for_plotting=False
        ).squeeze(1)
        restored_column_to = restored_column_to.flatten()

        print(f"SECOND TEST - Original column shape: {actual_column_to.shape}, Restored column shape: {restored_column_to.shape}")

        max_index = min(restored_column_to.shape[0], actual_column_to.shape[0])
        actual_column_shape_to = actual_column_to.shape[0]  # NEW: Calculate for level_to
        
        # FIX: Use actual_column_shape_to instead of actual_column_shape_from
        assert restored_column_to.shape[0] == actual_column_shape_to, f"Shape mismatch at level {dataloader.dataset.level_to}: {restored_column_to.shape} vs {actual_column_shape_to}"
        assert np.allclose(restored_column_to[:max_index-1], actual_column_to[:max_index-1], rtol=1e-10, atol=1e-10), f"Restored column is different than original column at level {dataloader.dataset.level_to}. Difference: {restored_column_to[:max_index-1] - actual_column_to[:max_index-1]}"

        print(f"✅ Second test passed!! Both levels validated successfully.")
if __name__ == "__main__":
    transforms = SARTransform.create_minmax_normalized_transform(
        normalize=True,
        rc_min=RC_MIN,
        rc_max=RC_MAX,
        gt_min=GT_MIN,
        gt_max=GT_MAX,
        complex_valued=True
    )
    
    #np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    test_original_axes_horizontal(transforms)
    test_original_axes_vertical(transforms)
    test_concat_axes_vertical_visualization(transforms)
    test_concat_axes_vertical_multirow_visualization(transforms)
    # def debug_dataset_processing(dataloader: SARDataloader):
    #     """Debug what transformations are being applied."""
    #     zfile = "/Data_large/marine/PythonProjects/SAR/sarpyx/data/s1a-s1-raw-s-hh-20230508t121142-20230508t121213-048442-05d3c0.zarr"
        
    #     print("=== DEBUGGING DATASET PROCESSING ===")
        
    #     # 1. Check dataset configuration
    #     print(f"Dataset config:")
    #     print(f"  - concatenate_patches: {dataloader.dataset.concatenate_patches}")
    #     print(f"  - positional_encoding: {dataloader.dataset.positional_encoding}")
    #     print(f"  - transform: {dataloader.dataset.transform}")
    #     print(f"  - complex_valued: {dataloader.dataset.complex_valued}")
    #     print(f"  - patch_size: {dataloader.dataset._patch_size}")
        
    #     # 2. Get raw data
    #     raw_data = zarr.open(zfile, mode='r')[dataloader.dataset.level_from][:, 0]
    #     print(f"\nRaw zarr data shape: {raw_data.shape}, dtype: {raw_data.dtype}")
    #     print(f"Raw data sample: {raw_data[:5]}")
        
    #     # 3. Get processed sample
    #     sample_from, sample_to = dataloader.dataset._get_base_sample(Path(zfile), 0, 0)
    #     sample_from = sample_from.flatten()
    #     print(f"\nProcessed sample shape: {sample_from.shape}, dtype: {sample_from.dtype}")
    #     print(f"Processed sample: {sample_from[:5]}") #if sample_from.ndim == 1 else sample_from.flatten()[:5]}")
        
    #     # 4. Check if they match
    #     if sample_from.shape[0] == raw_data.shape[0]:
    #         if not (sample_from == raw_data).all():
    #             diff = sample_from - raw_data
    #             print(f"\nDifference stats:")
    #             print(f"  - Max difference: {np.max(np.abs(diff))}")
    #             print(f"  - Mean difference: {np.mean(np.abs(diff))}")
    #             print(f"  - Are they close? {np.allclose(sample_from, raw_data, rtol=1e-10, atol=1e-10)}")
    #         else:
    #             print("Products are the same!")
    #     else:
    #         print(f"\nShape mismatch - cannot compare directly")

    # # Run the debug
    # debug_dataset_processing(loader)