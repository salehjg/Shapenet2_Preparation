# ShapeNetCore.v2 Preparation
Along with [MeshToPointcloudFPS](https://github.com/salehjg/MeshToPointcloudFPS), this script is used to convert mesh files of the type `*.obj` to point clouds stored in HDF5 format. The script will also handle splitting the dataset into train-val-test with 60-20-20 % ratio.

# Data Conversion
The `MeshToPointcloudFPS` executable is responsible for data conversion from meshes `*.obj` to point clouds `*.h5` while down-sampling the point clouds to the target number of points (`-n 1024`).  

# Pre-requirements
1. Build [MeshToPointcloudFPS](https://github.com/salehjg/MeshToPointcloudFPS) and copy the compiled executable beside the python script in this repository as `FpsCpu`.
2. Install python3 dependencies: `tqdm, numpy, joblib` 
3. Download `ShapeNetCore.v2.zip` from the official website ~ 26.4 GB
4. Unzip it.
5. In the script: 
   - Set `DATASET_PATH` to the unzipped dataset directory.
   - Set `OUTPUT_PATH` to an empty folder of your choice to store the processed data.
   - Set `TAXONOMY_PATH` to the absolute path of the `taxonomy.json` in the unzipped dataset directory.
6. Set `n_jobs` to the number of the CPU cores that you have on your system.
7. Run the script.
8. Check the results at `OUTPUT_PATH` (`train6-2-2.h5`, `val6-2-2.h5`, and `test6-2-2.h5`).

# Data Types
The `dtype` for the dataset is `np.float32` and `np.int32` for the labels.

# Data Class Names
* `labels.id.txt` at `OUTPUT_PATH` lists the `synid` of the classes in the dataset.
* `labels.names.txt` at `OUTPUT_PATH` lists the resolved names of the `synid`s using `taxonomy.json`.
* `labels.codes.json` at `OUTPUT_PATH` holds the content of a python dictionary to convert class names (`str`) to class codes (`int32`).