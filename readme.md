# ShapeNetCore.v2 Preparation
Along with [MeshToPointcloudFPS](https://github.com/salehjg/MeshToPointcloudFPS), this script is used to convert mesh files of the type `*.obj` to point clouds stored in HDF5 format. The script will also handle splitting the dataset into train-val-test with 60-20-20 % ratio.

# Data Conversion
The `MeshToPointcloudFPS` executable is responsible for data conversion from meshes `*.obj` to point clouds `*.h5` along with down-sampling the point clouds to the target number of points (`-n 1024`).  

# Pre-requirements
1. Build [MeshToPointcloudFPS](https://github.com/salehjg/MeshToPointcloudFPS) and copy the compiled executable beside the python script in this repository as `FpsCpu`.
2. Install python3 dependencies: `tqdm, numpy, joblib` 
3. Download `ShapeNetCore.v2.zip` from the official website ~ 26.4 GB
4. Unzip it.
5. In the script: 
   - Set `DATASET_PATH` to the unzipped dataset directory.
   - Set `OUTPUT_PATH` to an empty folder of your choice to store the processed data.
   - Set `TAXONOMY_PATH` to the absolute path of the `taxonomy.json` at the unzipped dataset directory.
6. Set `n_jobs` to the number of the CPU cores that you have on your system.
7. Run the script.
8. Check the results at `OUTPUT_PATH` (`train6-2-2.h5`, `val6-2-2.h5`, and `test6-2-2.h5`).

# Data Types
The `dtype` for the dataset is `np.float32` and `np.int32` for the labels.

# Data Class Names
* `labels.id.txt` at `OUTPUT_PATH` lists the `synid` of the classes in the dataset.
* `labels.names.txt` at `OUTPUT_PATH` lists the resolved names of the `synid`s using `taxonomy.json`.
* `labels.codes.json` at `OUTPUT_PATH` holds the content of a python dictionary to convert class names (`str`) to class codes (`int32`).

# Log
```
Listing *.h5 files...
Found class folders:  55
** Processing  03636649  ( lamp )
	 Concatenated class shape:  (2318, 1024, 3)
** Processing  02691156  ( airplane )
	 Concatenated class shape:  (4045, 1024, 3)
** Processing  02747177  ( ashcan )
	 Concatenated class shape:  (343, 1024, 3)
** Processing  02773838  ( bag )
	 Concatenated class shape:  (83, 1024, 3)
** Processing  02801938  ( basket )
	 Concatenated class shape:  (113, 1024, 3)
** Processing  02808440  ( bathtub )
	 Concatenated class shape:  (856, 1024, 3)
** Processing  02818832  ( bed )
	 Concatenated class shape:  (233, 1024, 3)
** Processing  02828884  ( bench )
	 Concatenated class shape:  (1813, 1024, 3)
** Processing  02843684  ( birdhouse )
	 Concatenated class shape:  (73, 1024, 3)
** Processing  02871439  ( bookshelf )
	 Concatenated class shape:  (452, 1024, 3)
** Processing  02876657  ( bottle )
	 Concatenated class shape:  (498, 1024, 3)
** Processing  02880940  ( bowl )
	 Concatenated class shape:  (186, 1024, 3)
** Processing  02924116  ( bus )
	 Concatenated class shape:  (939, 1024, 3)
** Processing  02933112  ( cabinet )
	 Concatenated class shape:  (1571, 1024, 3)
** Processing  02942699  ( camera )
	 Concatenated class shape:  (113, 1024, 3)
** Processing  02946921  ( can )
	 Concatenated class shape:  (108, 1024, 3)
** Processing  02954340  ( cap )
	 Concatenated class shape:  (56, 1024, 3)
** Processing  02958343  ( car )
	 Concatenated class shape:  (3513, 1024, 3)
** Processing  02992529  ( cellular telephone )
	 Concatenated class shape:  (831, 1024, 3)
** Processing  03001627  ( chair )
	 Concatenated class shape:  (6778, 1024, 3)
** Processing  03046257  ( clock )
	 Concatenated class shape:  (651, 1024, 3)
** Processing  03085013  ( computer keyboard )
	 Concatenated class shape:  (65, 1024, 3)
** Processing  03207941  ( dishwasher )
	 Concatenated class shape:  (93, 1024, 3)
** Processing  03211117  ( display )
	 Concatenated class shape:  (1093, 1024, 3)
** Processing  03261776  ( earphone )
	 Concatenated class shape:  (73, 1024, 3)
** Processing  03325088  ( faucet )
	 Concatenated class shape:  (744, 1024, 3)
** Processing  03337140  ( file )
	 Concatenated class shape:  (298, 1024, 3)
** Processing  03467517  ( guitar )
	 Concatenated class shape:  (797, 1024, 3)
** Processing  03513137  ( helmet )
	 Concatenated class shape:  (162, 1024, 3)
** Processing  03593526  ( jar )
	 Concatenated class shape:  (596, 1024, 3)
** Processing  03624134  ( knife )
	 Concatenated class shape:  (424, 1024, 3)
** Processing  03642806  ( laptop )
	 Concatenated class shape:  (460, 1024, 3)
** Processing  03691459  ( loudspeaker )
	 Concatenated class shape:  (1597, 1024, 3)
** Processing  03710193  ( mailbox )
	 Concatenated class shape:  (94, 1024, 3)
** Processing  03759954  ( microphone )
	 Concatenated class shape:  (67, 1024, 3)
** Processing  03761084  ( microwave )
	 Concatenated class shape:  (152, 1024, 3)
** Processing  03790512  ( motorcycle )
	 Concatenated class shape:  (337, 1024, 3)
** Processing  03797390  ( mug )
	 Concatenated class shape:  (214, 1024, 3)
** Processing  03928116  ( piano )
	 Concatenated class shape:  (239, 1024, 3)
** Processing  03938244  ( pillow )
	 Concatenated class shape:  (96, 1024, 3)
** Processing  03948459  ( pistol )
	 Concatenated class shape:  (307, 1024, 3)
** Processing  03991062  ( pot )
	 Concatenated class shape:  (602, 1024, 3)
** Processing  04004475  ( printer )
	 Concatenated class shape:  (166, 1024, 3)
** Processing  04074963  ( remote control )
	 Concatenated class shape:  (66, 1024, 3)
** Processing  04090263  ( rifle )
	 Concatenated class shape:  (2373, 1024, 3)
** Processing  04099429  ( rocket )
	 Concatenated class shape:  (85, 1024, 3)
** Processing  04225987  ( skateboard )
	 Concatenated class shape:  (152, 1024, 3)
** Processing  04256520  ( sofa )
	 Concatenated class shape:  (3173, 1024, 3)
** Processing  04330267  ( stove )
	 Concatenated class shape:  (218, 1024, 3)
** Processing  04379243  ( table )
	 Concatenated class shape:  (8436, 1024, 3)
** Processing  04401088  ( telephone )
	 Concatenated class shape:  (1089, 1024, 3)
** Processing  04460130  ( tower )
	 Concatenated class shape:  (133, 1024, 3)
** Processing  04468005  ( train )
	 Concatenated class shape:  (389, 1024, 3)
** Processing  04530566  ( vessel )
	 Concatenated class shape:  (1939, 1024, 3)
** Processing  04554684  ( washer )
	 Concatenated class shape:  (169, 1024, 3)


## FINAL REPORT:
	Train Set Data  :  (31535, 1024, 3)
	Train Set Labels:  (31535,)
	Validation Set Data  :  (10468, 1024, 3)
	Validation Set Labels:  (10468,)
	Test Set Data  :  (10468, 1024, 3)
	Test Set Labels:  (10468,)
```