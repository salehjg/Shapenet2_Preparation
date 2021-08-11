import os

import h5py
import json
import glob
import subprocess
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np

DATASET_PATH = '/run/media/saleh/Ext240SSD/ShapeNetCore.v2/'
OUTPUT_PATH = '/run/media/saleh/Ext240SSD/ShapeNetCore.v2.processed/'
TAXONOMY_PATH = '/run/media/saleh/Ext240SSD/ShapeNetCore.v2/taxonomy.json'


class DatasetParser:
    def __init__(self, path_dataset, path_output, path_taxonomy):
        self.path_dataset = path_dataset
        self.path_output = path_output
        self.path_taxonomy = path_taxonomy

        f = open(self.path_taxonomy, 'r')
        self.taxonomy = json.load(f)
        f.close()

    def get_name_from_synsetId(self, strSynsetId):
        matches = []
        for item in self.taxonomy:
            if item['synsetId'] == strSynsetId:
                matches.append(item)

        if len(matches) <= 1:
            names = matches[0]['name']
            names = names.split(',')
            return names[0]
        else:
            assert False

    def get_fname_all(self, base_dir, file_extension='*.obj'):
        obj = glob.iglob(base_dir + '**/' + file_extension, recursive=True)
        files = []
        for f in obj:
            files.append(f)
        return files

    def abspath_get_dir_with_slash(self, abs_path=''):
        return abs_path[0:abs_path.rfind('/')]

    def abspath_get_fname_with_ext(self, abs_path):
        return abs_path[abs_path.rfind('/'):]

    def abspath_get_fname_without_ext(self, abs_path):
        return abs_path[abs_path.rfind('/'):abs_path.rfind('.') - 1]

    def abspath_get_ext(self, abs_path):
        return abs_path[abs_path.rfind('.') + 1:]

    def convert_inplace_obj2hdf5_sampledFPS_all(self, n_jobs=1):
        def run_the_command_obj2hdf5(file_abspath):
            path_o_raw_pcd = self.abspath_get_dir_with_slash(file_abspath) + self.abspath_get_fname_without_ext(
                file_abspath) + '.raw.pcd'
            path_o_smpl_pcd = self.abspath_get_dir_with_slash(file_abspath) + self.abspath_get_fname_without_ext(
                file_abspath) + '.1024.pcd'
            path_o_smpl_h5 = self.abspath_get_dir_with_slash(file_abspath) + self.abspath_get_fname_without_ext(
                file_abspath) + '.1024.h5'
            cmd = './FpsCpu ' + \
                  '-i ' + file_abspath + ' ' + \
                  '-r ' + path_o_raw_pcd + ' ' + \
                  '-p ' + path_o_smpl_pcd + ' ' + \
                  '-o ' + path_o_smpl_h5 + ' ' + \
                  '-n 1024' + \
                  ' > /dev/null'

            results = subprocess.run(cmd, shell=True, stderr=subprocess.PIPE)
            stderr = results.stderr.decode('utf-8')
            if stderr != '':
                print('Error in obj2pcd conversion of ' + file_abspath)
                return False
            else:
                return True

        print('Listing files...')
        obj_files = self.get_fname_all(self.path_dataset)

        print('Converting *.obj files to *.h5 along with FPS down-sampling to 1024 points ...')
        if n_jobs == 1:
            print('** Running in NON-parallel fashion **')
            for file in tqdm(obj_files):
                run_the_command_obj2hdf5(file)
        else:
            print('** Running in parallel fashion **')
            Parallel(n_jobs=n_jobs)(delayed(run_the_command_obj2hdf5)(file) for file in tqdm(obj_files))

    def concatenate_hdf5_files_split622(self):
        def permute_data_label(dataset, label):
            result_data = np.copy(dataset)
            result_label = np.copy(label)
            indices = np.random.permutation(dataset.shape[0])
            np.take(dataset, indices, axis=0, out=result_data)
            np.take(label, indices, axis=0, out=result_label)
            return result_data, result_label

        print('Listing *.h5 files...')

        class_folder_names = [name for name in os.listdir(self.path_dataset) if
                              os.path.isdir(os.path.join(self.path_dataset, name))]
        class_names = [self.get_name_from_synsetId(synid) for synid in class_folder_names]
        class_codes = {}
        i = 0
        for c in class_names:
            class_codes[c] = i
            i += 1

        np.savetxt(self.path_output + 'labels.id.txt', np.array(class_folder_names), delimiter="\n", fmt="%s")
        np.savetxt(self.path_output + 'labels.names.txt', np.array(class_names), delimiter="\n", fmt="%s")
        with open(self.path_output + 'labels.codes.json', 'w') as fp:
            json.dump(class_codes, fp)

        print('Found class folders: ', len(class_folder_names))

        trainset_data = np.array([], dtype=np.float32).reshape([0, 1024, 3])  # 60%
        trainset_label = np.array([], dtype=np.int32).reshape([0])

        valset_data = np.array([], dtype=np.float32).reshape([0, 1024, 3])  # 20%
        valtset_label = np.array([], dtype=np.int32).reshape([0])

        testset_data = np.array([], dtype=np.float32).reshape([0, 1024, 3])  # 20%
        testset_label = np.array([], dtype=np.int32).reshape([0])

        for class_folder_name in class_folder_names:
            class_name = self.get_name_from_synsetId(class_folder_name)
            print('** Processing ', class_folder_name, ' (', class_name, ')')
            current_dir = self.path_dataset + class_folder_name + '/'
            h5_files_current_dir = self.get_fname_all(current_dir, '*.h5')

            data_current_concat = np.array([], dtype=np.float32)
            first = True
            for h5_file in h5_files_current_dir:
                f = h5py.File(h5_file, 'r')
                if not f.keys().__contains__('data'):
                    assert False
                data = np.array(f['data'])
                while len(data.shape) < 3:
                    data = np.expand_dims(data, 0)
                if first:
                    data_current_concat = data
                    first = False
                else:
                    data_current_concat = np.concatenate([data_current_concat, data], 0, dtype=np.float32)
                f.close()

            print('\t Concatenated class shape: ', data_current_concat.shape)

            concat_h5 = h5py.File(self.path_output + class_folder_name + '.' + class_name + '.h5', 'w')
            concat_h5.create_dataset('data', data=data_current_concat, dtype=np.float32)
            labels = np.full(data_current_concat.shape[0], class_codes[class_name], dtype=np.int32)
            concat_h5.create_dataset('label', data=labels)
            concat_h5.close()

            data_current_concat, labels = permute_data_label(data_current_concat, labels)

            # splitting train val test sets here
            n = data_current_concat.shape[0]
            splits_n = [
                n - 2 * np.floor(n / 5),
                n - 1 * np.floor(n / 5),
                n
            ]  # train val test index bonds
            splits_n = np.array(splits_n, dtype=np.int32)

            trainset_data = np.concatenate([trainset_data, data_current_concat[0:splits_n[0], :, :]], axis=0, dtype=np.float32)
            trainset_label = np.concatenate([trainset_label, labels[0:splits_n[0]]], axis=0, dtype=np.int32)

            valset_data = np.concatenate([valset_data, data_current_concat[splits_n[0]:splits_n[1], :, :]], axis=0, dtype=np.float32)
            valtset_label = np.concatenate([valtset_label, labels[splits_n[0]:splits_n[1]]], axis=0, dtype=np.int32)

            testset_data = np.concatenate([testset_data, data_current_concat[splits_n[1]:splits_n[2], :, :]], axis=0, dtype=np.float32)
            testset_label = np.concatenate([testset_label, labels[splits_n[1]:splits_n[2]]], axis=0, dtype=np.int32)

        print("\n\n## FINAL REPORT:")
        trainset_h5 = h5py.File(self.path_output + 'train6-2-2.h5', 'w')
        trainset_h5.create_dataset('data', data=trainset_data, dtype=np.float32)
        trainset_h5.create_dataset('label', data=trainset_label, dtype=np.int32)
        trainset_h5.close()
        np.save(self.path_output + 'train.data.npy', trainset_data)
        np.save(self.path_output + 'train.label.npy', trainset_label)
        print("\tTrain Set Data  : ", trainset_data.shape)
        print("\tTrain Set Labels: ", trainset_label.shape)

        valset_h5 = h5py.File(self.path_output + 'val6-2-2.h5', 'w')
        valset_h5.create_dataset('data', data=valset_data, dtype=np.float32)
        valset_h5.create_dataset('label', data=valtset_label, dtype=np.int32)
        valset_h5.close()
        np.save(self.path_output + 'val.data.npy', valset_data)
        np.save(self.path_output + 'val.label.npy', valtset_label)
        print("\tValidation Set Data  : ", valset_data.shape)
        print("\tValidation Set Labels: ", valtset_label.shape)

        testset_h5 = h5py.File(self.path_output + 'test6-2-2.h5', 'w')
        testset_h5.create_dataset('data', data=testset_data, dtype=np.float32)
        testset_h5.create_dataset('label', data=testset_label, dtype=np.int32)
        testset_h5.close()
        np.save(self.path_output + 'test.data.npy', testset_data)
        np.save(self.path_output + 'test.label.npy', testset_label)
        print("\tTest Set Data  : ", testset_data.shape)
        print("\tTest Set Labels: ", testset_label.shape)


parser = DatasetParser(DATASET_PATH, OUTPUT_PATH, TAXONOMY_PATH)
parser.convert_inplace_obj2hdf5_sampledFPS_all(n_jobs=8)
parser.concatenate_hdf5_files_split622()
