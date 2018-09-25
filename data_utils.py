import os
import numpy as np
from utils import load_vlad_pickle
import pickle
import scipy.io as sio
import glob
from sklearn import preprocessing

def split_data(input_files, n_parts, output_dir, del_old_files = True):
    """
        Load all files in input_files, concatenate them 
        Split a training matrix into n_parts

    """
    if not os.path.exists(output_dir):
        print("Output dir {} does not exists!...Creating it".format(output_dir))
        os.mkdir(output_dir)


    if del_old_files:
        for file_name in glob.glob(output_dir+"/*.*"):
            os.remove(os.path.join( file_name))
    
    X = None   
    for filename in input_files:       
        if not os.path.exists(filename):
            print ('File {} does not exists'.format(filename))
        if X is None:
            X=load_vlad_pickle(filename)
        else:
            X1 = load_vlad_pickle(filename)
            X = np.concatenate( (X, X1), axis = 0)
            
    # Loaded data, start splitting
    N = X.shape[0]
    n_pts_per_chunk = np.floor(N/n_parts).astype('int')
    
    # Normalize Data:
    X = preprocessing.scale(X)

    start_idx = 0
    for i in range(n_parts):
        end_idx = start_idx + n_pts_per_chunk

        if i == n_parts-1:
            end_idx = N

        X1 = X[start_idx: end_idx, :]
        split_file_name = 'split_{}_{}.pickle'.format(start_idx, end_idx)
        split_file = os.path.join(output_dir, split_file_name)

        with open(split_file, 'wb') as f:
            pickle.dump([X1], f)
        start_idx = start_idx + n_pts_per_chunk

    return 0
    
def resize_pickle(input_dir, file_name = 'lost_features'):
    """
    Resize mistakenly save pickle to protocol 1
    """
    if not os.path.exists(input_dir):
        raise (ValueError("Input Dir {} does not exist".format(input_dir)))
    
    #Scan directories for files
    split_folders = glob.glob(input_dir + '/split_*')
    dataset_files = []
    for folder in split_folders:
        dataset_files += [os.path.join(folder, file_name)]
    if len(dataset_files) == 0:
        dataset_files = [ os.path.basename(b).split('.')[0]  for b in glob.glob(input_dir + "/*.pickle") ]    
    
    # Start resizing the pickled files
    for dt_file in dataset_files:
        X = load_vlad_pickle(os.path.join(input_dir, dt_file + '.pickle'))                           
        with open(os.path.join(input_dir, dt_file + '.pickle'),'wb') as f:
            pickle.dump([X], f, protocol=1)



def combine_data(input_dir, output_dir, file_name = 'lost_features', split_after_combine = True):
    """
    Load all data from different pickle files, and scale them
    """
    if not os.path.exists(input_dir):
        raise(ValueError("Input Dir {} does not exists".format(input_dir)))
    if not os.path.exists(output_dir):
        print("....Output Dir does not exist, creating directory {}".format(output_dir))
        os.mkdir(output_dir)
    
    # Scan directories for files
    split_folders = glob.glob(input_dir + '/brisbane_*')
    dataset_files = []
    for folder in split_folders:
        dataset_files += [os.path.join(folder, file_name)]
    if len(dataset_files) == 0:
        dataset_files = [ os.path.basename(b).split('.')[0]  for b in glob.glob(input_dir + "/*.pickle") ]    
        
    # Load all files into matrix
    X = None    
    indexes = []
    start_idx = 0
    for dt_file in dataset_files:
        print('Loading File {} .....'.format(dt_file))
        if X is None:
            X = load_vlad_pickle(os.path.join(input_dir, dt_file + '.pickle'))                           
        else:
            X1 = load_vlad_pickle(os.path.join(input_dir, dt_file + '.pickle'))
            X = np.concatenate((X, X1), axis = 0)            

        indexes += [[start_idx, X.shape[0]]]            
        start_idx = X.shape[0]
    
    # Scale the data    
    print X.shape           
    per_idx = np.random.permutation(range(X.shape[0]-1)).tolist()
    
    X = preprocessing.scale(X)   

    # Write combined data into all folder
    all_output_dir = os.path.join(output_dir,'all')
    if not os.path.exists(all_output_dir):
        os.mkdir(all_output_dir)
    dt_file_name = 'all.pickle'
    dt_file_name = os.path.join(all_output_dir, dt_file_name)
    print('Writing normalized data to {}...'.format(dt_file_name))
    with open(dt_file_name, 'wb') as f:        
        pickle.dump([X],f, protocol=0)            

    if not split_after_combine:
        return
    # Now write the data back to files
    for dt_idx, dt_file in enumerate(dataset_files):        
        idx_range= indexes[dt_idx]
        dt_file_name = 'split_{}_{}.pickle'.format(idx_range[0], idx_range[1])
        dt_file_name = os.path.join(output_dir, dt_file_name)
        print('Writing normalized data to {}...'.format(dt_file_name))
        with open(dt_file_name, 'wb') as f:
            X1 = X[idx_range[0]:idx_range[1],:]
            pickle.dump([X1],f, protocol=1)            
    
    return 0
        
        

def mat_to_pickle(input_file, output_file):
    """
    Convert an array stored in a matlab MAT file into a pickle file
    """
    if not os.path.exists(input_file):
        raise(ValueError('Input file does not exists'))
        
    X = sio.loadmat(input_file)['features']
    X = X.T
    #X = np.asarray(X)
    with open(output_file, 'wb') as f:
        pickle.dump([X], f)
        
    
    



