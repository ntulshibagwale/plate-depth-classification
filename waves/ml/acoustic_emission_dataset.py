import torch
from torch.utils.data import Dataset
from torch import tensor
import numpy as np
from sklearn import preprocessing


from waves.load_data import load_json_file_from_path
from waves.ml.feature_vectors import calc_alpha
from waves.ml.feature_vectors import calc_beta
from waves.ml.feature_vectors import calc_gamma
from waves.ml.feature_vectors import calc_delta
from waves.ml.feature_vectors import calc_epsilon
from waves.ml.feature_vectors import calc_zeta
from waves.ml.feature_vectors import calc_eta
from waves.ml.feature_vectors import calc_theta
from waves.ml.feature_vectors import calc_iota

class AcousticEmissionDataset(Dataset):
    
    def __init__(self, path, sig_len ,dt, low_pass, high_pass, feature, 
                 scaler=None):
        """
        
        Constructor. Runs when object is created. Pulls data from specified
        directory and performs feature / label extraction according to user
        specified variables.

        Parameters
        ----------
        path : path
            String / directory location of .json datafile.
        sig_len : int
            Number of samples to use from raw waveforms. For example, if 16384 
            samples are measured experimentally, specifying sig_len = 1024 will
            take the first 1024 samples only for feature extraction.
        dt : float
            Sample period. 1/sampling_freq. Typically dt = 1/(10*10^6) = 10^-7.
        low_pass : int
            Filter freq data below this number in fft.
        high_pass : int
            Filter freq data above this number in fft.
        feature : string
            Specified feature to extract from raw waveform.
        scaler : scikit learn object, optional
                Passed in if scaling was performed as a preprocessing step in 
                training data preparation. This performs same transform on 
                current data (validation or test). I.e. Centering and scaling.
                Default is None.

        """
        # Variables associated with object
        self.path = path
        self.sig_len = sig_len
        self.dt = dt
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.feature = feature
        self.scaler = scaler
        
        # Load in Acoustic Emission Data
        data = load_json_file_from_path(path)
        print('Dataset dictionary keys:')
        print(f'{data.keys()}\n')
        
        # Coupling dataset
        self.waves = data['waves']
        #self.coupling = data['coupling']
        
        self.location = data['location']
        self.sensor = data['sensor']
               
        # Extract Feature Vector Matrix (x) from waveform data
        self.x = self._extract_features()
        
        # Extract labels (y) and one hot encoded labels (if classification)
        self.y, self.y_one_hot = self._extract_labels()

        # Number of examples 
        self.n_samples = self.y.shape[0]   

        print("AcousticEmissionDataset object succesfully created from:\n",
              f"{path} \n")
            
        return 
    
    def __getitem__(self,ex_idx):
        """
                
        Function called when dataset object is indexed. 
        For example:
        (x, y, ex_idx) = ae_dataset[0]

        Parameters
        ----------
        ex_idx : int
            Example index in dataset.

        Returns
        -------
        x : array-like
            Example feature vector. Size depends on selected feature.
        y : array-like
            One hot encoded vector for label.
        ex_idx : int
            Example index in dataset.

        """
        x = self.x[ex_idx]   
        y = self.y_one_hot[ex_idx] 
              
        return x, y, ex_idx
    
    def __len__(self):
        """
        
        Return number of samples in object dataset. 
        For example:
        n_samples = len(ae_dataset)
        
        """
        return self.n_samples
    
    def _extract_labels(self):
        """
        
        Extract labels used for supervised ml learning.

        Returns
        -------
        y : list
            Labels associated with each example in x. 
        y : list
            Labels in one hot encoded form. For classification.

        """
        y = []
        y_one_hot = []
            
        locations = set(self.location)
        # Map source location to a number (starting from 0)
        if len(locations) == 2:
            if locations == {'top','sid'}:
                for loc in self.location:
                    if loc == 'top':
                        y.append(0)
                    elif loc == 'sid':
                        y.append(1)
            elif locations == {'top','bot'}:
                for loc in self.location:
                    if loc == 'top':
                        y.append(0)
                    elif loc == 'bot':
                        y.append(1)                        
        else: # 3 classes
            for loc in self.location:
                if loc == 'top':
                    y.append(0)
                elif loc == 'sid':
                    y.append(1)
                elif loc == 'bot':
                    y.append(2)
                            
        # Type cast data
        print(f"Length of labels (y) is: {len(y)}")        
        print(f"Datatype of labels (y) is: {type(y)}")
        print(f"Datatype of a label example (y[0]) is: {type(y[0])}")
        print('Changing datatype to pytorch tensor...')

        y = tensor(y,dtype=torch.int64,requires_grad=False)

        print(f"Shape of label (y) is: {y.shape}")
        print(f"Datatype of labels (y) is: {y.dtype}")
        print(f"Datatype of label example (y[0]) is: {y[0].dtype}")
        print("y requires grad:", y.requires_grad)
        print(f"Example label: y[0] = {y[0]}")
        print("")

        # Get one hot encoded label vector
        print('Creating one hot encoded version of y (y_one_hot)...')
        # One hot encode the label
        y_one_hot = torch.nn.functional.one_hot(y.long()) 
        y_one_hot = y_one_hot.float()
        print("y_one_hot is the one hot encoding of label for source." ,
              " For ex: [1 0 0] is a top occuring PLB.")
        print(f"Shape of y_one_hot is: {y_one_hot.shape}")
        print(f"Datatype of y_one_hot is: {y_one_hot.dtype}")
        print(f"Example label: y_one_hot[0] = {y_one_hot[0]}")
        print("")

        return y, y_one_hot              
    
    def _extract_features(self):
        """
        
        Extracts features from raw waveform data as specified by variables
        defined in object's constructor. Scales data. The raw waveform data is
        reshaped as specified by signal length; this hyperparameter is avaiable
        because features such as fft depend on waveform length.

        Returns
        -------
        x : 2D numpy array
            Feature vector matrix. Shape = [# of examples, # of feature values]

        """
        print("Extracting feature vector matrix from raw waveform data...\n")
        print(f"FEATURE: {self.feature}")
        
        # Reduce waveform length to specified sig_len 
        resized_waves = self._adjust_waveform_sig_len()
        
        # Get feature vector matrix (x)
        x = []
        if self.feature == 'alpha':
            for index, wave in enumerate(resized_waves):
                x.append(calc_alpha(wave))
            x = self._scale_features(x)
            
        elif self.feature == 'beta':
            for index, wave in enumerate(resized_waves):
                x.append(calc_beta(wave))
            x = self._scale_features(x)

        elif self.feature == 'gamma':
            for index, wave in enumerate(resized_waves):
                x.append(calc_gamma(wave))
            x = self._scale_features(x)
            
        elif self.feature == 'delta':
            for index, wave in enumerate(resized_waves):
                x.append(calc_delta(wave))
            x = self._scale_features(x)
        
        elif self.feature == 'epsilon':
            for index, wave in enumerate(resized_waves):
                x.append(calc_epsilon(wave))
            x = self._scale_features(x)    
         
        elif self.feature == 'zeta':
            for index, wave in enumerate(resized_waves):
                x.append(calc_zeta(wave))
            x = self._scale_features(x)

        elif self.feature == 'eta':
            for index, wave in enumerate(resized_waves):
                x.append(calc_eta(wave))
            x = self._scale_features(x)    

        elif self.feature == 'theta':
            for index, wave in enumerate(resized_waves):
                x.append(calc_theta(wave))
            x = self._scale_features(x)

        elif self.feature == 'iota':
            for index, wave in enumerate(resized_waves):
                w=calc_iota(wave)
                x.append(w/np.max(np.abs(w)))
                #x.append(calc_iota(wave))
                
            #x = self._scale_features(x)

        # Type cast data
        x = np.array(x)          
        print(f"Shape of feature vector matrix is: {x.shape}")
        print(f"Datatype of feature vector matrix is: {x.dtype}")   
        print('Changing datatype to pytorch tensor...')        
        x = tensor(x,dtype=torch.float32,requires_grad=False)        
        print(f"Shape of feature vector matrix is now: {x.shape}")                  
        print(f"Datatype of feature vector matrix is now: {x.dtype}")
        print("x requires grad:", x.requires_grad)
        print("")

        return x
    
    def _scale_features(self, x):
        """

        Scale feature vector matrix to have zero mean and unit variance. Used
        when feature values are on different scales, such as a feature vector
        that used amplitude and peak frequency.

        Parameters
        ----------
        x : array-like
            Dataset to be scaled.
            
        Returns
        -------
        x_scaled : array-like.
            Scaled version of x.

        """
        print('Scaling data to have zero mean and unit variance...')
        x = np.array(x)
        if self.scaler == None: # Fit scaler to data if None given
            self.scaler = preprocessing.StandardScaler().fit(x) 
        # else: it was previously fit already
        x_scaled = self.scaler.transform(x)  
        print(f"self.scaler.mean_ = {self.scaler.mean_}")
        print(f"self.scaler.var_ = {self.scaler.var_}\n")

        return x_scaled
    
    def _adjust_waveform_sig_len(self):
        """
         
        Resize the waveforms to have the specified signal length. For example,
        if all the waveforms were collected experimentally at 16384 samples,
        this function reshapes them to have the first 1024 samples only, say.

        Returns
        -------
        resized_waves : array-like
            Resized waves, sampe example # as self.waves but with new length.

        """
        resized_waves = np.zeros((self.waves.shape[0],self.sig_len))
        for idx, wave in enumerate(self.waves):
            resized_waves[idx] = wave[0:self.sig_len]
            
        print(f'Raw experiment waveform signal length: {self.waves.shape[1]}')
        print(f'DESIRED SIG_LEN: {self.sig_len}')
        print(f'First {self.sig_len} samples used for feature extraction. \n')
        
        return resized_waves
    

class AutoEncoderDataset(Dataset):
    
    def __init__(self, path, scaler=None):
        
        # Variables associated with object
        self.path = path
        self.scaler = scaler
        
        # Load in Acoustic Emission Data
        data = load_json_file_from_path(path)
        print('Dataset dictionary keys:')
        print(f'{data.keys()}\n')
        
        # Coupling dataset
        self.x = data['waves']
        #self.coupling = data['coupling']

        # Extract labels (y) and one hot encoded labels (if classification)
        self.y = data['waves']

        # Number of examples 
        self.n_samples = self.y.shape[0]   

        print(f"Shape of feature vector matrix is: {self.x.shape}")
        print(f"Datatype of feature vector matrix is: {self.x.dtype}")   
        print('Changing datatype to pytorch tensor...')        
        self.x = tensor(self.x,dtype=torch.float32,requires_grad=False)        
        print(f"Shape of feature vector matrix is now: {self.x.shape}")                  
        print(f"Datatype of feature vector matrix is now: {self.x.dtype}")
        print("self.x requires grad:", self.x.requires_grad)
        print("")
        
        # Type cast data
        print(f"Length of labels (self.y) is: {len(self.y)}")        
        print(f"Datatype of labels (self.y) is: {type(self.y)}")
        print(f"Datatype of a label example (self.y[0]) is: {type(self.y[0])}")
        print('Changing datatype to pytorch tensor...')

        self.y = tensor(self.y,dtype=torch.float32,requires_grad=False)        

        print(f"Shape of label (self.y) is: {self.y.shape}")
        print(f"Datatype of labels (self.y) is: {self.y.dtype}")
        print(f"Datatype of label example (self.y[0]) is: {self.y[0].dtype}")
        print("self.y requires grad:", self.y.requires_grad)
        print(f"Example label: self.y[0] = {self.y[0]}")
        print("")
 
        print("AcousticEmissionDataset object succesfully created from:\n",
              f"{path} \n")
            
        return 
    
    def __getitem__(self,ex_idx):
        """
                
        Function called when dataset object is indexed. 
        For example:
        (x, y, ex_idx) = ae_dataset[0]

        Parameters
        ----------
        ex_idx : int
            Example index in dataset.

        Returns
        -------
        x : array-like
            Example feature vector. Size depends on selected feature.
        y : array-like
            One hot encoded vector for label.
        ex_idx : int
            Example index in dataset.

        """
        x = self.x[ex_idx]   
        y = self.y[ex_idx] 
              
        return x, y, ex_idx
    
    def __len__(self):
        """
        
        Return number of samples in object dataset. 
        For example:
        n_samples = len(ae_dataset)
        
        """
        return self.n_samples
        
 