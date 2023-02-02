import pandas as pd
import numpy as np

class LoadDataset:
        def __init__(self):
                self.directory = 'dataset/'
                self.datafiles = ['abalone.data',
                                 'forestfires.data', 'glass.data', 'cancer-wisconsin.data','soybeansmall.data', 'machine.data']
                self.datafiles = ['forestfires.data']
                self.alldataset = {}
                
        def load_data(self):
                for files in self.datafiles:       
                        #read each datafiles
                        data = pd.read_csv(self.directory + files)
                        #give filename without extension as dict key for each dataset
                        key = files.split('.')[0]
                        self.alldataset[key] = self.PreprocessingData(key, data)
                return self.normalize_data()
                
        def PreprocessingData(self, key, data):
                if key == 'abalone':
                        data = data.drop(['Sex'], axis= 1)
                        classes = data[data.columns[-1]].values
                        classes = classes - 1
                        data[data.columns[-1]] = classes
                        data = data.replace({data.columns[-1] : {28:27}})
                elif key == 'glass':
                        data = data.replace({'class':{1:0, 2: 1, 3:2, 5:3, 6:4, 7:5}})
                        class_c = data['class']
                        data = data.drop(['class'], axis = 1)
                        data['class'] = class_c
                elif key == 'forestfires':
                        data = data.drop(['month', 'day'], axis= 1)
                elif key == 'cancer-wisconsin':
                        data = data.drop(['ID'], axis= 1)
                        data['BareNuclei'] = data['BareNuclei'].astype(int)
                        data['class'] = data['class'].map({2:0 , 4:1})
                elif key == 'soybeansmall':
                        data = data.drop(['date'], axis = 1)
                elif key == 'machine':
            
                        data = data.drop(['Vendor name', 'Model name', 'ERP'], axis= 1)
                return data
        
        def normalize_data(self):
                #normalize dataset points with min-max normalization
                for key in self.alldataset:
                        data = self.alldataset.get(key)
                        isClassification = self.IsClassificationDict().get(key)
                        
                        self.alldataset[key] = self.normalize(data, isClassification)
                return self.alldataset
        
        def normalize(self, data, isClassification):
                #is dataset is classification don't normalize the class output
                #otherwise for regression normalize the prediction output.
                if isClassification:    cols = data.columns[:-1] 
                else:   cols = data.columns
                for col in cols:
                    col_values = data[col].values
                    value_min = min(col_values)
                    value_max = max(col_values)
                    data[col] = (col_values - value_min) / (value_max - value_min)
                data = data.fillna(0)
                return data
        
        def get_neural_net_input_shape(self, data_all,  dataset, isClassification = True):
                #get data for neural net format with a unit vector for class output label
                #containing 1 for that class and 0's for other class. For regression we have
                #only one output layer with the actual value
                data = list()
                label = list()
                class_len = len(data_all[data_all.columns[-1]].unique())
                for index, row in dataset.iterrows():
                        if isClassification:                                
                                row_label = int(row[dataset.columns[-1]])
                                unit_vec = np.zeros((class_len, 1))
                                unit_vec[row_label] = 1
                                label.append(unit_vec)
                        else:
                                label.append(row[dataset.columns[-1]])
                        data.append(np.reshape(np.asarray(row[dataset.columns[:-1]]),
                                               (len(dataset.columns[:-1]), 1)))
                return data, label
                        
        
        def IsClassificationDict(self):
                #return if dataset is classification or regression.
                return {'abalone': False, 'glass': True, 'soybeansmall': True, 'machine': False,
                        'forestfires': False,  'cancer-wisconsin': True} 
                
        def get1sthiddenlayernode(self, key):
                #define hidden layer 1 node number based on dataset and datapoints
                #and tuned to get best performance.
                dict_list = {'abalone': 100, 'glass': 30, 'machine': 30,
                        'forestfires': 30, 'cancer-wisconsin': 60, 'soybeansmall': 40}
                return dict_list.get(key)
        
        def get2ndhiddenlayernode(self, key):
                #define node number for 2 layers for each dataset and tune to get 
                #best performance.
                dict_list = {'abalone': [40,40], 'glass': [15,15], 'cancer-wisconsin': [10, 10], 'machine': [15,15] ,
                        'forestfires': [15,15], 'soybeansmall': [15,15]}
                return dict_list.get(key)
#ld = LoadDataset()
#alldata = ld.load_data()
#data, label = ld.get_neural_net_input_shape(alldata.get('segmentation'))


                