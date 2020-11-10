import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pandas import set_option
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from NeuralNetwork_util import do_scaler
from NeuralNetwork_models import baseline_mlp, baseline_sgd, baseline_deep, baseline_dropout, baseline_dropout_input
import h5py

proton_selection = "MultiRP"
PATH='/home/matheus/PPS-CMSDAS/h5py'
#PATH='output'
#eras=['B','C','D','E','F'] #uncomment to use all data
eras=['B']
stream='Mu' # 'El' OR 'Mu'

def GetData(flist,chunk_size=None):
    
    """ 
    opens a summary file or list of summary files and convert them to a pandas dataframe 
    if given the chunk_size will be used to collect events in chunks of this size
    """
    
    flist=flist if isinstance(flist,list) else [flist]
    
    df,df_counts=[],[]
    
    for filename in flist:
    
        with h5py.File(filename, 'r') as f:

            print('Collecting data from',filename)
            
            dset            = f['protons']
            dset_columns    = f['columns']
            dset_selections = f['selections']
            dset_counts     = f['event_counts']
            
            #read the data
            columns = list( dset_columns )
            columns_str = [ item.decode("utf-8") for item in columns ]
            if chunk_size is None:
                start=[0]
                stop=[dset.shape[0]]
            else:
                entries = dset.shape[0]
                start = list( range( 0, entries, chunk_size ) )
                stop = start[1:] + [entries]
                
            for idx in range( len( start) ):
                print('\tCollecting events',start[idx], stop[idx] )

                df.append( pd.DataFrame( dset[start[idx]:stop[idx]], 
                                         columns=columns_str ) )
                df[-1]=df[-1][['Run', 'LumiSection', 'EventNum', 'CrossingAngle', 
                               'MultiRP', 'Arm', 'RPId1',
                               'Xi', 'T', 'XiMuMuPlus', 'XiMuMuMinus',
                               'Lep0Pt', 'Lep1Pt', 'InvMass', 'ExtraPfCands_v1', 'Acopl'] ].astype( { "Run": "int64",
                                                                                                     "LumiSection": "int64",
                                                                                                     "EventNum": "int64",
                                                                                                     "MultiRP": "int32",
                                                                                                     "Arm": "int32",
                                                                                                     "RPId1": "int32",
                                                                                                     "ExtraPfCands_v1": "int32" } )
              
            #read the selection counters
            selections = list( dset_selections )
            selections_str = [ item.decode("utf-8") for item in selections ]        
            df_counts.append( pd.Series( list( dset_counts ), index=selections_str ) )
    
    n=len( df ) 
    print('\tReturning the result of %d merged datasets'%n)
    df_final=pd.concat(df)
    
    #merge the counts
    df_counts_final = df_counts[0]
    for idx in range( 1, len(df_counts) ):
        df_counts_final = df_counts_final.add( df_counts[idx] )

    #merge the data
    
    
    return df_final,df_counts_final
    
print('[Signal simulation]')
df_signal,df_counts_signal = GetData(PATH+'/output-GGTo{}_Elastic_v0_signal_xa120_era2017_preTS2.h5'.format('EE' if stream=='El' else 'MuMu'))
print('Selection counts')
print(df_counts_signal)

print('\n')
print('[Data (to be used as background)]')
data_files = [PATH+'/output-UL2017{}-{}-Rand20.h5'.format(era,stream) for era in eras]
df_bkg,df_counts_bkg = GetData(data_files,chunk_size=1000000)
print('Selection counts')
print(df_counts_bkg)

# Começo do código #

def PrepareData(df):
    
    """applies baseline selection cuts"""

    msk = ( df['InvMass'] > 100 )# & ( df['Acopl'] < 0.001 ) & (df['ExtraPfCands_v1'] < 15 ) 

    msk1 = None
    msk2 = None
    if proton_selection == "SingleRP":
        # Single-RP in pixel stations
        msk1_arm = ( df["RPId1"] == 23 )
        msk2_arm = ( df["RPId1"] == 123 )
        multiRP=0
    elif proton_selection == "MultiRP":
        # Multi-RP
        msk1_arm = ( df["Arm"] == 0 )
        msk2_arm = ( df["Arm"] == 1 )
        multiRP=1
   
    df[ "XiMuMu" ] = np.nan
    df[ "XiMuMu" ].where( ~msk1_arm, df[ "XiMuMuPlus" ],  inplace=True )
    df[ "XiMuMu" ].where( ~msk2_arm, df[ "XiMuMuMinus" ], inplace=True )
    msk1 = msk & ( df["MultiRP"] == multiRP) & msk1_arm
    msk2 = msk & ( df["MultiRP"] == multiRP) & msk2_arm

   
    return df[msk1 | msk2].copy()

df_signal_prep = PrepareData(df_signal)
df_signal_prep['deltaXi'] =  df_signal_prep['Xi'] - df_signal_prep['XiMuMu']
df_signal_prep = df_signal_prep[(df_signal_prep['InvMass'] > 100)] #& (df_signal_prep['Acopl'] < 0.001 ) & (df_signal_prep['ExtraPfCands_v1'] < 15 ) & ( abs(df_signal_prep['deltaXi']) < 0.02 )]

df_bkg_prep    = PrepareData(df_bkg)
df_bkg_prep['deltaXi'] =  df_bkg_prep['Xi'] - df_bkg_prep['XiMuMu']

df_bkg_prep = df_bkg_prep[ ( df_bkg_prep['InvMass'] > 100 )]# & ( df_bkg_prep['Acopl'] < 0.001 ) & ( df_bkg_prep['ExtraPfCands_v1'] < 15 ) & ( abs(df_bkg_prep['deltaXi']) < 0.02 )]


print('Signal prepared',df_signal_prep.shape)

train_vars=['Lep0Pt', 'Lep1Pt', 'InvMass',   'XiMuMu', 'Xi', 'Acopl','ExtraPfCands_v1']

#draw the correlation matrix for the training variables
print(train_vars)
fig=plt.figure(figsize=(10, 10))
plt.matshow(df_signal_prep[train_vars].corr(), fignum=fig.number)
cb = plt.colorbar()
plt.clim(-1,1)
plt.title('Signal correlation Matrix', fontsize=16)
#plt.show()

df_signal_prep[train_vars].head(10)

print('Background prepared',df_bkg_prep.shape)

#draw the correlation matrix for these variables
print(train_vars)
fig=plt.figure(figsize=(10, 10))
plt.matshow(df_bkg_prep[train_vars].corr(), fignum=fig.number)
cb = plt.colorbar()
plt.clim(-1,1)
plt.title('Background correlation Matrix', fontsize=16)
#plt.show()

df_bkg_prep[train_vars].head(10)

#save the trained model
from joblib import dump, load

def Plot2D(x,y,data_sig,data_bkg,msk_sig,msk_bkg,xran=[0.,0.15],yran=[0.,0.15]):
 
    """a simple routine to plot the signal and background components"""

    fig= plt.figure( figsize=(10,10) )
    plt.plot( data_bkg[x][ msk_bkg ], data_bkg[y][ msk_bkg ], 'ro', label='Background' )
    plt.plot( data_sig[x][ msk_sig ], data_sig[y][ msk_sig ], 'bo', label='Signal' )
    
    if 'Xi' in x and 'Xi' in y:
        plt.plot( xran,yran, 'k--', linewidth=1 )
        plt.plot( xran, [j*0.90 for j in yran], 'k:', linewidth=1 )
        plt.plot( xran, [j*1.10 for j in yran], 'k:', linewidth=1 )

    plt.xlim(*xran)
    plt.xlabel(x)
    plt.ylim(*yran)
    plt.ylabel(y)
    plt.legend(loc='best')


def showDist(x,data,mask_sig, mask_bkg,nbins):
    
    """a simple function to compare signal-like and background-like"""
    
    fig = plt.figure( figsize=(6,6) )
    _,bins=np.histogram(data[x],bins=nbins)
    plt.hist( data[ x ][ mask_bkg ], color='lightgray', bins=bins, density=True, label='Background' )
    plt.hist( data[ x ][ mask_sig ], histtype='step',   bins=bins, density=True, label='Signal', linewidth=2)
    plt.xlabel(x)
    plt.ylabel('Events (a.u.)')
    plt.grid()
    plt.legend(loc='best')
    plt.show()
    

#load the data
print('\n')
print('[Data]')
data_files = [PATH+'/output-UL2017{}-{}.h5'.format(era,stream) for era in eras]
df_data,df_counts_data = GetData(data_files,chunk_size=1000000)
print('Selection counts')
print(df_counts_data)

#prepare the data
df_data_prep=PrepareData(df_data)
df_data_prep['deltaXi'] = df_data_prep['Xi'] - df_data_prep['XiMuMu'] 
df_data_prep = df_data_prep[(df_data_prep['InvMass']> 100)]# & (df_data_prep['Acopl'] < 0.001 ) & (df_data_prep['ExtraPfCands_v1'] < 15 ) & ( abs(df_data_prep['deltaXi']) < 0.002 )]


df_signal_prep[ 'target' ] = 1 # target for class signal 
df_bkg_prep[ 'target'] = 0 # target for class background


DataSet = pd.concat([df_signal_prep,df_bkg_prep],axis=0) # full dataset with signal and background

drop_columns = [ 'Run', 'LumiSection', 'EventNum', 
'CrossingAngle', 'MultiRP', 'Arm', 'RPId1','target','XiMuMuMinus','XiMuMuPlus'] # Eliminating columns that are not important for training

DataSet_Train, dataset_test = train_test_split( DataSet, test_size = 0.50, random_state = 41 ) #selects samples in training and test

Y_train = DataSet_Train['target'] # selecting only the training classes
Y_test = dataset_test['target'] # selecting only the test classes

number_of_features = len(DataSet_Train.drop( drop_columns , axis = 1).columns)

X_train = DataSet_Train.drop( drop_columns , axis = 1)

#Use a scaler in the data

scaler = StandardScaler()
scaler_prediction = None

# scaler only on the input
X_train_norm, Y_train_norm, scaler, scaler_prediction = do_scaler(DataSet_Train.drop( drop_columns , axis = 1), Y_train, scaler, scaler_prediction)

# epochs = [200, 500, 1000, 2000]
# neurons = [10,50,100, 200]
# neurons1 = [10,50,100, 200] # Only for deep
# neurons2 = [10,50,100, 200]# Only for deep
# neurons3 = [10,50,100, 200] # Only for deep
# batch_size = [64, 128, 256, 512,1024]
# optimizer = ['RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# activation = [ 'relu', 'tanh', 'sigmoid', 'linear']
#learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3] # Only for SGD
#momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9] # Only for SGD
# dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#epochs = [100, 200, 300, 400, 500, 600, 700, 1000]
#neurons = [1,10, 20, 50, 100, 200, 300]
#batch_size = [64, 128]
#optimizer = ['Adam']
#activation = [ 'relu']

epochs = [100, 200]
neurons = [ 200, 300]
neurons1 = [ 200, 300] # Only for deep
neurons2 = [ 200, 300] # Only for deep
neurons3 = [ 200, 300] # Only for deep
batch_size = [32, 64, 128]
optimizer = ['Adam']
activation = ['relu']
learn_rate = [0.0001, 0.001,0.01,0.1] # Only for SGD
momentum = [0.3,0.5,0.7,0.8, 0.9] # Only for SGD
dropout_rate = [0.1, 0.2,0.3,0.4,0.6,0.8]

#  define model: choose the model number
#  you can also put the parameters values in the if for each model

model_number = 5
# Create Model
if model_number == 1:
    model_name = "mlp"
    build_fn = baseline_mlp
    param_search = dict(neurons=neurons, epochs=epochs, batch_size=batch_size,
                        optimizer=optimizer, activation=activation)
elif model_number == 2:
    model_name = "sgd"
    build_fn = baseline_sgd
    param_search = dict(neurons=neurons, learn_rate=learn_rate, momentum=momentum, epochs=epochs, batch_size=batch_size)
elif model_number == 3:
    model_name = "deep"
    build_fn = baseline_deep
    param_search = dict(neurons1=neurons1, neurons2=neurons2, neurons3=neurons3, epochs=epochs, batch_size=batch_size)
elif model_number == 4:
    model_name = "dropout"
    build_fn = baseline_dropout
    param_search = dict(neurons1=neurons1, neurons2=neurons2, neurons3=neurons3, epochs=epochs, batch_size=batch_size,
                        dropout_rate=dropout_rate)
elif model_number == 5:
    model_name = "dropout_input"
    build_fn = baseline_dropout_input
    param_search = dict(neurons1=neurons1, neurons2=neurons2, neurons3=neurons3, epochs=epochs, batch_size=batch_size,
                        dropout_rate=dropout_rate)
else:
    model_name = "sgd"
    build_fn = baseline_sgd(neurons=neurons, learn_rate=learn_rate, momentum=momentum, activation='relu')
    param_search = dict(neurons=neurons, learn_rate=learn_rate, momentum=momentum, epochs=epochs, batch_size=batch_size)


# define n_iter ( number of models) and cv (number of crossvalidation) and scoring

n_iter = 3
cv = 3
#scoring = make_scorer(fbeta_score, beta=0.5)
# scoring = 'f1'
scoring = None

#nmodel
model = KerasClassifier(build_fn = build_fn, verbose=1)

#search object
#search = GridSearchCV(estimator=model, param_grid=param_search, scoring = scoring ,cv=cv, verbose=1)
search = RandomizedSearchCV(estimator=model, param_distributions=param_search, n_iter= n_iter,scoring = scoring ,cv=cv, verbose=1)

start_time = time.time()

search_result = search.fit(X_train_norm, Y_train_norm)
print("--- %s seconds ---" % (time.time() - start_time))

# Results
print("Model: %s Best: %f using %s" % (model_name, search_result.best_score_, search_result.best_params_))
means = search_result.cv_results_['mean_test_score']
stds = search_result.cv_results_['std_test_score']
params = search_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

