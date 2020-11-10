import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import keras.backend as K
from NeuralNetwork_util import f1_score
from NeuralNetwork_models import baseline_sgd, baseline_mlp, baseline_deep, baseline_dropout, baseline_dropout_input
import h5py
from keras.optimizers import SGD
from sklearn.metrics import PrecisionRecallDisplay,precision_score,plot_roc_curve,recall_score,accuracy_score,log_loss,roc_auc_score,classification_report,f1_score,confusion_matrix,roc_curve,precision_recall_curve,plot_precision_recall_curve,average_precision_score

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
DataSet_Train, dataset_val = train_test_split( DataSet_Train, test_size = 0.05, random_state = 41 ) #selects samples in training and test

Y_train = DataSet_Train['target'] # selecting only the training classes
Y_test = dataset_test['target'] # selecting only the test classes

number_of_features = len(DataSet_Train.drop( drop_columns , axis = 1).columns)

X_train = DataSet_Train.drop( drop_columns , axis = 1)

#Use a scaler in the data

scaler = StandardScaler()
scaler_prediction = None

# scaler only on the input
#X_train_norm, Y_train_norm, scaler, scaler_prediction = do_scaler(DataSet_Train.drop( drop_columns , axis = 1), Y_train, scaler, scaler_prediction)

X_train_norm = scaler.fit_transform(DataSet_Train.drop( drop_columns , axis = 1))

X_val_norm = scaler.transform(dataset_val.drop( drop_columns , axis = 1))

X_test_norm = scaler.transform(dataset_test.drop( drop_columns , axis = 1))

Y_train = DataSet_Train.target

Y_val =  dataset_val.target

Y_test =  dataset_test.target

# Best model number
# 1: mlp
# 2: sgd
# 3: deep
# 4:dropout
# 5: dropout_input

best_model_number = 2

# Best Parameters
epochs = 200
neurons = 100
neurons1 = 300 # Only for deep
neurons2 = 200 # Only for deep
neurons3 = 200 # Only for deep
batch_size = 128
optimizer = 'Adam'
activation = 'relu'
learn_rate = 0.001 # Only for SGD
momentum = 0.9 # Only for SGD
dropout_rate = 0.6


# Create Model
if best_model_number == 1:
    print("mlp")

    model = baseline_mlp(neurons=neurons, optimizer=optimizer, activation = activation)
elif best_model_number == 2:
    print("sgd")
    model = baseline_sgd( number_of_features = number_of_features,neurons=neurons, learn_rate=learn_rate, momentum=momentum, activation='relu')
elif best_model_number == 3:
    print("deep")
    model = baseline_deep(neurons1 = neurons1,neurons2 = neurons2,neurons3 = neurons3)
elif best_model_number == 4:
    print('dropout')
    model = baseline_dropout(neurons1 = neurons1,neurons2 = neurons2,neurons3 = neurons3, dropout_rate=dropout_rate)
elif best_model_number ==5:
    print('dropout_input')
    model = baseline_dropout_input(neurons1=neurons1, neurons2=neurons2, neurons3=neurons3, dropout_rate=dropout_rate)
else:
    print("sgd")
    model = baseline_sgd(neurons=neurons, learn_rate=learn_rate, momentum=momentum, activation='relu')

# callbacks de Early stopping  and Model Check Point
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint('ml4jets_best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True) # Save the best model
# Train the model
history = model.fit(X_train_norm, Y_train, validation_data=(X_val_norm, Y_val), batch_size= batch_size,epochs=epochs, verbose=0, callbacks=[es, mc])

print(history.history.keys())
# plot training history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# Load Best Model

saved_model = load_model('ml4jets_best_model.h5', custom_objects={"f1_score":f1_score})
# Eval best model

train_loss, train_acc,train_f1_score = saved_model.evaluate(X_train_norm, Y_train, verbose=2)
test_loss, test_acc,test_f1_score = saved_model.evaluate(X_test_norm, Y_test, verbose=2)
print('Train accuracy: %.6f, Test accuracy: %.6f' % (train_acc, test_acc))
print('Train loss: %.6f, Test loss: %.6f' % (train_loss, test_loss))
print('Train f1_score: %.6f, Test f1_score: %.6f' % (train_f1_score, test_f1_score))

predict = model.predict(X_test_norm)

fpr_lgb, tpr_lgb, thresholds_lgb = roc_curve( Y_test, predict ) # fpr -> false positive rate | tpr -> true positive rate
prec_lgb, rec_lgb, threshs_lgb = precision_recall_curve( Y_test, predict ) # prec -> precision | rec -> recall 
f1 = 2 * ( prec_lgb * rec_lgb ) / ( prec_lgb + rec_lgb )
threshs_lgb  = np.concatenate( [ threshs_lgb , [1] ] , axis = 0 )

bidx = np.argmax(prec_lgb*rec_lgb)
best_cut = threshs_lgb[bidx]
print('\n',' Best Cut ------>> ', best_cut,'\n')
y_pred_cut = predict >= best_cut

import seaborn as sns

fig= plt.figure( figsize=(10,10) )
conf_mat = confusion_matrix(y_true=Y_test, y_pred=y_pred_cut)
print('Confusion matrix:\n', conf_mat)
conf_mat = pd.DataFrame(conf_mat)
conf_mat = conf_mat.rename(columns={0:'Background', 1:'Signal'})
conf_mat = conf_mat.T
conf_mat = conf_mat.rename(columns={0:'Background', 1:'Signal'})
conf_mat = conf_mat.T
sns.heatmap(conf_mat, annot=True, fmt="d",cmap = 'RdBu')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.tight_layout()
plt.show()
#plt.savefig('/home/matheus/test/confusion_matrix.png')
plt.close()


hbgt_lgb =  plt.hist(predict[Y_test==0],bins=np.linspace(0,1,50), histtype='step',label='Background', color = 'b')
hsigt_lgb = plt.hist(predict[Y_test==1],bins=np.linspace(0,1,50), histtype='step',label='Signal', color = 'orange')
uppery_lgb=np.max(hbgt_lgb[0])
plt.plot([best_cut,best_cut],[0,uppery_lgb],"-.r",label='Best cut : {:2.2f}'.format(best_cut))
plt.xlabel("Probability", fontsize = 15)
plt.ylabel("Events",fontsize = 15)
plt.legend(loc="upper right", fontsize = 15)
plt.text(0.1,10e3, "Purity: {:2.2f}%".format(100*precision_score(Y_test,y_pred_cut)),fontsize = 12)
plt.text(0.1,10e2, "Efficiency: {:2.2f}%".format(100*recall_score(Y_test,y_pred_cut)),fontsize = 12)
#plt.text(0.1,10e3, "Accuracy: {:2.2f}%".format(100*accuracy_score(y_test_,y_pred_cut)), fontsize = 18)
plt.text(0.1,10e1, "ROC AUC: {:2.2f}%".format(100*roc_auc_score(Y_test,y_pred_cut)), fontsize = 12)
plt.text(0.1,10e0, "F1_Score: {:2.2f}%".format(100*f1_score(Y_test,y_pred_cut)), fontsize = 12)
plt.text(best_cut+0.06,1000, 'SIGNAL REGION', color = 'red', fontsize = 18)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,3), useMathText = True)
#plt.style.use(hep.style.CMS)
#hep.cms.cmslabel(data=False, paper=False, year='$18.34 fb^{-1}$')
plt.yscale('log')
plt.tight_layout()
#plt.figure(figsize = (20, 12))
plt.show()
#plt.savefig('/home/matheus/Discriminant.png')
plt.close()

print("End of Program")