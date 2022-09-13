import numpy as np  # For array operations
import pandas as pd  # For Dataframe operations (similar to Excel spreadsheets)
from scipy.stats import norm
import sys
sys.path.append(r"/home/jaina/scripts/my_py_functions")
# sys.path.append("C:\\Users\\akashjn\\Desktop\\my_py_functions")
import all_mbo_functions as myfn
from tqdm.auto import tqdm  # progress bar

from sklearn.preprocessing import StandardScaler # For normalizing inputs
from sklearn.decomposition import PCA # Principle component analysis

def do_scaling(scaler=StandardScaler(), xtrain=None, xtest=None):
    """
    Usage: do_scaling(scaler=MinMaxScaler(), xtrain=xtrain, xtest=test) 
    Caution: Do test_train_split before scaling
    Return: return scaled non-None xtrain and xtest
    """
    st = scaler

    if xtrain is not None:
        # col=xtrain.columns.values.tolist()
        xtrain=st.fit_transform(xtrain)  
        # xtrain=pd.DataFrame(xtrain,columns=col)

        if xtest is not None:
            
            xtest=st.transform(xtest)
            # xtest=pd.DataFrame(xtest,columns=col)
            print("returning scaled train and test data")
            return xtrain,xtest
        else:
            print("test data is not provided, returning only scaled train data")
            return xtrain
    else:
        print("Give train data, returning None")
        return xtrain,xtest


smiles = "00_DFT_1594_clean.csv"
features = "00_DFT_1594_clean_features.csv"

# Load SMILES and features
dfSMILES = pd.read_csv(smiles)
Xsmiles = dfSMILES.smiles
X = pd.read_csv(features)
X=X.drop(X.columns[X.eq(0).all()], axis=1)

n_PC = 16  # Set number of principle components
st = StandardScaler()
Xdata = st.fit_transform(X)  # Normalize feature vectors

pca = PCA(n_components=n_PC)
Xdata = pca.fit_transform(Xdata)  # Transform feature vectors to PCs


dfEox = dfSMILES.drop(columns=['Index', 'smiles', 'complexity', 'atoms', 'MW'])

# def lookupComputedEox(smiles):  # Look up function for computed Eox values
#     smilesLoc = dfSMILES[dfSMILES.smiles == smiles].index[0]
#     # smilesLoc = dfSMILES[dfSMILES.SMILES == smiles].index[0]
#     return dfEox.iloc[smilesLoc].values[0]
fileName = 'BayesOptRunProgress'
pfile = open(fileName+'.csv','a+')
pfile.write("Best SMILES, Ered, Iteration")
pfile.write("\n")


def BayesOpt(Xdata,Ydata,Xinfo,ndata,nPC,eps,af):
    ntrain = 10 # Number of initial training data points
    nremain = ndata - ntrain
    dataset = np.random.permutation(ndata)
    a1data = np.empty(ntrain, dtype=int)
    a2data = np.empty(nremain, dtype=int)
    a1data[:] = dataset[0:ntrain]
    a2data[:] = dataset[ntrain:ndata]

    Xtrain = np.ndarray(shape=(ntrain, nPC), dtype=float)
    Xtraininfo = np.chararray(ntrain, itemsize=100)
    Ytrain = np.empty(ntrain, dtype=float)
    Xtrain[0:ntrain, :] = Xdata[a1data, :]
    Xtraininfo[0:ntrain] = Xinfo[a1data]
    Ytrain[0:ntrain] = Ydata[a1data]
    
    yoptLoc = np.argmax(Ytrain)
    yopttval = Ytrain[yoptLoc]
    # print("yopttval",yopttval)
    xoptval = Xtraininfo[yoptLoc]
    yoptstep=0
    yopinit = yopttval
    xoptint = xoptval

    Xremain = np.ndarray(shape=(nremain, nPC), dtype=float)
    Xremaininfo = np.chararray(nremain, itemsize=100)
    Yremain = np.empty(nremain, dtype=float)
    Xremain[0:nremain, :] = Xdata[a2data, :]
    Xremaininfo[0:nremain] = Xinfo[a2data]
    Yremain[0:nremain] = Ydata[a2data]
    
    print('*** Initial training set ***')
    print(115*'-')
    print('{:<5s}{:<80s}{:<15s}'.format('Id','SMILES','Eox'))
    print(115*'-')
    for i in range(ntrain):
        print('{:<5d}{:<80s}{:<15f}'.format(i,Xtraininfo[i].decode(),Ytrain[i]))
    print(115*'-')  

    print("Total number of inital training points: ", ntrain)
    print("Initial best SMILES is "+xoptval.decode()+' with Eox = '+str(yopttval)+' V')
    
    for ii in tqdm(range(0, Niteration),desc='Progress'):
        model, likelihood = myfn.gpregression_pytorch(X_train=Xtrain,y_train=Ytrain,num_iter=350,learning_rate=0.05,verbose=False)
        yt_pred, tsigma = myfn.gprediction_pytorch(model,likelihood,X_test=Xtrain)
        
        ybestloc = np.argmax(Ytrain) # The current best y value
         
        ybest = yt_pred[ybestloc]
        ytrue = Ytrain[ybestloc]
                
        if yopttval < ytrue:
            yopttval = ytrue
            xoptval = Xtraininfo[ybestloc]
            
        if af=='EI':
            # afValues = expectedImprovement(Xremain, gpnetwork, ybest, eps)    
            afValues = myfn.expectedimprovement_pytorch(xdata=Xremain, gp_model=model,gp_likelihood=likelihood, ybest=ybest, itag=1, epsilon=eps)
        elif af=='POI':
        #     afValues = probabilityOfImprovement(Xremain, gpnetwork, ybest, eps)
            afValues = myfn.probabilityOfImprovement_pytorch(xdata=Xremain, gp_model=model,gp_likelihood=likelihood, ybest=ybest, epsilon=eps)
        elif af=='UCB':
            # afValues = upperConfidenceBound(Xremain, gpnetwork, ybest, eps)
            afValues = myfn.upperConfidenceBound_pytorch(xdata=Xremain, gp_model=model,gp_likelihood=likelihood, psilon=eps)

        # afMax = np.max(afValues)
        afmaxloc = np.argmax(afValues)
        
        xnew = np.append(Xtrain, Xremain[afmaxloc]).reshape(-1, nPC)
        xnewinfo = np.append(Xtraininfo, Xremaininfo[afmaxloc])
        ynew = np.append(Ytrain, Yremain[afmaxloc])
        xrnew = np.delete(Xremain, afmaxloc, 0)
        xrnewinfo = np.delete(Xremaininfo, afmaxloc)
        yrnew = np.delete(Yremain, afmaxloc)
        if ii==0:
            Xexplored=Xremaininfo[afmaxloc]
            Yexplored=Yremain[afmaxloc]
        else:
            Xexploredtemp=np.append(Xexplored, Xremaininfo[afmaxloc])
            Yexploredtemp=np.append(Yexplored, Yremain[afmaxloc])
            del Xexplored,Yexplored
            Xexplored=Xexploredtemp
            Yexplored=Yexploredtemp
        del Xtrain, Ytrain, Xremaininfo, model, likelihood
        # del Xtrain, Ytrain, Xremaininfo
        Xtrain = xnew
        Xtraininfo = xnewinfo
        Ytrain = ynew
        Xremain = xrnew
        Xremaininfo = xrnewinfo
        Yremain = yrnew
        del xnew, xnewinfo, ynew, xrnew, xrnewinfo, yrnew
    
    # What if the best candidate is found in the last step? (this was not present in the Hieu's code)
    # Look for the best candidate in the final YTrain
    
    ybestloc = np.argmax(Ytrain) # The current best y value
    ytrue = Ytrain[ybestloc]

    print("yt_true_best  ",ytrue)
            
    if yopttval < ytrue:
        yopttval = ytrue
        xoptval = Xtraininfo[ybestloc]
    ###

    if not yopinit==yopttval:
        yoptstep = np.argmax(Yexplored) + 1       
    else:
        yoptstep=0
    dataorder = np.argsort(Yexplored)
    Yexploredtemp=Yexplored[dataorder]
    Xexploredtemp = Xexplored[dataorder]
    print('*** Summary ***')
    print(115*'-')
    print('{:<15s}{:<80s}{:<15s}'.format('Iteration','SMILES','Eox'))
    print(115*'-')
    for i,sml in enumerate(Xexplored):   
        print('{:<15d}{:<80s}{:<15f}'.format(i+1,sml.decode(),Yexplored[i]))
    print(115*'-')  
    print("\n")
    # print(yopttval)
    pfile.write(xoptval.decode()+","+str(-yopttval)+","+str(yoptstep))
    pfile.write("\n")
    print("The best SMILES is "+xoptval.decode()+" with Eox = "+str(-yopttval)+" V, which was found in iteration "+str(yoptstep))
    return xoptint,yopinit,xoptval,yopttval, yoptstep

dfEox = dfSMILES.drop(columns=['Index', 'smiles', 'complexity', 'atoms', 'MW'])
Ydata = dfEox["DelE_red(V)"].values 
Ydata = -Ydata  # minimize

print('*** Finding SMILES with maximum Eox ***')
ndata = len(Ydata)
print("Original shape of X and Y :",np.shape(Xdata),np.shape(Ydata))
epsilon=0.01
acquiFunc='EI'  # POI, UCB
Nruns=3 # for statistics
Niteration = 20   # number of iteration in a given Bayesian  Optimization
Xinitguess = np.chararray(Nruns,itemsize=100)
Yinitguess = np.empty(Nruns,dtype=float)
Xoptimal = np.chararray(Nruns,itemsize=100)
Yoptimal = np.empty(Nruns,dtype=float)
Yopt_step = np.empty(Nruns,dtype=float)
# res=[]
for ii in range(0,Nruns):
    print('Run ',ii)
    Xinitguess[ii], Yinitguess[ii], Xoptimal[ii], Yoptimal[ii], Yopt_step[ii] = BayesOpt(Xdata, Ydata, Xsmiles, ndata, n_PC,epsilon,acquiFunc)

pfile.close()
