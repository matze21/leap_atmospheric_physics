import numpy as np
import pandas as pd
import dask.dataframe as dd


feat60 = ['state_t', 'state_q0001','state_q0002','state_q0003','state_u','state_v','pbuf_ozone','pbuf_CH4','pbuf_N2O']
feat1 = ['state_ps','pbuf_SOLIN','pbuf_LHFLX','pbuf_SHFLX','pbuf_TAUX','pbuf_TAUY','pbuf_COSZRS','cam_in_ALDIF','cam_in_ALDIR','cam_in_ASDIF','cam_in_ASDIR','cam_in_LWUP','cam_in_ICEFRAC','cam_in_LANDFRAC','cam_in_OCNFRAC','cam_in_SNOWHLAND']

target60 = ['ptend_t','ptend_q0001','ptend_q0002','ptend_q0003','ptend_u','ptend_v']
target1 = ['cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC','cam_out_PRECC','cam_out_SOLS','cam_out_SOLL','cam_out_SOLSD','cam_out_SOLLD']

features60 = [] 
for f in feat60:
    features60 = features60 + [f+'_'+str(i) for i in range(60)]
allF = features60 + feat1

targets60 = [] 
for f in target60:
    targets60 = targets60 + [f+'_'+str(i) for i in range(60)]
allT = targets60 + target1

targetsToDrop12 = [ 'ptend_q0001', 'ptend_q0002', 'ptend_q0003', 'ptend_u', 'ptend_v']
dropT = ['ptend_q0002_12','ptend_q0002_13','ptend_q0002_14'] # attention, I think i also need to predict _15
for f in targetsToDrop12:
    dropT = dropT + [f+'_'+str(i) for i in range(12)]

allT2 = [i for i in allT if i not in dropT]

n60Feat = len(feat60)
n1dFeat = len(feat1)
n60Targ = len(target60)
n1dTarg = len(target1)

def addFeatures(a):
    addF = []
    colDict={}
    for i in range(60):
        newF = f'absWind_{i}'
        addF.append(newF)
        colDict[newF] =(a[f'state_u_{i}']**2 + a[f'state_v_{i}']**2)

        newF = f'angleWind_{i}'
        addF.append(newF)
        colDict[newF] =np.arctan2(a[f'state_u_{i}']**2,a[f'state_v_{i}']**2)
    a = pd.concat([a.reset_index(), pd.DataFrame(colDict).reset_index()], axis=1)

    colDict = {}
    for f in (feat60+['absWind']):
        for i in range(59):
            newF = 'diff'+f+'_'+str(i)
            addF.append(newF)
            colDict[newF] = (a[f+'_'+str(i+1)] - a[f+'_'+str(i)])
    a = pd.concat([a.reset_index(), pd.DataFrame(colDict).reset_index()], axis=1)
    return a, addF

def getTensorDataSeq(data, partPerLoop, startPartIdx,sampledPartIdx):
    X1d, X2d, y1d, y2d, X1dI, X2dI, y1dI,y2dI  = None, None, None, None, False, False, False, False
    for j in range(partPerLoop):
        a = data.get_partition(int(sampledPartIdx[startPartIdx+j])).compute()
        #a, newF = addFeatures(a)
        b = np.reshape(a[features60], (a.shape[0], n60Feat, 60))
        b = np.transpose(b, (0,2,1))
        X2d = np.concatenate([X2d,b], axis=0) if X2dI else b
        b = np.reshape(a[targets60], (a.shape[0], n60Targ, 60))
        b = np.transpose(b, (0,2,1))
        y2d = np.concatenate([y2d,b], axis=0) if y2dI else b
        X1d = np.concatenate([X1d,a[feat1]], axis=0) if X1dI else a[feat1]
        y1d = np.concatenate([y1d,a[target1]], axis=0) if y1dI else a[target1]
        X1dI, X2dI, y1dI,y2dI = True, True, True, True
    return X1d, X2d, y1d, y2d

def getTensorDataFlattend(data, partPerLoop, startPartIdx,sampledPartIdx):
    X, y, xi, yi = None, None, False, False
    for j in range(partPerLoop):
        a = data.get_partition(int(sampledPartIdx[startPartIdx+j])).compute()
        a, newF = addFeatures(a)
        allF = features60+newF+feat1
        b = a[allF]
        X = np.concatenate([X,b], axis=0) if xi else b
        allT = targets60+target1
        b = a[allT]
        y = np.concatenate([y,b], axis=0) if yi else b

        xi, yi = True, True
    return X,y, allF, allT