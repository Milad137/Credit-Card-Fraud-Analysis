
import pandas as pd
from sklearn.model_selection import GridSearchCV
import numpy as np

def ModelTable(clf, grid, DataSets, Metrics, MetricsDic, Names):
    PT={}
    CLF={}

    Params= list(grid.keys())
    Params.remove('random_state')

    param_Params = ['param_'+p for p in Params]
    ParamsDic= dict(zip(param_Params,Params))

    for Name in Names:

        X_train=DataSets[Name][0]
        y_train=DataSets[Name][1]
        X_test=DataSets[Name][2]
        y_test=DataSets[Name][3]

        
        clf_cv=GridSearchCV(clf, grid, cv=5, scoring=["roc_auc", "f1"], refit="roc_auc", n_jobs=4)
        clf_cv.fit(X_train, y_train)
        PerfTable=dict((k, clf_cv.cv_results_[k]) for k in param_Params+Metrics)
        PerfTable=pd.DataFrame(PerfTable).apply(pd.to_numeric, errors='ignore')
        PerfTable.rename(columns={**ParamsDic, **MetricsDic}, inplace=True)
        
        PT[Name]=PerfTable
        CLF[Name]=clf_cv

    PT['Parameters'] = PerfTable[Params]
    PT=pd.concat(PT, axis=1)

    for name in Names:
        for param in Params:
            PT.drop((name, param), axis = 1,inplace=True)
            
    PT=PT[['Parameters']+Names]

    return {'Table':PT, 'Model':CLF}

def return_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection

def hist_intersection(df, target, bins):
    Classes = df[target].unique()
    HI=pd.DataFrame(columns=df.drop(columns=target).columns,index=[0])
    for name in df.drop(columns=target).columns:
        series1=df[df[target]==Classes[0]][name]
        series2=df[df[target]==Classes[1]][name]
        hist_1, _ = np.histogram(series1, density=True, bins=bins)
        hist_2, _ = np.histogram(series2, density=True, bins=bins)
        HI.loc[0,name]=return_intersection(hist_1, hist_2)
    return HI
