# reticulate

# Loads Python Shell
repl_python()

# Check the current Python version
reticulate::py_config()

# import matplotlib.pyplot as plt
plt <- import('matplotlib.pyplot') 

# from sklearn.neighbors.kde import KernelDensity
KernelDensity = import('sklearn.neighbors.kde')

#from scipy.signal import parzen 
parzen = import('scipy.signal')

# from sklearn.model_selection import ParameterGrid
ParameterGrid = import('sklearn.model_selection')
'''{python}
'''

param_grid = {'bandwidth': [0.1,0.125,0.15,0.25,0.5,1,2,3]}

kde = KernelDensity(kernel='gaussian')

clf = GridSearchCV(kde, param_grid)

clf.fit(X_val_norm,np.ravel(y_val))

# GridSearchCV(cv=None, error_score=nan,
#              estimator=KernelDensity(algorithm='auto', atol=0, bandwidth=1.0,
#                                      breadth_first=True, kernel='gaussian',
#                                      leaf_size=40, metric='euclidean',
#                                      metric_params=None, rtol=0),
#              iid='deprecated', n_jobs=None,
#              param_grid=[{'bandwidth': [0.1, 0.125, 0.15, 0.25, 0.5, 1, 2, 3]}],
#              pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
#              scoring=None, verbose=0)

h = clf.best_params_['bandwidth']

clf.cv_results_['mean_test_score']