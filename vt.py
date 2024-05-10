from sklearn.feature_selection import VarianceThreshold 
xt = [[0,1,0],
     [0,0,0],
     [1,0,1],
     [0,0,0],
     [0,1,0]]
selector= VarianceThreshold(threshold=0.2) 
selector.fit(xt) 
xt_red=selector.transform(xt) 
print('Original Dataset: ')
print(xt)
print('Reduced Dataset: ')
print(xt_red)
