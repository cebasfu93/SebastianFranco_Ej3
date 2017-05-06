import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from scipy import optimize

thres=5
N_att=100000
N_lig=800
X_temp=np.zeros((N_lig,N_att))
Y=np.genfromtxt('./dorothea_train.labels')
Y[Y==-1]=0

i = 0
output = {}
with open('./dorothea_train.data','r') as input_file:
    for line in input_file:
        line_1=line.strip(' \n').split(' ')
        output[i] = map(int,line_1)
        i+=1

for i in range(N_lig):
    for j in output[i]:
        X_temp[i,j-1]=1

count=np.zeros(N_att)
for i in range(N_att):
    count[i]=np.sum(X_temp[:,i])

orde=np.sort(count)
ndxs=np.where(count>thres)[0]
l_ndxs=len(ndxs)

X=np.zeros((N_lig,l_ndxs))

for i in range(l_ndxs):
    X[:,i]=X_temp[:,ndxs[i]]


X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size=0.80)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, train_size=0.80)

#print "Activos en el test set=",len(Y_test[Y_test==1])

def forest(n_trees, prof, feat):
    rf = RandomForestClassifier(n_estimators=n_trees, max_depth=prof, max_features=feat)
    rf.fit(X_train, Y_train)

    y_predict = rf.predict(X_test)
    matrix=confusion_matrix(Y_test, y_predict)
    return (float(matrix[0,0]+matrix[1,1]))/np.sum(matrix)

arboles=np.arange(1, 300, 20)

fracs_tree=np.zeros(len(arboles))
for i in range(len(arboles)):
    fracs_tree[i]=forest(arboles[i], None, 'auto')

fig=plt.figure()
plt.xlabel("Numero de arboles")
plt.ylabel("Fraccion de aciertos")
plt.title("Dependencia de la fraccion de aciertos con numero de arboles")
plt.plot(arboles, fracs_tree)
plt.savefig("num_trees.pdf", format='pdf')
plt.close()

max_tree=np.max(fracs_tree)
pos_max_tree=np.where(fracs_tree==max_tree)[0][0]

best_tree=arboles[pos_max_tree]
print "El numero de arboles optimo es: ", best_tree

depths=np.arange(1, 80, 8)

fracs_depths=np.zeros(len(depths))
for i in range(len(depths)):
    fracs_depths[i]= forest(best_tree, depths[i], 'auto')

fig=plt.figure()
plt.xlabel("Profundidad de arboles")
plt.ylabel("Fraccion de aciertos")
plt.title("Dependencia de la fraccion de aciertos con profundidad de arboles")
plt.plot(depths, fracs_depths)
plt.savefig("depths.pdf", format='pdf')
plt.close()

max_depth=np.max(fracs_depths)
pos_max_depth=np.where(fracs_depths==max_depth)[0][0]
best_depth=depths[pos_max_depth]
print "La profundidad de arbol optima es: ", best_depth

atrib=np.arange(1, 200, 10)

fracs_atrib=np.zeros(len(atrib))
for i in range(len(atrib)):
    fracs_atrib[i]= forest(best_tree, best_depth, atrib[i])

fig=plt.figure()
plt.xlabel("Numero de atributos")
plt.ylabel("Fraccion de aciertos")
plt.title("Dependencia de la fraccion de aciertos con numero de atributos")
plt.plot(atrib, fracs_atrib)
plt.savefig("atrib.pdf", format='pdf')
plt.close()

max_atrib=np.max(fracs_atrib)
pos_max_atrib=np.where(fracs_atrib==max_atrib)[0][0]
best_atrib=atrib[pos_max_atrib]
print "El numero de atributos optimo es: ", best_atrib

rf = RandomForestClassifier(n_estimators=best_tree, max_depth=best_depth, max_features=best_atrib)
rf.fit(X_train, Y_train)

y_predict = rf.predict(X_val)
matrix=confusion_matrix(Y_val, y_predict)
frac_final=(float(matrix[0,0]+matrix[1,1]))/np.sum(matrix)
bono=(frac_final-0.5)*40
print "La fraccion de aciertos para el conjunto de validacion es: ", frac_final
print "Entonces B=", bono

atributo=[]
importancia=[]
ii = np.argsort(rf.feature_importances_)
for i in range(1,21):
    atributo.append(ii[-1*i])
    importancia.append(rf.feature_importances_[ii[-1*i]])
    print "Atributo numero: ", ii[-1*i], " tiene importancia relativa con valor de: ", rf.feature_importances_[ii[-1*i]]

fig=plt.figure()
plt.xlabel('Atributo')
plt.ylabel('Importancia relativa')
plt.title("Atributos mas importantes y su importancia relativa")
plt.plot(atributo, importancia, 'bo', ms=5, label="Fraccion de aciertos con validacion: "+str(frac_final))
plt.legend()
plt.savefig('Importancia.pdf', format='pdf')
plt.close()
