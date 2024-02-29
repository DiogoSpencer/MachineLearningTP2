import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from tp2_aux import images_as_matrix
from tp2_aux import report_clusters
from tp2_aux import report_clusters_hierarchical
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.cluster import AffinityPropagation

NUM_OF_FEATURES_TO_EXTRACT = 6
VERBOSE = False

## Extract 18 features from the 2500 initial
def extractFeatures():
    dataset = images_as_matrix()
    t_data_pca = PCA(n_components= NUM_OF_FEATURES_TO_EXTRACT).fit(dataset).transform(dataset)
    t_data_tsne = TSNE(n_components= NUM_OF_FEATURES_TO_EXTRACT, method='exact').fit_transform(dataset)
    t_data_isomap = Isomap(n_components= NUM_OF_FEATURES_TO_EXTRACT).fit_transform(dataset)
    t_data = np.hstack((np.hstack((t_data_pca,t_data_tsne)),t_data_isomap))
    np.save("t_data",t_data)
    return t_data

## Standertize the data
def standertize(Xs):
    Xsmeans = np.mean(Xs, axis = 0)
    Xsstdevs = np.std(Xs, axis = 0)
    res = (Xs-Xsmeans) / Xsstdevs    
    return res

## Get the metrics to evaluate the clustering algorithm
def evaluate(X,Y, Y_labeled_pred, Y_true):
    if(VERBOSE):
        print(metrics.confusion_matrix(Y_true, Y_labeled_pred))
        print(metrics.classification_report(Y_true, Y_labeled_pred, digits=3))
        
    adjusted_rand_score_val = 0
    silhouette_score_val = 0
    num_of_labels = len(np.unique(Y_labeled_pred, return_counts=True)[0])
    if(num_of_labels > 1):
        silhouette_score_val = silhouette_score(X,Y)
        adjusted_rand_score_val = adjusted_rand_score(Y_true, Y_labeled_pred)
    precision = precision_score(Y_true, Y_labeled_pred, average='micro')
    recall = recall_score(Y_true, Y_labeled_pred, average='micro')
    f1 = f1_score(Y_true, Y_labeled_pred, average='micro')
    return silhouette_score_val, adjusted_rand_score_val, precision, recall, f1

## Fit the nº of clusters for KMEANS
def fitKMeans():
    k = range(2,20)
    precision_arr = []
    recall_arr = []
    f1_arr = []
    silhouette_arr = []
    rand_index_arr = []
    for i in k:
        kmeans = KMeans(n_clusters= i).fit(t_data)
        labels = kmeans.predict(t_data)
        labels_annotated = np.array([labels[int(i)] for i in Y_labeled[:,0]])
        silhouette_score_val, adjusted_rand_score_val, precision, recall, f1 = evaluate(t_data,labels, labels_annotated, Y_labeled[:,1])
        precision_arr.append(precision)
        recall_arr.append(recall)
        f1_arr.append(f1)
        silhouette_arr.append(silhouette_score_val)
        rand_index_arr.append(adjusted_rand_score_val)
        if(VERBOSE):
            print("KMeans nº clusters = ", i)
            print("\tsilhouette_score = ", silhouette_score_val)  
            print("\tadjusted_rand_score = ", adjusted_rand_score_val)
            print("\tprecision = ", precision)
            print("\trecall = ", recall)
            print("\tf1 = ", f1)
    plt.title("KMEANS")
    plt.plot(k, precision_arr, label="Precision")
    plt.plot(k,recall_arr, label="Recall")
    plt.plot(k,f1_arr, label="F1")
    plt.plot(k,silhouette_arr, label="Silhouette")
    plt.plot(k,rand_index_arr, label="Rand Index")
    plt.xscale("linear")
    plt.xticks(k)
    plt.legend()
    plt.savefig("KMEANSfit.png")
    plt.show()

## Plot the distance between points in decrescent order and get a interval
def getBestEpsIntrevalWithPlot():
    Y_aux = np.zeros(t_data.shape[0])
    neigh = KNeighborsClassifier().fit(t_data, Y_aux)
    distances, idx = neigh.kneighbors()
    distances = distances[:,4]
    distances = np.sort(distances)[::-1]
    plt.plot(distances)
    plt.ylabel("Distance to the 5 neighbors")
    plt.show()
    
def fitGM():
    k = range(2,20)
    precision_arr = []
    recall_arr = []
    f1_arr = []
    silhouette_arr = []
    rand_index_arr = []
    for i in k:
        gmm = GaussianMixture(n_components=i, covariance_type='spherical').fit(t_data)
        labels = gmm.predict(t_data)
        labels_annotated = np.array([labels[int(i)] for i in Y_labeled[:,0]])
        silhouette_score_val, adjusted_rand_score_val, precision, recall, f1 = evaluate(t_data,labels, labels_annotated, Y_labeled[:,1])
        precision_arr.append(precision)
        recall_arr.append(recall)
        f1_arr.append(f1)
        silhouette_arr.append(silhouette_score_val)
        rand_index_arr.append(adjusted_rand_score_val)
        if(VERBOSE):
            print("Gaussian Mixture nº components = ", i)
            print("\tsilhouette_score = ", silhouette_score_val)  
            print("\tadjusted_rand_score = ", adjusted_rand_score_val)
            print("\tprecision = ", precision)
            print("\trecall = ", recall)
            print("\tf1 = ", f1)
    plt.title("GM")
    plt.plot(k, precision_arr, label="Precision")
    plt.plot(k,recall_arr, label="Recall")
    plt.plot(k,f1_arr, label="F1")
    plt.plot(k,silhouette_arr, label="Silhouette")
    plt.plot(k,rand_index_arr, label="Rand Index")
    plt.xscale("linear")
    plt.xticks(k)
    plt.legend()
    plt.savefig("GMfit.png")
    plt.show()    
    
def fitDBSCAN():
    print("Enter the eps interval")
    lower_eps = float(input("Lower bound ?"))
    higher_eps = float(input("Upper bound ?"))
    step = float(input("Step ?"))
    eps = np.arange(lower_eps, higher_eps, step)
    precision_arr = []
    recall_arr = []
    f1_arr = []
    silhouette_arr = []
    rand_index_arr = []
    for e in eps:
        labels = DBSCAN(eps=e).fit_predict(t_data)
        labels_annotated = np.array([labels[int(i)] for i in Y_labeled[:,0]])
        silhouette_score_val, adjusted_rand_score_val, precision, recall, f1 = evaluate(t_data,labels, labels_annotated, Y_labeled[:,1])
        precision_arr.append(precision)
        recall_arr.append(recall)
        f1_arr.append(f1)
        silhouette_arr.append(silhouette_score_val)
        rand_index_arr.append(adjusted_rand_score_val)
        if(VERBOSE):
            print("DBSCAN WITH EPS = ", e)
            print("\tsilhouette_score = ", silhouette_score_val)  
            print("\tadjusted_rand_score = ", adjusted_rand_score_val)
            print("\tprecision = ", precision)
            print("\trecall = ", recall)
            print("\tf1 = ", f1)
    plt.title("DBSCAN")
    plt.plot(eps, precision_arr, label="Precision")
    plt.plot(eps,recall_arr, label="Recall")
    plt.plot(eps,f1_arr, label="F1")
    plt.plot(eps,silhouette_arr, label="Silhouette")
    plt.plot(eps,rand_index_arr, label="Rand Index")
    plt.xlabel("Eps Value")
    plt.legend()
    plt.savefig("DBSCANfit.png")
    plt.show()        
    
def fitAffinityProp():
    precision_arr = []
    recall_arr = []
    f1_arr = []
    silhouette_arr = []
    rand_index_arr = []
    damp = np.arange(0.5, 1, 0.05)
    for d in damp:
        affinityProp = AffinityPropagation(damping=d).fit(t_data)
        labels = affinityProp.predict(t_data)
        labels_annotated = np.array([labels[int(i)] for i in Y_labeled[:,0]])
        silhouette_score_val, adjusted_rand_score_val, precision, recall, f1 = evaluate(t_data,labels, labels_annotated, Y_labeled[:,1])
        precision_arr.append(precision)
        recall_arr.append(recall)
        f1_arr.append(f1)
        silhouette_arr.append(silhouette_score_val)
        rand_index_arr.append(adjusted_rand_score_val)
        if(VERBOSE):
            print("\tsilhouette_score = ", silhouette_score_val)  
            print("\tadjusted_rand_score = ", adjusted_rand_score_val)
            print("\tprecision = ", precision)
            print("\trecall = ", recall)
            print("\tf1 = ", f1)
    plt.title("AffinityProp")
    plt.plot(damp, precision_arr, label="Precision")
    plt.plot(damp,recall_arr, label="Recall")
    plt.plot(damp,f1_arr, label="F1")
    plt.plot(damp,silhouette_arr, label="Silhouette")
    plt.plot(damp,rand_index_arr, label="Rand Index")
    plt.xlabel("Eps Value")
    plt.legend()
    plt.savefig("DBSCANfit.png")
    plt.show()
    
    
def fitBissectingKMeans(num_iterations):
    kmeans = KMeans(n_clusters= 1).fit(t_data)
    labels_kmeans = kmeans.predict(t_data)
    res = [None] * t_data.shape[0]
    for i in range(num_iterations):
        uniques = np.unique(labels_kmeans, return_counts=True)
        bigger_cluster = uniques[0][np.argmax(uniques[1])]
        idxs = np.where(labels_kmeans == bigger_cluster)
        for idx in idxs[0]:
            if(res[idx] != None):
                res[idx].insert(len(res[idx]), bigger_cluster)                
            else:
                res[idx] = [bigger_cluster,]
        new_data = t_data[idxs]
        bisKMeans = KMeans(n_clusters= 2).fit(new_data)
        labels_kmeans = bisKMeans.predict(new_data)
    return res
    
# If not have the t_data.npy file uncomment next line to extract features to file
#t_data = extractFeatures() 
t_data = np.load('t_data.npy')
print(t_data)
t_data = standertize(t_data)
print(t_data)

## Load the labeled data to choose the best features
Y = np.loadtxt("labels.txt", delimiter = ',')
Y_labeled = Y[Y[:,1] != 0]
labeled_data = np.array([t_data[int(i)] for i in Y_labeled[:,0]])

## Observe the f-value to choose in the 18 features the most independent ones
f, prob = f_classif(labeled_data, Y_labeled[:,1])
print(f)
print(prob)

## Fix the number of relevant features observed and transform the data
kbest = SelectKBest(f_classif, k=int(input("Number of features to select?")))
X_new = kbest.fit_transform(labeled_data, Y_labeled[:,1])
X_idx = kbest.get_support()
t_data = t_data[:,X_idx]


#fitKMeans()
#getBestEpsIntrevalWithPlot()
#fitDBSCAN()
#fitGM()
#fitAffinityProp()

#while(True):
#    n_cluster = int(input("Final best number of clusters for KMEAN?"))
#    kmeans = KMeans(n_clusters= n_cluster).fit(t_data)
#    labels_kmeans = kmeans.predict(t_data)
#    report_clusters(Y[:,0], labels_kmeans, "KMEANS-" + str(n_cluster) + ".html")
#    
#    n_eps = float(input("Final best number of eps for DBSCAN?"))
#    labels_dbscan = DBSCAN(eps=n_eps).fit_predict(t_data)
#    report_clusters(Y[:,0], labels_dbscan, "DBSCAN-" + str(n_eps) + ".html")
#    
#    n_comp = int(input("Final best number of components for Gaussian Mixture"))
#    gm = GaussianMixture(n_components=n_comp, covariance_type='spherical').fit(t_data)
#    labels_gm = gm.predict(t_data)
#    report_clusters(Y[:,0], labels_gm, "GM-" + str(n_comp) + ".html")
    
#   damp = float(input("Final best number of damping for Affinity Propagation?"))
#   affinityProp = AffinityPropagation(damping=damp, preference=-200).fit(t_data)
#   labels_aff = affinityProp.predict(t_data)
#   report_clusters(Y[:,0],labels_aff,"AffinityPropagation" + str(damp) + ".html")

#for i in range(2,10):
#    labels_list = fitBissectingKMeans(i)
#    report_clusters_hierarchical(Y[:,0],labels_list,"bisecting_test_" + str(i) +"_iterations.html")


