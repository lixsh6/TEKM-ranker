import matplotlib
matplotlib.use('Agg')
import cPickle
import numpy as np
import random,tqdm
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import Counter
(word_embeddings,ent_embeddings,topic_embeddings) = cPickle.load(open('./w_e_t_embedding.pkl'))

random.seed(13)
np.random.seed(13)

cluster_num = 2
select_num = cluster_num
kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(topic_embeddings)
topic_embeddings = kmeans.cluster_centers_


mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
print 'mycolors: ', len(mycolors)

word_embeddings = word_embeddings[:1000]
train_data = np.concatenate((word_embeddings,topic_embeddings),axis=0)
tsne_model = PCA(n_components=50)
tsne_model.fit(train_data)

def draw_topic_points():

    tsne_X = tsne_model.transform(topic_embeddings)
    fig = plt.figure()
    plt.scatter(tsne_X[:, 0], tsne_X[:, 1],marker='o', s=10, cmap=plt.cm.Spectral)
    fig.savefig('topics.png', dpi=fig.dpi)

draw_topic_points()
#exit(-1)


def get_class(embedding):
    distances = []
    for i in range(topic_embeddings.shape[0]):
        dis = np.sqrt(np.sum(np.asarray(embedding - topic_embeddings[i]) ** 2))
        #dis = np.linalg.norm(embedding-topic_embeddings[i])
        distances.append(dis)
    print 'distance: ', distances
    class_index = np.argmin(distances)
    class_dis = np.min(distances)
    print class_index
    return (class_index,class_dis)


print 'getting word classes...'
word_classes = [get_class(w_embedding) for w_embedding in tqdm.tqdm(word_embeddings)]

print Counter(map(lambda t:t[0],word_classes))
#print 'word_classes: ', word_classes
print 'finished get'

class_thres = 5
select_classes = random.sample(range(topic_embeddings.shape[0]), select_num) #select 20 topics

filtered_word_embeddings = [word_embeddings[i] for i,(class_index,class_dis) in enumerate(word_classes) if (class_index in select_classes and class_dis < class_thres)]
filtered_classes = [class_index for (class_index,class_dis) in word_classes if (class_index in select_classes and class_dis < class_thres)]

print 'Fitting TSNE...'
#tsne_model = TSNE(n_components=50,random_state=101,init='pca')
#tsne_model = PCA(n_components=50)
print 'filtered_word_embeddings: ', len(filtered_word_embeddings)
#print np.array(filtered_word_embeddings)
tsne_X = tsne_model.transform(np.array(filtered_word_embeddings))
print 'Done'


class_index_map_dict = {}
for c in filtered_classes:
    if c not in class_index_map_dict:
        class_index_map_dict[c] = len(class_index_map_dict)
#filter_points_X = [tsne_X[i][0] for i,class_index in enumerate(word_classes) if class_index in select_classes]
#filter_points_Y = [tsne_X[i][1] for i,class_index in enumerate(word_classes) if class_index in select_classes]

fig = plt.figure()
plt.scatter(tsne_X[:,0], tsne_X[:,1], c=mycolors[map(lambda c:class_index_map_dict[c],filtered_classes)],\
            marker = 'o',s=1, cmap=plt.cm.Spectral) #s=10
#for i in range(len(tsne_X)):
#    plt.annotate("p%d" % (i+1), (tsne_X[i][0],tsne_X[i][1]))
#plt.show()
x_range = 2.5
#plt.xlim(-x_range, x_range)

y_range = 2
#plt.ylim(-y_range, y_range)
fig.savefig('word.png',dpi=fig.dpi)

