# Representations abstraites
import matplotlib.pyplot as plt
import numpy as np
import pickle
import params

# # Ouverture des listes
# with open ('liste1', 'rb') as fp:
#     l1x = pickle.load(fp)
# with open ('liste2', 'rb') as fp:
#     l1y = pickle.load(fp)
# with open ('liste1v', 'rb') as fp:
#     l2x = pickle.load(fp)
# with open ('liste2v', 'rb') as fp:
#     l2y = pickle.load(fp)

#Affichage
def Affichage(l1x,l1y,l2x,l2y):
    color = params.color_abstr

    plt.figure()
    ax = plt.subplot()

    plt.plot(l1x, l1y, 'b'+'--')
    plt.plot(l1x, l1y, 'b'+'o',label = 'Piano')
    for i in range(len(l1x)):
        ax.annotate(' {}'.format(i+1), (l1x[i], l1y[i]), color='black')
    plt.plot(l2x, l2y, 'r'+'--')
    plt.plot(l2x, l2y, 'r'+'o', label = 'Violon')
    for i in range(len(l2x)):
        ax.annotate(' {}'.format(i+1), (l2x[i], l2y[i]), color='black')


    d1, d2 = 'diffConcordance', 'crossConcordance'
    plt.xlabel(d1[0].upper() + d1[1:])
    plt.ylabel(d2[0].upper() + d2[1:])
    plt.title('Cadence ' + ' (' + d1[0].upper() + d1[1:] + ', ' + d2[0].upper() + d2[1:] + ')')
    plt.legend(frameon=True, framealpha=0.75)
    plt.show()


pts1 = [np.array((l1x[t],l1y[t])) for t in range(len(l1x))]
pts2 = [np.array((l2x[t],l2y[t])) for t in range(len(l1x))]

# #distance euclidienne
# def dist(x,y):
#     return np.sqrt(np.sum((x-y)**2))
#
#
# def distance(pts1,pts2,type = 'diff'):
#     distance = 0
#     if type == 'stat':
#         for t in range(len(pts1)):
#             distance += dist(pts1[t], pts2[t])
#         return distance
#     else :
#         pts1_diff = [pts1[t+1]-pts1[t] for t in range(len(pts1)-1)]
#         pts2_diff = [pts2[t+1]-pts2[t] for t in range(len(pts2)-1)]
#         for t in range(len(pts1_diff)):
#             distance += dist(pts1_diff[t], pts2_diff[t])
#         return distance


# print(distance(pts1,pts2,'stat'))

points = np.asarray([pts1, pts2])

# Fonction qui calcule l'éloignement de courbes nomalisées, correspondant à différents timbres 
def dispersion(points,type = 'diff'):
    if type == 'stat':
        return np.linalg.norm(np.std(points,axis = 0), axis = 1)
    else :
        points_diff = np.zeros((points.shape[0],points.shape[1]-1,points.shape[2]))
        for i in range(points.shape[1]-1):
            points_diff[:,i] = points[:,i+1]-points[:,i]
        return np.linalg.norm(np.std(points_diff,axis = 0), axis = 1)



print(dispersion(points))





# Affichage(l1x,l1y,l2x,l2y)
