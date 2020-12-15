# -*- coding: utf-8 -*-

#####
# Joanny Raby (15062245)
# Dona Chadid (20102835)
###

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class MAPnoyau:
    def __init__(self, lamb=0.2, sigma_square=1.06, b=1.0, c=0.1, d=1.0, M=2, noyau='rbf'):
        """
        Classe effectuant de la segmentation de données 2D 2 classes à l'aide de la méthode à noyau.
        lamb: coefficiant de régularisation L2
        sigma_square: paramètre du noyau rbf
        b, d: paramètres du noyau sigmoidal
        M,c: paramètres du noyau polynomial
        noyau: rbf, lineaire, polynomial ou sigmoidal
        """
        self.lamb = lamb
        self.a = None
        self.sigma_square = sigma_square
        self.M = M
        self.c = c
        self.b = b
        self.d = d
        self.noyau = noyau
        self.x_train = None

    def entrainement(self, x_train, t_train):
        """
        Entraîne une méthode d'apprentissage à noyau de type Maximum a
        posteriori (MAP) avec un terme d'attache aux données de type
        "moindre carrés" et un terme de lissage quadratique (voir
        Eq.(1.67) et Eq.(6.2) du livre de Bishop). La variable x_train
        contient les entrées (un tableau 2D Numpy, où la n-ième rangée
        correspond à l'entrée x_n) et des cibles t_train (un tableau 1D Numpy
        où le n-ième élément correspond à la cible t_n).
        L'entraînement doit utiliser un noyau de type RBF, lineaire, sigmoidal,
        ou polynomial (spécifié par ''self.noyau'') et dont les parametres
        sont contenus dans les variables self.sigma_square, self.c, self.b, self.d
        et self.M et un poids de régularisation spécifié par ``self.lamb``.
        Cette méthode doit assigner le champs ``self.a`` tel que spécifié à
        l'equation 6.8 du livre de Bishop et garder en mémoire les données
        d'apprentissage dans ``self.x_train``
        """
        I = np.identity(x_train.shape[0])
        self.x_train = x_train

        # generate Gram matrix
        K = self._apply_kernel(x_train, x_train)

        # a=(K + λI)−1 t
        self.a = np.dot(np.linalg.inv(K + self.lamb * I), t_train)

    def _apply_kernel(self, x, y):
        """Compute the kernel function on data arrays x and y. Only takes
        2D Numpy arrays. Returns the Gram matrix if x is y.
        To use for prediction, use x=self.x_train and y=np.array([x]),
        where x is only one data points.
        """
        if self.noyau == "lineaire": # 𝑘(𝑥,𝑥′)=𝑥.T 𝑥′
            K = x.dot(y.T)

        elif self.noyau == "polynomial": # 𝑘(𝑥,𝑥′)=(𝑥.T 𝑥′+𝑐)**𝑀
            K = (x.dot(y.T) + self.c) ** self.M

        elif self.noyau == "rbf": # 𝑘(𝑥,𝑥′)=exp(−‖𝑥−𝑥′‖**2 / 2 * 𝜎**2)
            # complete Gram matrix case (result has shape nxn)
            # using ||x-y||^2 = ||x||^2 + ||y||^2 - 2 * x^T * y
            # inspired by https://stackoverflow.com/questions/47271662/what-is-the-fastest-way-to-compute-an-rbf-kernel-in-python
            if x is y :
                x_norm = np.sum(x**2, axis=-1)
                K = np.exp(- (x_norm[:,None] + x_norm[None,:] - 2 * np.dot(x, x.T)) / (2 * self.sigma_square))
            # prediction case, k(y) (result has shape nx1)
            else:
                K = np.exp(-np.linalg.norm(x-y, axis=1)**2 / (2 * self.sigma_square))

        elif self.noyau == "sigmoidal" : # 𝑘(𝑥,𝑥′)=tanh(𝑏𝑥𝑇𝑥′+𝑑).
            K = np.tanh(self.b * x.dot(y.T) + self.d)

        else:
            raise ValueError("{} n'est pas un noyau valide. Voir l'aide.".format(self.noyau))

        return K

    def prediction(self, x):
        """
        Retourne la prédiction pour une entrée representée par un tableau
        1D Numpy ``x``.
        Cette méthode suppose que la méthode ``entrainement()`` a préalablement
        été appelée. Elle doit utiliser le champs ``self.a`` afin de calculer
        la prédiction y(x) (équation 6.9).
        NOTE : Puisque nous utilisons cette classe pour faire de la
        classification binaire, la prediction est +1 lorsque y(x)>0.5 et 0
        sinon
        """
        k = self._apply_kernel(self.x_train, np.array([x]))
        predict = np.dot(k.T, self.a)
        return int(predict > 0.5)

    def erreur(self, t, prediction):
        """
        Retourne la différence au carré entre
        la cible ``t`` et la prédiction ``prediction``.
        """
        return (t-prediction)**2

    def validation_croisee(self, x_tab, t_tab):
        """
        Cette fonction trouve les meilleurs hyperparametres ``self.sigma_square``,
        ``self.c`` et ``self.M`` (tout dépendant du noyau selectionné) et
        ``self.lamb`` avec une validation croisée de type "k-fold" où k=10 avec les
        données contenues dans x_tab et t_tab. Une fois les meilleurs hyperparamètres
        trouvés, le modèle est entraîné une dernière fois.
        SUGGESTION: Les valeurs de ``self.sigma_square`` et ``self.lamb`` à explorer vont
        de 0.000000001 à 2, les valeurs de ``self.c`` de 0 à 5, les valeurs
        de ''self.b'' et ''self.d'' de 0.00001 à 0.01 et ``self.M`` de 2 à 6
        """
        N = len(x_tab)
        k = 10

        if N < k:
            k = N

        # shuffle data
        rng_state = np.random.get_state()
        np.random.shuffle(x_tab)
        np.random.set_state(rng_state)
        np.random.shuffle(t_tab)

        # split data into k arrays
        X_split = np.array_split(x_tab, k)
        t_split = np.array_split(t_tab, k)

        # initialize lists for cross-validation
        params = []
        errors = []

        # cross-validate
        # search space size chosen to take max 1m with inputs train/test size=100
        # search space can be easily modified with "num" linspace parameter
        print("Validation croisée {}-blocs...".format(k))
        if self.noyau == "lineaire": # lamb
            for l in tqdm(np.linspace(1e-09, 2, num=50)):
                params.append(l)
                self.lamb = l
                errors.append(self._split_validate(X_split, t_split, k))

            self.lamb = params[int(np.argmin(errors))]


        elif self.noyau == "polynomial": # 𝑐 & 𝑀 & lamb

            for l in tqdm(np.linspace(1e-09, 2, num=17)):
                for c in np.linspace(0, 5, num=17):
                    for m in range(2, 7, 1):
                        params.append((l, c, m))
                        self.lamb, self.c, self.M = (l, c, m)
                        errors.append(self._split_validate(X_split, t_split, k))

            self.lamb, self.c, self.M = params[int(np.argmin(errors))]


        elif self.noyau == "rbf": # sigma_square & lamb

            for l in tqdm(np.linspace(1e-09, 2, num=20)):
                for sigma in np.linspace(1e-09, 2, num=50):

                    params.append((l, sigma))
                    self.lamb, self.sigma_square = (l, sigma)
                    errors.append(self._split_validate(X_split, t_split, k))

            self.lamb, self.sigma_square = params[int(np.argmin(errors))]


        elif self.noyau == "sigmoidal": # b & d & lamb

            for l in tqdm(np.linspace(1e-09, 2, num=10)):
                for b in np.linspace(1e-05, 0.01, num=15):
                    for d in np.linspace(1e-05, 0.01, num=15):

                        params.append((l, b, d))
                        self.lamb, self.b, self.d = (l, b, d)
                        errors.append(self._split_validate(X_split, t_split, k))

            self.lamb, self.b, self.d = params[int(np.argmin(errors))]


        else:
            raise ValueError("{} n'est pas un noyau valide. Voir l'aide.".format(self.noyau))

        # train data
        print("{} configurations d'hyperparamètres testées.".format(len(params)))
        self.entrainement(x_tab, t_tab)

    def _split_validate(self, X_split, t_split, k):
        """Returns sum of errors on k-fold cross-validation."""
        somme_err_valid = 0
        for bloc in np.arange(0, k, 1):

            X_train = np.concatenate((X_split[0:np.int(bloc)] + X_split[np.int(bloc+1):np.int(k)]))
            X_valid = X_split[np.int(bloc)]

            t_train = np.concatenate((t_split[0:np.int(bloc)] + t_split[np.int(bloc+1):np.int(k)]))
            t_valid = t_split[np.int(bloc)]

            # train & validate
            self.entrainement(X_train, t_train)
            somme_err_valid += np.sum(self.erreur(t_valid, [self.prediction(x) for x in X_valid]))

        return somme_err_valid

    def affichage(self, x_tab, t_tab):

        # Affichage
        ix = np.arange(x_tab[:, 0].min(), x_tab[:, 0].max(), 0.1)
        iy = np.arange(x_tab[:, 1].min(), x_tab[:, 1].max(), 0.1)
        iX, iY = np.meshgrid(ix, iy)
        x_vis = np.hstack([iX.reshape((-1, 1)), iY.reshape((-1, 1))])
        contour_out = np.array([self.prediction(x) for x in x_vis])
        contour_out = contour_out.reshape(iX.shape)

        plt.contourf(iX, iY, contour_out > 0.5)
        plt.scatter(x_tab[:, 0], x_tab[:, 1], s=(t_tab + 0.5) * 100, c=t_tab, edgecolors='y')
        plt.show()
