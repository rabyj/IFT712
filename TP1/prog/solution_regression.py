# -*- coding: utf-8 -*-

#####
# Joanny Raby (15062245)
# Dona Chadid ()
###

import numpy as np
import random
from sklearn import linear_model


class Regression:
    def __init__(self, lamb, m=1):
        self.lamb = lamb
        self.w = None
        self.M = m

    def fonction_base_polynomiale(self, x):
        """
        Fonction de base qui projette la donnee x vers un espace polynomial tel que mentionné au chapitre 3.
        --> Si x est un scalaire, alors phi_x sera un vecteur de longueur self.M + 1 (incluant le biais) :
        (1, x^1,x^2,...,x^self.M)
        --> Si x est un vecteur de N scalaires, alors phi_x sera un tableau 2D de taille Nx(M+1) (incluant le biais)

        NOTE : En mettant phi_x = x, on a une fonction de base lineaire qui fonctionne pour une regression lineaire
        """
        # on place un vecteur 1D ou un scalaire dans un tableau 2D
        try:
            if np.isscalar(x):
                phi = np.array([[x]])
            else:
                phi = x.reshape(-1, 1)
        except AttributeError:
            raise ValueError("L'entree doit etre un scalaire ou un vecteur 1D numpy.")

        # creation d'un tableau 2D de taille NxM+1
        for i in range(2, self.M + 1):
            phi = np.hstack((phi, (phi[:, 0] ** i).reshape((phi.shape[0], 1))))

        # ajout du biais
        phi = np.column_stack((np.full((phi.shape[0], 1), 1), phi))

        return phi

    def recherche_hyperparametre(self, X, t):
        """
        Trouver la meilleure valeur pour l'hyper-parametre self.M (pour un lambda fixe donné en entrée).

        Option 1
        Validation croisée de type "k-fold" avec k=10. La méthode array_split de numpy peut être utlisée
        pour diviser les données en "k" parties. Si le nombre de données en entrée N est plus petit que "k",
        k devient égal à N. Il est important de mélanger les données ("shuffle") avant de les sous-diviser
        en "k" parties.

        Option 2
        Sous-échantillonage aléatoire avec ratio 80:20 pour Dtrain et Dvalid, avec un nombre de répétition k=10.

        Note:

        Le resultat est mis dans la variable self.M

        X: vecteur de donnees
        t: vecteur de cibles
        """
        N = len(X)
        k = 10

        if N < k:
            raise AssertionError("Seulement {} valeurs pour validation-croisee {}-bloc. Impossible.".format(N, k))

        # On melange les indices de X/t pour une selection aleatoire des donnees
        melange = np.arange(0, N)
        np.random.shuffle(melange)

        # Liste de paires d'index permettant separation en k blocs
        selection = self.separation_k_blocs(N, k)

        # On scanne pour M = 1 à 10
        erreurs_moyenne = []
        for val_m in np.arange(1, 10, 1):
            self.M = val_m
            somme_err_valid = 0

            # On utilise chaque bloc séparément comme ensemble de validation
            for bloc in np.arange(0, k, 1):
                debut_valid, fin_valid = selection[bloc]

                # Listes des indices des donnees selectionnees
                train_indexes = np.concatenate((melange[0:debut_valid], melange[fin_valid:N]))
                valid_indexes = melange[debut_valid:fin_valid]

                # Entrainement puis validation
                self.entrainement(X[train_indexes], t[train_indexes])
                somme_err_valid += self.erreur(t[valid_indexes], self.prediction(X[valid_indexes]))

            erreurs_moyenne.append(somme_err_valid/k)

        # l'erreur moyenne minimale determine le meilleur M
        self.M = int(np.argmin(erreurs_moyenne) + 1)

    @staticmethod
    def separation_k_blocs(N, k):
        """Retourne la liste d'index permettant de separer un vecteur de taille N en k blocs."""
        # On va avoir 'N%k' blocs de taille 'N//k + 1' et
        # 'k - N%k' blocs de taille 'N//k'
        taille_bloc = N // k
        nb_blocs_diff = N % k

        # Creation de la liste de selection d'index (index_debut, index_fin)
        bloc_courant = 0
        debut = 0
        selection = []

        # blocs de taille 'N//k + 1'
        while bloc_courant < nb_blocs_diff:
            fin = debut + taille_bloc + 1
            selection.append((debut, fin))
            debut = fin
            bloc_courant += 1

        # blocs de taille 'N//k'
        while bloc_courant < k:
            fin = debut + taille_bloc
            selection.append((debut, fin))
            debut = fin
            bloc_courant += 1

        return selection

    def entrainement(self, X, t, using_sklearn=False):
        """
        Entraîne la regression lineaire sur l'ensemble d'entraînement forme des
        entrees ``X`` (un tableau 2D Numpy, ou la n-ieme rangee correspond à l'entree
        x_n) et des cibles ``t`` (un tableau 1D Numpy ou le
        n-ieme element correspond à la cible t_n). L'entraînement doit
        utiliser le poids de regularisation specifie par ``self.lamb``.

        Cette methode doit assigner le champs ``self.w`` au vecteur
        (tableau Numpy 1D) de taille M+1, tel que specifie à la section 3.1.4
        du livre de Bishop.

        Lorsque using_sklearn=True, vous devez utiliser la classe "Ridge" de
        la librairie sklearn (voir http://scikit-learn.org/stable/modules/linear_model.html)

        Lorsque using_sklearn=Fasle, vous devez implementer l'equation 3.28 du
        livre de Bishop. Il est suggere que le calcul de ``self.w`` n'utilise
        pas d'inversion de matrice, mais utilise plutôt une procedure
        de resolution de systeme d'equations lineaires (voir np.linalg.solve).

        Aussi, la variable membre self.M sert à projeter les variables X vers un espace polynomiale de degre M
        (voir fonction self.fonction_base_polynomiale())

        NOTE IMPORTANTE : lorsque self.M <= 0, il faut trouver la bonne valeur de self.M
        """
        if self.M <= 0:
            self.recherche_hyperparametre(X, t)

        phi_x = self.fonction_base_polynomiale(X)

        if using_sklearn:
            reg = linear_model.Ridge(alpha=self.lamb)
            fit = reg.fit(phi_x, t)
            self.w = fit.coef_
        else:
            # A w = B
            A = self.lamb * np.identity(phi_x.shape[1]) + np.dot(phi_x.T, phi_x)
            B = np.dot(phi_x.T, t)
            self.w = np.linalg.solve(A, B)

    def prediction(self, x):
        """
        Retourne la prediction de la regression lineaire
        pour une entree, representee par un tableau 1D Numpy ``x``.

        Cette methode suppose que la methode ``entrainement()``
        a prealablement ete appelee. Elle doit utiliser le champs ``self.w``
        afin de calculer la prediction y(x,w) (equation 3.1 et 3.3).
        """
        return np.sum(self.w * self.fonction_base_polynomiale(x))

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de la difference au carre entre
        la cible ``t`` et la prediction ``prediction``.
        """
        return np.sum((t-prediction)**2)
