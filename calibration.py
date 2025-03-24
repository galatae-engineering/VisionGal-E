import cv2
import numpy as np
import glob
import os
import time
from datetime import datetime


def calibrer_camera():
    # Paramètres de la caméra HBV-W202012HD
    # Résolution 1280x720, capteur OV9726
    largeur_camera = 1280
    hauteur_camera = 720

    print("=== Programme de calibration pour caméra HBV-W202012HD ===")
    print("Modèle: HBV-W202012HD")
    print("Capteur: OV9726 (1/6\")")
    print("Résolution: 1280 x 720")
    print("Champ de vision: 50°, Focale: 3,2 mm")

    # Créer un répertoire pour sauvegarder les images de calibration
    dossier_images = 'images_calibration'
    if not os.path.exists(dossier_images):
        os.makedirs(dossier_images)
        print(f"Dossier '{dossier_images}' créé pour stocker les images de calibration")

    # Définir les paramètres d'échiquier de calibration
    nb_coins_x = 9  # Nombre de coins intérieurs en largeur
    nb_coins_y = 6  # Nombre de coins intérieurs en hauteur
    taille_damier = nb_coins_x * nb_coins_y

    # Préparer les points 3D de l'échiquier (0,0,0), (1,0,0), (2,0,0), ...
    # Supposons que la taille d'un carré est de 1 unité
    points_objet = np.zeros((taille_damier, 3), np.float32)
    points_objet[:, :2] = np.mgrid[0:nb_coins_x, 0:nb_coins_y].T.reshape(-1, 2)

    # Arrays pour stocker les points d'objet et d'image de toutes les images
    points_objets = []  # points 3D dans l'espace réel
    points_images = []  # points 2D dans le plan image

    # Initialiser la caméra
    print("Initialisation de la caméra...")
    cap = cv2.VideoCapture(0)  # Utiliser 0 pour la caméra par défaut, ajuster si nécessaire

    # Vérifier si la caméra est ouverte
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra")
        return

    # Configurer la résolution de la caméra
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, largeur_camera)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hauteur_camera)

    print("\nInstructions:")
    print("1. Placez un échiquier de calibration devant la caméra")
    print("2. Appuyez sur 'c' pour capturer une image quand l'échiquier est détecté")
    print("3. Capturez au moins 10 images de l'échiquier dans différentes positions")
    print("4. Appuyez sur 'q' pour quitter le mode capture et lancer la calibration")

    nb_images = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur lors de la capture de l'image")
            break

        # Afficher le frame
        affichage = frame.copy()
        cv2.putText(affichage, f"Images capturees: {nb_images}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(affichage, "Appuyez sur 'c' pour capturer, 'q' pour quitter",
                    (10, hauteur_camera - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Convertir en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Chercher l'échiquier
        ret_corners, corners = cv2.findChessboardCorners(gray, (nb_coins_x, nb_coins_y), None)

        # Si trouvé, ajouter au points d'objet et raffiner les points d'image
        if ret_corners:
            cv2.putText(affichage, "Echiquier detecte", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Dessiner et afficher les coins
            cv2.drawChessboardCorners(affichage, (nb_coins_x, nb_coins_y), corners, ret_corners)

        cv2.imshow('Calibration Camera HBV-W202012HD', affichage)

        # Attendre les interactions clavier
        key = cv2.waitKey(1) & 0xFF

        # Si l'utilisateur appuie sur 'c' et que l'échiquier est détecté
        if key == ord('c') and ret_corners:
            # Raffiner la détection des coins
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Ajouter les points
            points_objets.append(points_objet)
            points_images.append(corners2)

            # Sauvegarder l'image pour référence
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nom_fichier = os.path.join(dossier_images, f'calibration_{timestamp}.jpg')
            cv2.imwrite(nom_fichier, frame)

            nb_images += 1
            print(f"Image {nb_images} capturée")
            time.sleep(1)  # Attendre un peu pour éviter les captures multiples accidentelles

        # Si l'utilisateur appuie sur 'q' ou ESC
        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if nb_images < 5:
        print("Pas assez d'images capturées pour une calibration fiable (minimum 5)")
        return

    print(f"\nCalibration en cours avec {nb_images} images...")

    # Effectuer la calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        points_objets, points_images, gray.shape[::-1], None, None)

    # Calculer l'erreur de re-projection
    erreur_totale = 0
    for i in range(len(points_objets)):
        imgpoints2, _ = cv2.projectPoints(points_objets[i], rvecs[i], tvecs[i], mtx, dist)
        erreur = cv2.norm(points_images[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        erreur_totale += erreur

    print(f"Erreur de re-projection moyenne: {erreur_totale / len(points_objets)}")

    # Sauvegarder les paramètres de calibration
    np.savez('calibration_camera_HBV-W202012HD.npz',
             mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

    print("\nParamètres de calibration:")
    print("Matrice de la caméra:")
    print(mtx)
    print("\nCoefficients de distorsion:")
    print(dist)

    print("\nPour utiliser la calibration dans vos applications:")
    print("1. Chargez les paramètres avec: data = np.load('calibration_camera_HBV-W202012HD.npz')")
    print("2. Extrayez la matrice et les coefficients: mtx = data['mtx'], dist = data['dist']")
    print("3. Utilisez cv2.undistort(image, mtx, dist) pour corriger la distorsion d'image")

    # Test optionnel avec une nouvelle capture
    tester_calibration(mtx, dist)


def tester_calibration(mtx, dist):
    print("\nTest de la calibration avec une nouvelle image...")
    print("Appuyez sur n'importe quelle touche pour continuer, ou 'q' pour quitter")
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra pour le test")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Appliquer la correction de distorsion
        undist = cv2.undistort(frame, mtx, dist, None, mtx)

        # Afficher les images originales et corrigées côte à côte
        h, w = frame.shape[:2]
        comparaison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        comparaison[:, :w] = frame
        comparaison[:, w:] = undist

        # Ajouter des labels
        cv2.putText(comparaison, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparaison, "Corrigée", (w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Test de calibration', comparaison)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Test terminé.")


if __name__ == "__main__":
    calibrer_camera()
