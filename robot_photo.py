import cv2
import numpy as np
import time
import os
import sys

sys.path.append('//Users//samos//Documents//StageCv//galatae-api')
from robot import Robot

# Variables globales
vitesse = 20
nbPhoto = 15
posReplis = (300, 0, 300, 90)


def load_calibration_data():
    """Charge les paramètres de calibration de la caméra."""
    if os.path.exists('calibration_camera_HBV-W202012HD.npz'):
        calibration_params = np.load('calibration_camera_HBV-W202012HD.npz', allow_pickle=True)
        mtx = np.array(calibration_params['mtx'])
        dist = np.array(calibration_params['dist'])
        print("Paramètres de calibration chargés avec succès")
        return mtx, dist
    else:
        print("Aucun paramètre de calibration trouvé")
        return None, None


def capture_images(output_folder="images", image_prefix="VisM3_6_", image_format="jpg"):
    """Capture des images en utilisant la caméra et un robot."""

    # Création du dossier de stockage des images
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Dossier '{output_folder}' créé.")

    # Initialisation du robot
    r = Robot('/dev/ttyACM0')
    time.sleep(3)
    r.set_joint_speed(vitesse)
    r.reset_pos()
    r.calibrate_gripper()

    # Chargement des paramètres de calibration
    mtx, dist = load_calibration_data()
    calibration_available = mtx is not None and dist is not None

    if calibration_available:
        print("Calibration chargée avec succès")
    else:
        print("Erreur: Calibration non chargée")
        exit()

    # Initialisation de la caméra
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra.")
        return

    # Compter le nombre d'images déjà présentes dans le dossier
    existing_images = len([f for f in os.listdir(output_folder) if f.lower().endswith(f'.{image_format.lower()}')])
    img_counter = existing_images

    # Précalculer la matrice de calibration si disponible
    if calibration_available:
        h, w = 720, 1280  # Taille de la caméra
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Position d'attente du robot avant la capture
    r.go_to_point(*posReplis)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur: Impossible de lire une frame de la caméra.")
            break

        # Correction de la distorsion
        frame_undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        frame_undistorted = frame_undistorted[y:y + h, x:x + w]
        frame_undistorted = cv2.resize(frame_undistorted, (frame.shape[1], frame.shape[0]))

        cv2.imshow("Capture d'images", frame_undistorted)

        # Boucle de capture
        for _ in range(nbPhoto):
            positions = [
                (300, 0, 300, 180),
                (300, 0, 300, 180, 45),
                (300, 0, 300, 180, -45),
                (300, 15, 300, 180, 45),
                (300, 15, 300, 180, -45),
                (300, 0, 150, 180),
                (300, 0, 150, 180, 45),
                (300, 0, 150, 180, -45),
                (300, 15, 150, 180, 45),
                (300, 15, 150, 180, -45),
                (300, 0, 50, 180),
                (300, 0, 50, 180, 45),
                (300, 0, 50, 180, -45),
                (300, 15, 50, 180, 45),
                (300, 15, 50, 180, -45)
            ]

            for pos in positions:
                r.go_to_point(*pos)
                img_name = f"{image_prefix}{img_counter + 1}.{image_format}"
                img_path = os.path.join(output_folder, img_name)
                cv2.imwrite(img_path, frame_undistorted)
                img_counter += 1
                print(f"Image sauvegardée: {img_path}")

        r.go_to_point(*posReplis)
        r.reset_pos()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libération des ressources
    cap.release()
    cv2.destroyAllWindows()
    print(f"{img_counter - existing_images} nouvelles images ont été sauvegardées dans le dossier '{output_folder}'.")
    print(f"Nombre total d'images dans le dossier: {img_counter}")


if __name__ == "__main__":
    capture_images()
