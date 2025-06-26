import cv2
import numpy as np
import time
import sys
import os
import logging

sys.path.append('../galatae-api/')
from robot import Robot
from ultralytics import YOLO

# Configuration du système de journalisation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_Calibration():
    """
    Fonction qui charge la calibration et qui retourne les données de celle-ci
    Retourne:
    - mtx: matrice intrinsèque de la caméra
    - dist: coefficients de distorsion
    - None, None si les paramètres ne sont pas trouvés
    """
    try:
        if os.path.exists("calibration_camera_HBV-W202012HD.2.npz"):
            calib = np.load('calibration_camera_HBV-W202012HD.2.npz')
            logger.info("Calibration Matrice load with success")
            return calib['mtx'], calib['dist'], calib['camera_to_workspace']
        else:
            logger.error("Fichier de calibration non trouvé")
            return None, None
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la calibration: {e}")
        return None, None

def pose_to_matrix(x, y, z, a_deg, b_deg):
    """
    Convertit une pose en matrice de transformation homogène
    """
    try:
        a = np.radians(a_deg)
        b = np.radians(b_deg)

        Rx = np.array([
            [1, 0,          0],
            [0, np.cos(a), -np.sin(a)],
            [0, np.sin(a),  np.cos(a)]
        ])

        Ry = np.array([
            [np.cos(b),  0, np.sin(b)],
            [0,          1, 0],
            [-np.sin(b), 0, np.cos(b)]
        ])

        R = Ry @ Rx

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]

        return T
    except Exception as e:
        logger.error(f"Erreur lors de la conversion de la pose en matrice: {e}")
        return None

def CapAndManip():
    try:
        v = 30
        logger.info("Initialisation du modèle YOLO")
        model = YOLO('best.pt')

        # Configuration pour caméra sur le flange
        # Position relative de la caméra par rapport au centre du flange
        # Ces valeurs doivent être mesurées précisément sur votre système
        tCamToFlange = np.eye(4)
        tCamToFlange[:3, 3] = [57.26, 0, 41.78]  # [x, y, z] en mm

        # Initialisation robot
        logger.info("Initialisation du robot")
        r = Robot('/dev/ttyACM0')
        time.sleep(3)
        r.set_joint_speed(v)
        r.reset_pos()
        r.calibrate_gripper()

        # Position de départ sécurisée
        logger.info("Déplacement vers la position de départ sécurisée")
        r.go_to_point([300, 0, 150, 180, 0])  # Position de départ sécurisée à 150mm
        time.sleep(2)  # Attendre que le robot soit bien positionné

        # Calibration caméra
        logger.info("Chargement de la calibration de la caméra")
        calib_data = np.load('calibration_camera_HBV-W202012HD.npz')
        mtx, dist = calib_data['mtx'], calib_data['dist']
        
        # Charger la distance caméra-plan de travail depuis le fichier de calibration
        # Si la clé n'existe pas (anciennes calibrations), utiliser la valeur par défaut
        if 'camera_to_workspace' in calib_data:
            Z_CAMERA_TO_WORKSPACE = float(calib_data['camera_to_workspace'])
            logger.info(f"Distance caméra-plan chargée depuis la calibration: {Z_CAMERA_TO_WORKSPACE:.2f} mm")
        else:
            Z_CAMERA_TO_WORKSPACE = 150.0  # Valeur par défaut si non disponible
            logger.warning(f"Distance caméra-plan non trouvée dans le fichier de calibration, utilisation de la valeur par défaut: {Z_CAMERA_TO_WORKSPACE:.2f} mm")
        
        if mtx is None or dist is None:
            logger.error("Impossible de continuer sans calibration de la caméra")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Impossible d'ouvrir la caméra")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        logger.info("Calibration de la caméra terminée avec succès")

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Erreur lors de la lecture de la caméra")
                break

            undistorted = cv2.undistort(frame, mtx, dist, None, new_mtx)
            x, y, w_roi, h_roi = roi
            undistorted = undistorted[y:y+h_roi, x:x+w_roi]

            # Conversion en niveaux de gris pour le traitement
            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
            
            results = model(undistorted)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    label = f'{class_name}: {confidence:.2f}'

                    logger.info(f"Détection de {class_name} avec une confiance de {confidence:.2f}")

                    # Extraction de la région d'intérêt (ROI)
                    roi = gray[y1:y2, x1:x2]
                    
                    # Détection des contours pour trouver l'orientation
                    _, thresh = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Valeur par défaut de l'angle
                    object_orientation_angle = 0
                    
                    if contours:
                        # Trouver le plus grand contour
                        cnt = max(contours, key=cv2.contourArea)
                        # Calculer la boîte orientée minimale
                        rect = cv2.minAreaRect(cnt)
                        object_orientation_angle = rect[2]
                        
                        # Normaliser l'angle entre -45 et 45 degrés
                        if object_orientation_angle < -45:
                            object_orientation_angle = 90 + object_orientation_angle
                        if object_orientation_angle > 45:
                            object_orientation_angle = 90 - object_orientation_angle

                    # Centre de la boîte
                    u, v = (x1 + x2) / 2, (y1 + y2) / 2

                    # Obtention de la position actuelle du robot
                    current_pose = r.get_pose()
                    current_x, current_y, current_z, current_a, current_b, _ = current_pose
                    
                    logger.info(f"Position actuelle du robot: x={current_x:.1f}, y={current_y:.1f}, z={current_z:.1f}, a={current_a:.1f}, b={current_b:.1f}")

                    # Utilisation de la distance fixe caméra->plan de travail
                    # Pour une caméra sur flange, c'est la distance estimée à l'objet
                    Zc = Z_CAMERA_TO_WORKSPACE
                    
                    logger.info(f"Distance caméra-plan estimée: Zc={Zc:.1f}mm")

                    # Conversion 2D -> 3D dans le repère caméra
                    fx, fy = mtx[0, 0], mtx[1, 1]
                    cx, cy = mtx[0, 2], mtx[1, 2]
                    
                    # Log des paramètres de calibration
                    logger.info(f"Paramètres de calibration: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
                    logger.info(f"Position dans l'image: u={u:.1f}, v={v:.1f}")
                    
                    # Calcul des coordonnées dans le repère caméra
                    Xc = (u - cx) * Zc / fx
                    Yc = (v - cy) * Zc / fy
                    
                    logger.info(f"Coordonnées caméra avant transformation: Xc={Xc:.1f}mm, Yc={Yc:.1f}mm, Zc={Zc:.1f}mm")
                    
                    # Création du point dans le repère caméra
                    pCam = np.array([Xc, Yc, Zc, 1]).reshape(4, 1)
                    
                    # Transformation vers le repère flange (poignet du robot)
                    pFlange = tCamToFlange @ pCam
                    
                    logger.info(f"Coordonnées après transformation flange: {pFlange[:3].flatten()}")
                    
                    #Construction de la matrice de transformation flange -> base
                    tFlangeToBase = pose_to_matrix(current_x, current_y, current_z, current_a, current_b)
                    if tFlangeToBase is None:
                        logger.error("Erreur lors de la transformation de la pose")
                        continue
                    
                    # Log de la matrice pour débogage
                    logger.info(f"Matrice de transformation flange->base:\n{tFlangeToBase}")
                        
                    # Transform de repère flange -> base
                    pRobot = tFlangeToBase @ pFlange
                    xR, yR, zR = pRobot[:3].flatten()
                    
                    # Ajustement de la hauteur en fonction de la position actuelle du robot
                    # Si le Z calculé est négatif, on utilise la hauteur actuelle du robot
                    if zR < 0:
                        logger.warning(f"Z calculé négatif: {zR:.1f}mm, utilisation de la hauteur actuelle du robot")
                        zR = current_z
                    
                    logger.info(f"Coordonnées après transformation base: x={xR:.1f}mm, y={yR:.1f}mm, z={zR:.1f}mm")
                    
                    # Vérification des coordonnées calculées
                    if not np.isfinite([xR, yR, zR]).all():
                        logger.error("Coordonnées invalides calculées")
                        continue
                    
                    logger.info(f"Position calculée finale: x={xR:.1f}mm, y={yR:.1f}mm, z={zR:.1f}mm, angle={object_orientation_angle:.1f}°")

                    # Ajustement de la hauteur pour l'approche
                    zR_adjusted = zR + 20  # Approche à 20mm au-dessus de l'objet
                    
                    # Adapter l'angle du robot en fonction de l'orientation de l'objet et limiter à 90°
                    robot_wrist_angle = 180 + object_orientation_angle
                    if robot_wrist_angle > 90:
                        robot_wrist_angle = 90
                    elif robot_wrist_angle < -90:
                        robot_wrist_angle = -90

                    try:
                        # Aller au-dessus de l'objet (pré-position)
                        logger.info("Déplacement vers la position de préhension")
                        r.go_to_point([xR, yR, zR_adjusted, 180, robot_wrist_angle])
                        time.sleep(1)

                        # Descendre vers l'objet avec l'angle approprié
                        logger.info("Approche de l'objet")
                        r.go_to_point([xR, yR, zR, 180, robot_wrist_angle])
                        time.sleep(1)

                        # Fermer le gripper
                        logger.info("Fermeture du gripper")
                        r.close_gripper()

                        # Remonter après prise
                        logger.info("Remontée après prise")
                        r.go_to_point([xR, yR, zR_adjusted, 180, robot_wrist_angle])
                        time.sleep(1)

                        # Retourner à une position verticale pour le transport
                        logger.info("Retour à la position verticale")
                        r.go_to_point([xR, yR, zR_adjusted, 180, 0])
                        time.sleep(1)

                        # Aller à une position de dépose
                        logger.info("Déplacement vers la position de dépose")
                        r.go_to_point([390, 0, 150, 180, 0])  # Position de dépose à 150mm
                        r.open_gripper()

                    except Exception as e:
                        logger.error(f"Erreur lors de la manipulation: {e}")
                        continue

                    # Dessin dans l'image
                    cv2.rectangle(undistorted, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(undistorted, (int(u), int(v)), 5, (255, 0, 0), -1)
                    cv2.putText(undistorted, label, (x1, y1 - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(undistorted, f"Angle: {object_orientation_angle:.1f}", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("YOLO Detection", undistorted)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Arrêt du programme demandé par l'utilisateur")
                break

    except Exception as e:
        logger.error(f"Erreur critique: {e}")
    finally:
        logger.info("Nettoyage des ressources")
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    CapAndManip()
