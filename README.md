# :books: - Génération et reconstruction de visages par des réseaux adverses génératifs


## :computer: - Équipe 

- BAILLY Valentin
- GUILLOU-KEREDAN Antoine


## :memo: - Utilisation

Ce repo est divisé en deux dossiers :
- "first approach", contenant les premiers modèle de GAN et cGAN, ainsi qu'un module "utils" contenant les fonctions auxiliaires permettant de creer le dataset d'entrainement
- "second approach", contenant les modèles de GAN et segmenation semantique, de même qu'un module "utils" pour chaque modèle. Pour run la partie GAN, il s'agit du fichier "inpainting.py". Pour la segmentation sémantique, "train_deeplabv3.py"

Chaque modèle contient son fichier .env permettant de définir les paths associés aux datasets d'entrainement, de test et validation.

