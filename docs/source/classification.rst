Classification
==============


Il faut absolument lancer la détection de véhicules sur toutes les images avant
la classification. En effet, le générateur pytorch d’IAflash qui prépare les
batchs pour le modèle a besoin d’avoir les coordonnées des véhicules. C’est
dans le script classification/custom_generator.py.

Création des jeux de données d’entraînement, validation et test. 

La fonction filter permet de filtrer les clichés suivant le type de radar, le
sens, les marques/modèles et bien d’autres caractéristiques à partir d’une
table ressemblant à  CarteGrise_norm_melt_joined_psql. Supposons que la liste
des modèles classes-2018-151.csv a été constituée. 

Après être rentré dans le docker IAflash avec :

.. code:: bash

  cd ~/src/lab
  make exec
  
  dir=/model/resnet18-151
  time python -m iaflash.filter --dir \
  --table cartegrise_norm_melt_joined_psql --status 4 6 13 \
  --class_list /model/classes-2018-151.csv \
  --sampling 0 \
  --limit 0 \
  --score 0.95 \
  --where "(\"TYPEEQUIP_Libelle\"='ETC' AND img_name LIKE '%_1.jpg') OR (\"TYPEEQUIP_Libelle\"!='ETC')" \
  --api-key dummyKey_ \
  --project-key 'VIT3' \
  --vertica-host 192.168.4.25 \
  --connector postgres \
  --shuffle \
  --not-null x1 path img_name

Pour avoir le détail de chaque argument, on invite le lecteur à lancer : 

.. code:: bash

    python -m iaflash.filter -h

La commande génère train.csv, test.csv, test.csv et idx_to_class.json dans dir 
train.csv ressemble à :


::

    head /model/resnet18-151-refined/train.csv
    img_path,target,x1,y1,x2,y2,score
    DISK3/img/c/bfc/166118668/08529_20180904_111931_00048_2.jpg,115.0,504,84,1236,496,0.84
    DISK3/img/c/bfc/166118668/08529_20180904_111931_00048_1.jpg,115.0,551,86,1310,520,0.82
    DISK3/img/2/e7a/161465758/08666_20180712_082640_00001_2.jpg,95.0,548,140,1213,581,0.85
    DISK3/img/2/e7a/161465758/08666_20180712_082640_00001_1.jpg,95.0,437,141,1140,613,0.68
    DISK2/img/6/da2/158092944/12507_20180525_123705_00018_2.jpg,16.0,493,333,1518,1022,0.76
    DISK1/img/5/2c9/150276287/40297_20180107_110835_00142_1.jpg,62.0,111,386,494,716,0.69
    DISK2/img/0/d43/155805370/00374_20180419_094123_00006_2.jpg,59.0,435,54,1058,388,0.92


On peut se rendre compte du contenu de train.csv par exemple grâce à l’outil de visualisation.

Le mappage entre les nom des classes et le numéro du neurone est dans idx_to_class.json.

Lancement de l’entraînement
---------------------------

Cela fait toujours quelque chose quand on entraîne un réseau de neurone sur
pour la première fois sur un jeu de données. Est-ce que notre travail jusqu’ici
n’aurait pas été vain?

Je vous rassure, nous allons obtenir d’excellents résultats, à condition
d’avoir des classes d’apprentissage propres. Il va peut-être falloir faire des
allers-retours entre le choix des classes et la matrice de confusion produite
après ce premier entraînement. C’est même comme cela que nous avons gagné au
moins 5% et franchir la barre des 90% d’accuracy.

.. code:: bash

  dir=/model/resnet18-151
  
  nohup python -m iaflash.classification.main_classifier \
  -a resnet18 \
  --lr 0.01 --batch-size 256  \
  --pretrained \
  --dist-url 'tcp://127.0.0.1:1234' \
  --dist-backend 'nccl' \
  --multiprocessing-distributed \
  --world-size 1 \
  --rank 0 \
  --workers 16 \
  dir > train-151.out &




Main classifier
---------------

main_classifier.py est inspiré de l’exemple officiel de pytorch avec Imagenet.
Cela permet d’entraîner ou de faire une inférence sur plusieurs GPU chacun
alimenté avec plusieurs CPU.

Il est même possible de mettre en réseau plusieurs serveurs GPU. Nous avons
adapté le code afin de recevoir des batchs de données préparés à partir de nos
fichiers train.csv et val.csv (custom_generator.py). En fait, comme nous ne
pouvons pas nous permettre d’enregistrer les imagettes de voitures dans un
disque pour faute de place, le générateur rogne l’image suivant les coordonnées
et l’ajoute au batch.

.. automodule:: classification.main_classifier
    :members:


La commande va produire dans dir :
    - model_best.pth.tar (le meilleur modèle), 
    - checkpoint.pth.tar,
    - predictions.csv, probabilities.csv, targets.csv des csv indexés par le numéro de ligne de l’image.
    - results.csv qui est la jointure de val.csv avec les csv précédents. Ce csv est pratique car il permet une visualisation des résultats.


Visualisation de la matrice de résultat
---------------------------------------

Il suffit dans dir  d’ouvrir le fichier confusion.png. Il faudra comprendre pourquoi certaines classes sont confondues avec d’autres.


