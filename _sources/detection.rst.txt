Detection
=========
  
Nous avons mis à disposition le code pour la détection sur github. Il sont sur
https://github.com/ia-flash/lab 

Dans le répertoire de détection, on peut lancer la commande : 

.. code:: bash

  cd iaflash/detection
  cfg=/workspace/mmdetection/configs/retinanet_r50_fpn_1x.py
  model=/model/retina/retinanet_r50_fpn_1x_20181125-7b0c2548.pth
  gpu=4
  # lancement de la detection
  ./dist_test.sh $cfg $model $gpu


Cela lancera la détection à partir des images de files_trunc. Les résultats
seront consignés dans la table box_detection. On peut changer ces tables dans
le script test.py. Le shell dist_test.sh parallélise proprement test.py sur les
GPU, sans avoir à gérer les process fils dans un script python.

La connection à la base de données doit être spécifiée dans le fichier de configuration
/docker/env.list. Cette connection est transparente vis-à-vis du connecteur :
postgres ou vertica. La dépendance à DSS est évitée, puisqu’on se connecte
directement au gestionnaire de DB. Il faut seulement (avec DSS par exemple)
créer le shéma de la table de sortie box_detection.


Custom Generator
----------------

.. automodule:: detection.generator_detection
    :members:

