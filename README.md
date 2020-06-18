# ModelImplement
We train the two-stage models separately and save the best models.

#Experiment enviroment:
python——3.6 

Keras——2.2.0

Tensorflow——1.8.0

keras-self-attention——0.31.0

scikit-learn——0.20.1


In this project we provided the code for the above two stages of stage1.py and stage2.py.

User can use useTwoStageModel.py to automatic extract DDIs from xml format corpora.

The test instances are finally classified using the saved two model through the architecture of Fig 2.

Since the model contains multiple layer, it generally need some time to train. If the users have no time or GPU to train model, the saved model in the model_saved can be loaded to test.


