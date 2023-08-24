# NNECode

NNE (Neural Network Embedding) is a framework that automatically compresses the connection matrix using a Deep Autoencoder model and gives meaning to the obtained features through network metrics. Therefore, the code necessary for analysis consists of the Deep Autoencoder model and the code to compute the network metrics. These are listed below:

deep_autoencoder.py is a coder to train deep autoencoder models.
Usage: python deep_autoencoder.py [depth of layers] [number of middle layer nodes]
Example: python deep_autoencoder.py 3 5


Network_analysis.py is a coder to calculate our new network metrics as well as representative network metrics.
Usage: python Network_analysis.py [directory for dataset]
Example python Network_analysis.py 180731001


