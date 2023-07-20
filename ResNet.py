from keras import layers
# Import necessary modules from Keras
from keras import Model, layers

# Import custom layer classes ResidualConvBlock and ResidualIdentityBlock
from Layers import ResidualConvBlock, ResidualIdentityBlock

"""
Model: "res_net"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 reshape (Reshape)           multiple                  0

 residual_conv_block (Residu  multiple                 9888
 alConvBlock)

 residual_identity_block (Re  multiple                 18752
 sidualIdentityBlock)

 residual_conv_block_1 (Resi  multiple                 58048
 dualConvBlock)

 residual_identity_block_1 (  multiple                 74368
 ResidualIdentityBlock)

 average_pooling2d (AverageP  multiple                 0
 ooling2D)

 flatten (Flatten)           multiple                  0

 dense (Dense)               multiple                  295424

 dense_1 (Dense)             multiple                  5130

=================================================================
Total params: 461,610
Trainable params: 460,842
Non-trainable params: 768
_________________________________________________________________
"""

# Define the ResNet class, which inherits from the Model class in Keras
class ResNet(Model):
    def __init__(self):
        # Call the constructor of the parent class
        super().__init__()

        # Initialize the layer for reshaping the input data to (28, 28, 1)
        self.resh = layers.Reshape((28, 28, 1))

        # Initialize the ResidualConvBlock and ResidualIdentityBlock layers with 32 filters
        self.res1 = ResidualConvBlock(32)
        self.res2 = ResidualIdentityBlock(32)

        # Initialize the ResidualConvBlock and ResidualIdentityBlock layers with 64 filters
        self.res3 = ResidualConvBlock(64)
        self.res4 = ResidualIdentityBlock(64)

        # Initialize the AveragePooling2D and Flatten layers for pooling and flattening the data
        self.avg_pool = layers.AveragePooling2D()
        self.fl = layers.Flatten()

        # Initialize two fully connected layers with "relu" and "softmax" activation functions respectively
        self.d1 = layers.Dense(512, activation="relu")
        self.d2 = layers.Dense(10, activation="softmax")

    # Forward pass through the model
    def call(self, inputs):
        x = self.resh(inputs)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.avg_pool(x)
        x = self.fl(x)
        x = self.d1(x)
        return self.d2(x)
