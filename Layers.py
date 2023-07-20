from keras import layers


# Definition of the ResidualConvBlock class
class ResidualConvBlock(layers.Layer):
    def __init__(self, filters):
        super().__init__()

        # 1x1 Convolution layer for skip connection
        self.x_scip = layers.Conv2D(filters, 1, (2, 2), padding="same")

        # 3x3 Convolution layer
        self.conv1 = layers.Conv2D(filters, 3, (2, 2), padding="same")
        self.norm1 = layers.BatchNormalization()
        self.act1 = layers.ReLU()

        # 3x3 Convolution layer
        self.conv2 = layers.Conv2D(filters, 3, (1, 1), padding="same")
        self.norm2 = layers.BatchNormalization()

        # Addition operation
        self.add = layers.Add()
        self.act2 = layers.ReLU()

    def call(self, inputs):
        x_scip = self.x_scip(inputs)

        # Forward pass through the block
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.add([x_scip, x])  # Skip connection
        return self.act2(x)


# Definition of the ResidualIdentitiBlock class
class ResidualIdentitiBlock(layers.Layer):
    def __init__(self, filters):
        super().__init__()

        # 3x3 Convolution layer
        self.conv1 = layers.Conv2D(filters, 3, padding="same")
        self.norm1 = layers.BatchNormalization()
        self.act1 = layers.ReLU()

        # 3x3 Convolution layer
        self.conv2 = layers.Conv2D(filters, 3, padding="same")
        self.norm2 = layers.BatchNormalization()

        # Addition operation
        self.add = layers.Add()
        self.act2 = layers.ReLU()

    def call(self, inputs):
        # Forward pass through the block
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.add([inputs, x])  # Skip connection
        return self.act2(x)
