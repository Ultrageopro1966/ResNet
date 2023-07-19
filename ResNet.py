# Импортируем необходимые модули из Keras
from keras import Model, layers

# Импортируем пользовательские классы слоев ResidualConvBlock и ResidualIdentitiBlock
from Layers import ResidualConvBlock, ResidualIdentitiBlock

"""
Model: "res_net"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 reshape (Reshape)           multiple                  0

 residual_conv_block (Residu  multiple                 9888
 alConvBlock)

 residual_identiti_block (Re  multiple                 18752
 sidualIdentitiBlock)

 residual_conv_block_1 (Resi  multiple                 58048
 dualConvBlock)

 residual_identiti_block_1 (  multiple                 74368
 ResidualIdentitiBlock)

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

# Определяем класс ResNet, который наследуется от класса Model из Keras
class ResNet(Model):
    def __init__(self):
        # Вызываем конструктор родительского класса
        super().__init__()

        # Инициализируем слой для изменения размерности входных данных до (28, 28, 1)
        self.resh = layers.Reshape((28, 28, 1))

        # Инициализируем слои ResidualConvBlock и ResidualIdentitiBlock с 32 фильтрами
        self.res1 = ResidualConvBlock(32)
        self.res2 = ResidualIdentitiBlock(32)

        # Инициализируем слои ResidualConvBlock и ResidualIdentitiBlock с 64 фильтрами
        self.res3 = ResidualConvBlock(64)
        self.res4 = ResidualIdentitiBlock(64)

        # Инициализируем слои AveragePooling2D и Flatten для усреднения и выравнивания данных
        self.avg_pool = layers.AveragePooling2D()
        self.fl = layers.Flatten()

        # Инициализируем два полносвязных слоя с функцией активации "relu" и "softmax" соответственно
        self.d1 = layers.Dense(512, activation="relu")
        self.d2 = layers.Dense(10, activation="softmax")

    # Прямой проход по модели
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
