from keras import layers


# Определение класса ResidualConvBlock
class ResidualConvBlock(layers.Layer):
    def __init__(self, filters):
        super().__init__()

        # Слой свертки 1x1 для пропуска соединения (skip connection)
        self.x_scip = layers.Conv2D(filters, 1, (2, 2), padding="same")

        # Слой свертки 3x3
        self.conv1 = layers.Conv2D(filters, 3, (2, 2), padding="same")
        self.norm1 = layers.BatchNormalization()
        self.act1 = layers.ReLU()

        # Слой свертки 3x3
        self.conv2 = layers.Conv2D(filters, 3, (1, 1), padding="same")
        self.norm2 = layers.BatchNormalization()

        # Операция сложения
        self.add = layers.Add()
        self.act2 = layers.ReLU()

    def call(self, inputs):
        x_scip = self.x_scip(inputs)

        # Прямой проход через блок
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.add([x_scip, x])  # Пропуск соединения (skip connection)
        return self.act2(x)


# Определение класса ResidualIdentitiBlock
class ResidualIdentitiBlock(layers.Layer):
    def __init__(self, filters):
        super().__init__()

        # Слой свертки 3x3
        self.conv1 = layers.Conv2D(filters, 3, padding="same")
        self.norm1 = layers.BatchNormalization()
        self.act1 = layers.ReLU()

        # Слой свертки 3x3
        self.conv2 = layers.Conv2D(filters, 3, padding="same")
        self.norm2 = layers.BatchNormalization()

        # Операция сложения
        self.add = layers.Add()
        self.act2 = layers.ReLU()

    def call(self, inputs):
        # Прямой проход через блок
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.add([inputs, x])  # Пропуск соединения (skip connection)
        return self.act2(x)
