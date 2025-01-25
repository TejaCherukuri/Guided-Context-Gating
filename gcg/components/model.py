from keras.layers import Conv1D, Conv2D, Conv3D
from keras.layers import Conv2D, LayerNormalization, Layer
from tensorflow.keras.layers import Reshape, Activation, Softmax, Permute, Add, Dot
from keras.optimizers import RMSprop
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.callbacks import ModelCheckpoint
from keras import ops
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score,roc_auc_score,classification_report
from gcg import config
from gcg.utils import CustomException, logging
import sys

class GlobalContextAttention(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=8, transform_activation='linear', **kwargs):
        """
        Initializes the GlobalContextAttention layer.

        Args:
            reduction_ratio (int): Reduces the input filters by this factor for the
                                  bottleneck block of the transform submodule.
            transform_activation (str): Activation function to apply to the output
                                        of the transform block.
            **kwargs: Additional keyword arguments for the Layer class.
        """
        super(GlobalContextAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.transform_activation = transform_activation

    def build(self, input_shape):
        """
        Builds the layer by initializing weights and sub-layers.

        Args:
            input_shape: Shape of the input tensor.
        """
        self.channel_dim = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
        self.rank = len(input_shape)

        # Validate input rank
        if self.rank not in [3, 4, 5]:
            raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial), or 5 (spatio-temporal)')

        # Calculate the number of channels
        self.channels = input_shape[self.channel_dim]

        # Initialize sub-layers
        self.conv_context = self._convND_layer(1)  # Context modelling block
        self.conv_transform_bottleneck = self._convND_layer(self.channels // self.reduction_ratio)  # Transform bottleneck
        self.conv_transform_output = self._convND_layer(self.channels)  # Transform output block

        # Softmax and Dot layers
        self.softmax = Softmax(axis=self._get_flat_spatial_dim())
        self.dot = Dot(axes=(1, 1))  # Dot product over the flattened spatial dimensions

        # Activation layers
        self.activation_relu = Activation('relu')
        self.activation_transform = Activation(self.transform_activation)

        # Add layer for final output
        self.add = Add()

        super(GlobalContextAttention, self).build(input_shape)

    def call(self, inputs):
        """
        Performs the forward pass of the layer.

        Args:
            inputs: Input tensor.

        Returns:
            Output tensor with global context attention applied.
        """
        # Context Modelling Block
        input_flat = self._spatial_flattenND(inputs)  # [B, spatial_dims, C]
        context = self.conv_context(inputs)  # [B, spatial_dims, 1]
        context = self._spatial_flattenND(context)  # [B, spatial_dims, 1]
        context = self.softmax(context)  # Apply softmax over spatial_dims
        context = self.dot([input_flat, context])  # [B, C, 1]
        context = self._spatial_expandND(context)  # [B, C, 1, 1, ...]

        # Transform Block
        transform = self.conv_transform_bottleneck(context)  # [B, C // R, 1, 1, ...]
        transform = self.activation_relu(transform)
        transform = self.conv_transform_output(transform)  # [B, C, 1, 1, ...]
        transform = self.activation_transform(transform)

        # Apply context transform
        out = self.add([inputs, transform])  # [B, spatial_dims, C]

        return out

    def _convND_layer(self, filters):
        """
        Creates a Conv1D, Conv2D, or Conv3D layer based on the input rank.

        Args:
            filters (int): Number of filters for the convolutional layer.

        Returns:
            A Conv1D, Conv2D, or Conv3D layer.
        """
        if self.rank == 3:
            return Conv1D(filters, kernel_size=1, padding='same', use_bias=False, kernel_initializer='he_normal')
        elif self.rank == 4:
            return Conv2D(filters, kernel_size=1, padding='same', use_bias=False, kernel_initializer='he_normal')
        elif self.rank == 5:
            return Conv3D(filters, kernel_size=1, padding='same', use_bias=False, kernel_initializer='he_normal')

    def _spatial_flattenND(self, ip):
        """
        Flattens the spatial dimensions of the input tensor.

        Args:
            ip: Input tensor.

        Returns:
            Flattened tensor.
        """
        if self.rank == 3:
            return ip  # Identity op for rank 3
        else:
            shape = (ip.shape[self.channel_dim], -1) if self.channel_dim == 1 else (-1, ip.shape[-1])
            return Reshape(shape)(ip)

    def _spatial_expandND(self, ip):
        """
        Expands the spatial dimensions of the input tensor.

        Args:
            ip: Input tensor.

        Returns:
            Expanded tensor.
        """
        if self.rank == 3:
            return Permute((2, 1))(ip)  # Identity op for rank 3
        else:
            shape = (-1, *(1 for _ in range(self.rank - 2))) if self.channel_dim == 1 else (*(1 for _ in range(self.rank - 2)), -1)
            return Reshape(shape)(ip)

    def _get_flat_spatial_dim(self):
        """
        Returns the axis for flattening spatial dimensions.

        Returns:
            Axis for flattening.
        """
        return 1 if self.channel_dim == 1 else -1

    def get_config(self):
        """
        Returns the configuration of the layer for serialization.

        Returns:
            A dictionary containing the layer configuration.
        """
        config = super(GlobalContextAttention, self).get_config()
        config.update({
            'reduction_ratio': self.reduction_ratio,
            'transform_activation': self.transform_activation,
        })
        return config
    

class AttentionGate(Layer):
    def __init__(self, filters, **kwargs):
        self.filters = filters
        super(AttentionGate, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create trainable parameters for attention gate
        self.conv_xl = Conv2D(self.filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.conv_g = Conv2D(self.filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')
        self.psi = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='linear')
        self.layer_norm = LayerNormalization(axis=-1)

        # Build the child layers
        self.conv_xl.build(input_shape[0])  # Build conv_xl with the shape of xl
        self.conv_g.build(input_shape[1])   # Build conv_g with the shape of g
        self.psi.build(input_shape[0])      # Build psi with the shape of xl
        self.layer_norm.build(input_shape[0])  # Build layer_norm with the shape of xl

        # Add trainable weights
        self.bxg = self.add_weight(name='bxg',
                                  shape=(self.filters,),
                                  initializer='zeros',
                                  trainable=True)
        self.bpsi = self.add_weight(name='bpsi',
                                   shape=(1,),
                                   initializer='zeros',
                                   trainable=True)

        super(AttentionGate, self).build(input_shape)

    def call(self, inputs):
        xl, g = inputs

        # Apply convolutional operations
        xl_conv = self.conv_xl(xl)
        g_conv = self.conv_g(g)

        # Compute additive attention
        att = tf.keras.backend.relu(xl_conv + g_conv + self.bxg)
        att = self.layer_norm(att)  # Add LayerNormalization
        att = self.psi(att) + self.bpsi
        att = tf.keras.backend.sigmoid(att)

        # Apply attention gate
        x_hat = att * xl

        return x_hat
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(AttentionGate, self).get_config()
        config.update({'filters': self.filters})
        return config
    

class GCRMSprop(RMSprop):
    def get_gradients(self, loss, params):
        # We here just provide a modified get_gradients() function since we are trying to just compute the centralized gradients.

        grads = []
        gradients = super().get_gradients()
        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= ops.mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads

# Build the model
def build_model(input_shape, num_classes):
    try:
        logging.info("Loading weights of EfficientNetV2B0...")
        base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=input_shape)

        fmaps = base_model.output

        logging.info("Initializing Global Context Attention...")
        context_fmaps = GlobalContextAttention()(fmaps)

        logging.info("Initializing AttentionGate...")
        att_fmaps = AttentionGate(fmaps.shape[-1])([fmaps, context_fmaps])

        x = layers.GlobalAveragePooling2D()(att_fmaps)

        # First Dense Layer
        x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.005))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        # Second Dense Layer
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.005))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)

        output = layers.Dense(num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer=GCRMSprop(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

        logging.info("Model Built Successfully!")

        return model
    
    except Exception as e:
        raise CustomException(e, sys)

# Train the model
def train_model(model, X_train, X_test, y_train, y_test):

    try:
        # Define the necessary callbacks
        checkpoint = ModelCheckpoint(config.MODEL_SAVE_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        callbacks = [checkpoint]

        logging.info(f"Training network for {config.EPOCHS} epochs...")
        hist = model.fit(X_train, y_train, batch_size=config.BATCH_SIZE,
                        validation_data=(X_test, y_test),
                        epochs=config.EPOCHS, callbacks=callbacks)
        
        return hist
    
    except Exception as e:
        raise CustomException(e, sys)
    
# Evaluate the model
def evaluate_model(model, X_test, y_test):
    try:
        
        y_score = model.predict(X_test)
        y_pred = np.argmax(y_score, axis=-1)
        Y_test = np.argmax(y_test, axis=-1)

        acc = accuracy_score(Y_test,y_pred)
        mpre = precision_score(Y_test,y_pred,average='macro')
        mrecall = recall_score(Y_test,y_pred,average='macro')
        mf1 = f1_score(Y_test,y_pred,average='macro')
        kappa = cohen_kappa_score(Y_test,y_pred,weights='quadratic')
        auc = roc_auc_score(Y_test, y_score, average='macro', multi_class='ovr')

        logging.info(f"Accuracy: {round(acc*100,2)}")
        logging.info(f"Macro Precision: {round(mpre*100,2)}")
        logging.info(f"Macro Recall: {round(mrecall*100,2)}")
        logging.info(f"Macro F1-Score: {round(mf1*100,2)}")
        logging.info(f"Quadratic Kappa Score: {round(kappa*100,2)}")
        logging.info(f"ROC AUC Score: {round(auc*100,2)}")
        logging.info(classification_report(Y_test, y_pred, digits=4))

    except Exception as e:
        raise CustomException(e, sys)
    
