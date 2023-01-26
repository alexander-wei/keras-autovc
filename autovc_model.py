
from keras import layers, Model, Sequential

from keras.losses import mae, mse

import tensorflow as tf

import ipdb

SAMPLE_LEN = 128

class LinearNorm(layers.Layer):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()

        assert in_dim > 0

        self.linear_layer = layers.Dense(
            units=out_dim, activation=None, use_bias=bias,
            kernel_initializer = "glorot_uniform"
        )

    def call(self, x):
        return self.linear_layer(x)


class ConvNorm(layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=1,
                 stride=1, padding="same", bias=True,
                 kernel_initializer= "glorot_uniform"):

        super().__init__()

        assert in_channels > 0
        self.conv = layers.Conv1D(
            filters = out_channels,
            kernel_size = kernel_size,
            use_bias = bias,
            strides = stride,
            padding = padding,
            kernel_initializer = kernel_initializer
        )

    def call(self, x):
        return self.conv(x)


class Encoder(layers.Layer):
    def __init__(self, dim_neck, dim_emb, t_freq):
        super().__init__()
        self.dim_neck = dim_neck
        self.dim_emb = dim_emb
        self.freq = t_freq

        convolutions = []

        for i in range(3):
            conv_layer = Sequential(
                [ ConvNorm(80+dim_emb if i==0 else 512,
                           512,
                           kernel_size=5, stride=1,
                           padding='same'),
                  layers.BatchNormalization() ]
            )
            convolutions.append(conv_layer)

        self.convolutions = convolutions

        self.lstm = layers.Bidirectional(
            layers.CuDNNLSTM(units=dim_neck, return_sequences=True),
            merge_mode='concat')

    def call(self, x, c_org):
        x = layers.GaussianNoise(.05)(x)

        x = layers.Concatenate(axis=-1)([x,c_org])

        for conv in self.convolutions:
            x = layers.ReLU()(
                conv(x))

        outputs = self.lstm(x)

        outputs = layers.Reshape((SAMPLE_LEN,self.dim_neck*2,1))(outputs)

        out_forward = outputs[:,:,:self.dim_neck,:]
        out_backward = outputs[:,:,self.dim_neck:,:]

        codes = []

        for i in range(0, SAMPLE_LEN, self.freq):
            codes.append(
                layers.Concatenate(axis=-1)([
                    out_forward[:,i+self.freq-1,:],
                    out_backward[:,i,:]]))

        codes = layers.Concatenate(axis=-1)(codes)
        codes = layers.Permute((2,1))(codes)
        return codes


class Decoder(layers.Layer):
    def __init__(self, dim_emb, dim_pre):
        super().__init__()

        self.lstm1 = layers.CuDNNLSTM(
            units=dim_pre, return_sequences=True)

        convolutions = []

        for i in range(3):
            conv_layer = Sequential(
                [ ConvNorm(80+dim_emb if i==0 else 512,
                           512,
                           kernel_size=5, stride=1,
                           padding='same',
                           kernel_initializer='orthogonal'),
                  layers.Dropout(.0002),
                  layers.BatchNormalization() ]
            )
            convolutions.append(conv_layer)

        self.convolutions = convolutions

        self.lstm2 = Sequential(
            [ layers.CuDNNLSTM(units=1024, return_sequences=True),
              layers.CuDNNLSTM(units=1024,return_sequences=True) ]
        )

        self.linear_projection = LinearNorm(1024,80)

    def call(self, x):
        x = self.lstm1(x)

        for conv in self.convolutions:
            x = layers.ReLU()(
                conv(x))

        outputs = self.lstm2(x)

        decoder_output = self.linear_projection(outputs)

        return decoder_output


class Postnet(layers.Layer):
    def __init__(self):
        super().__init__()
        self.convolutions = []
        self.convolutions.append(
            Sequential(
                [ ConvNorm(80, 512,
                           kernel_size=5, stride=1,
                           padding='same'),
                  layers.Activation('tanh'),
                  layers.BatchNormalization() ])
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
            Sequential(
                [ ConvNorm(512, 512,
                           kernel_size=5, stride=1,
                           padding='same'),
                  layers.Activation('tanh'),
                  layers.BatchNormalization() ])
            )

        self.convolutions.append(
            Sequential(
                [ ConvNorm(512, 80,
                           kernel_size=5, stride=1,
                           padding='same'),
                  layers.BatchNormalization() ])
        )

    def call(self, x):
        for i in range(len(self.convolutions) - 1):
            x = self.convolutions[i](x)

        x = self.convolutions[-1](x)

        return x


class AutoVC(Model):
    def __init__(self, inputs, name="Name", **av):
        super().__init__(inputs, name=name, **av)
        self.Enc = Encoder(32,64,32)
        self.Dec = Decoder(64,512)
        self.postnet = Postnet()

        self.image0_tracker = tf.keras.metrics.Mean(name="image0")
        self.postnet_tracker = tf.keras.metrics.Mean(name="postnet")
        self.code_tracker = tf.keras.metrics.Mean(name="code")

        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        assert True

    def call(self, inputs):
        x = inputs['xim']
        c_org_ = inputs['embeds']
        c_targ_ = inputs['embeds_']

        x = layers.Reshape((128,80))(x)
        c_org = layers.Reshape((32,))(c_org_)
        c_targ = layers.Reshape((32,))(c_targ_)

        c_org = layers.RepeatVector(SAMPLE_LEN)(c_org)
        c_targ = layers.RepeatVector(SAMPLE_LEN)(c_targ)
        # the upsampling scale K /\ the bottleneck downsampling freq

        codes_src_ = self.Enc(x,c_org)
        codes_targ_ = self.Enc(x,c_targ)

        codes_src_ = layers.UpSampling1D(16)(codes_src_)
        codes_targ_ = layers.UpSampling1D(16)(codes_targ_)

        codes_src = layers.Concatenate(axis=-1)([codes_src_, c_org])
        codes_targ = layers.Concatenate(axis=-1)([codes_src_, c_targ])

        y_src = self.Dec(codes_src)
        y_targ = self.Dec(codes_targ)

        y_postnet_src = self.postnet(y_src) + y_src
        y_postnet_targ = self.postnet(y_targ) + y_targ

        codes_recon1, codes_recon2 = \
            layers.UpSampling1D(16)(self.Enc(y_postnet_src, c_org)), \
            layers.UpSampling1D(16)(self.Enc(y_postnet_targ, c_targ))

        returnvals = y_postnet_src, y_targ,\
            codes_src_, codes_targ_,\
            codes_recon1, codes_recon2,\
            y_src, y_postnet_targ

        return returnvals

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            ipdb.set_trace()
            postnet_loss = 1. * mse(y, y_pred[0])
            code_loss = 3. * mae(y_pred[2], y_pred[4])

            image_loss = 1. * mse(y, y_pred[6])
            loss = image_loss + code_loss + postnet_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.image0_tracker.update_state(image_loss)
        self.postnet_tracker.update_state(postnet_loss)
        self.code_tracker.update_state(code_loss)
        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred)

        return {
            'image0':  self.image0_tracker.result(),
            'postnet':  self.postnet_tracker.result(),
            'code':  self.code_tracker.result(),
            'loss':  self.loss_tracker.result()
        }


# XGENERATOR implements keras.utils.Sequence
XGENERATOR = \
    { 'xim': None,
      'embeds': None,
      'embeds_': None }, \
      None

model = AutoVC(inputs = XGENERATOR)
model.compile(optimizer=tf.keras.optimizers.Adam(.001))

model.fit(XGENERATOR,   epochs=1, shuffle=True)
