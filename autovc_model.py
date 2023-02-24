import tensorflow as tf
from keras import layers
from keras import Model
from keras import Sequential
from keras.initializers import VarianceScaling

# kernel initializer scaling constants
scale_relu = 2
scale_tanh = 25/9
scale_linear = 1

class LinearNorm(layers.Layer):
    def __init__(self, out_dim, bias=True, scale=1.):
        super(LinearNorm, self).__init__()

        init = VarianceScaling(
            scale=scale, mode="fan_avg", distribution="uniform"
        )
        self.linear_layer = layers.Dense(
            units=out_dim, activation=None, use_bias=bias,
            kernel_initializer = init
        )
    def call(self, inputs):
        return self.linear_layer(inputs)


class ConvNorm(layers.Layer):
    def __init__(self, out_channels, kernel_size=1,
                 stride=1, padding="same", bias=True, scale=1.):
        super(ConvNorm, self).__init__()

        init = VarianceScaling(
            scale=scale, mode="fan_avg", distribution="uniform"
        )
        self.conv = layers.Conv1D(
            filters=out_channels, kernel_size=kernel_size, use_bias=bias,
            strides=stride, padding=padding, kernel_initializer = init,
            data_format="channels_last"
        )

    def call(self, inputs):
        return self.conv(inputs)


class Encoder(layers.Layer):
    def __init__(self, dim_neck, dim_emb, t_freq):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.dim_emb = dim_emb
        self.freq = t_freq

        convolutions = []

        for i in range(3):
            conv_layer = Sequential(
                [ ConvNorm(512,
                           kernel_size=5, stride=1,
                           padding="same",
                           scale=scale_relu),
                  layers.BatchNormalization() ]
            )
            convolutions.append(conv_layer)

        self.convolutions = convolutions

        self.lstm1 = layers.Bidirectional(
            layers.CuDNNLSTM(units=dim_neck, return_sequences=True),
            merge_mode='concat')
        self.lstm2 = layers.Bidirectional(
            layers.CuDNNLSTM(units=dim_neck, return_sequences=True),
            merge_mode='concat')


    def call(self, x, c_org):
        x = layers.GaussianNoise(.05)(x)
        x = layers.Concatenate(axis=-1)([x,c_org])

        for conv in self.convolutions:
            x = layers.ReLU()(
                conv(x))
        outputs = self.lstm1(x)
        outputs = self.lstm2(outputs)
        outputs = layers.Reshape((128,self.dim_neck*2,1))(outputs)
        out_forward = outputs[:,:,:self.dim_neck]
        out_backward = outputs[:,:,self.dim_neck:]

        codes = []
        for i in range(0, 128, self.freq):
            codes.append(
                layers.Concatenate(axis=1)([
                    out_forward[:,i+self.freq-1,:],
                    out_backward[:,i,:]]))

        codes = layers.Concatenate(axis=-1)(codes)

        codes = layers.Permute((2,1))(codes)
        return codes


class Decoder(layers.Layer):
    def __init__(self, dim_pre):
        super(Decoder, self).__init__()

        self.lstm1 = layers.CuDNNLSTM(
            units=dim_pre, return_sequences=True)

        convolutions = []
        for i in range(3):
            conv_layer = Sequential(
                [ ConvNorm(512,
                           kernel_size=5, stride=1,
                           padding="same",
                           scale=scale_relu),
                  layers.BatchNormalization() ]
            )
            convolutions.append(conv_layer)

        self.convolutions = convolutions

        self.lstm2 = Sequential(
            [ layers.CuDNNLSTM(units=1024, return_sequences=True),
              layers.CuDNNLSTM(units=1024,return_sequences=True) ]
        )

        self.linear_projection = LinearNorm(80)

    def call(self, inputs):
        x = self.lstm1(inputs)
        for conv in self.convolutions:
            x = layers.ReLU()(
                conv(x))
        outputs = self.lstm2(x)

        decoder_output = self.linear_projection(outputs)

        return decoder_output

class Postnet(layers.Layer):
    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = []
        self.convolutions.append(
            Sequential(
                [ ConvNorm(512,
                           kernel_size=5, stride=1,
                           padding="same",
                           scale=scale_tanh),
                  layers.Activation('tanh'),
                  layers.BatchNormalization() ])
        )
        for i in range(1, 5 - 1):
            self.convolutions.append(
            Sequential(
                [ ConvNorm(512,
                           kernel_size=5, stride=1,
                           padding="same",
                           scale=scale_tanh),
                  layers.Activation('tanh'),
                  layers.BatchNormalization() ])
            )

        self.convolutions.append(
            Sequential(
                [ ConvNorm(80,
                           kernel_size=5, stride=1,
                           padding="same",
                           scale=scale_linear),
                  layers.BatchNormalization() ])
        )

    def call(self, x):
        for i in range(len(self.convolutions) - 1):
            x = self.convolutions[i](x)

        x = self.convolutions[-1](x)

        return x

from keras.losses import mae as mae

from keras.losses import mse as mse

class AutoVC(Model):
    def __init__(self, inputs, name="AutoVC", dim_neck=16, dim_emb=32, dim_freq=16, **av):
        super(AutoVC, self).__init__(inputs, name=name, **av)
        self.Enc = Encoder(dim_neck,dim_emb,dim_freq)
        self.Dec = Decoder(dim_pre=512)
        self.postnet = Postnet()

        self.zoom = layers.RandomZoom(
            height_factor= (-.1,.1),
            width_factor = (-.05, .05))

        self.image0_tracker = tf.keras.metrics.Mean(name="image0")
        self.postnet_tracker = tf.keras.metrics.Mean(name="postnet")
        self.code_tracker = tf.keras.metrics.Mean(name="code")

        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

        self.dim_freq = dim_freq

        # loss function weights
        try:
            self.lam1 = av['lam1']
            self.lam2 = av['lam2']
            self.lam3 = av['lam3']
        except:
            self.lam1, self.lam2, self.lam3 = 1., 1., .1

        assert True

    def call(self, inputs):
        x = inputs['xim']
        c_org_ = inputs['embeds']
        c_targ_ =  inputs['embeds_']

        x = layers.Reshape((128,80))(x)

        c_org = layers.Reshape((32,))(c_org_)
        c_targ = layers.Reshape((32,))(c_targ_)

        c_org = layers.RepeatVector(128)(c_org)
        c_targ = layers.RepeatVector(128)(c_targ)
        codes_src_ = self.Enc(x,c_org)

        codes_src_ = layers.UpSampling1D(self.dim_freq)(codes_src_)
        codes_targ = layers.Concatenate(axis=-1)([codes_src_, c_targ])

        y_targ = self.Dec(codes_targ)
        y_postnet_targ = self.postnet(y_targ) + y_targ

        codes_recon1 = layers.UpSampling1D(
            self.dim_freq)(self.Enc(y_postnet_targ, c_targ)
        )

        returnvals = y_targ, y_postnet_targ, codes_src_, codes_recon1

        return returnvals

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            image_loss = self.lam1 * mse(y, y_pred[0])
            postnet_loss = self.lam2 * mse(y, y_pred[1])
            code_loss = self.lam3 * mae(y_pred[2], y_pred[3])
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
