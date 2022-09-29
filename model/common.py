
import tensorflow as tf
from tensorflow import keras
import numpy as np



##### basic ####
class blocks:
    CommonBlocks = True

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Pad(keras.layers.Layer):
    # Pad inputs in spatial dimensions 1 and 2
    def __init__(self, pad):
        super().__init__()
        if isinstance(pad, int):
            self.pad = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
        else:  # tuple/list
            self.pad = tf.constant([[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]])

    def call(self, inputs):
        return tf.pad(inputs, self.pad, mode='constant', constant_values=0)



def MP(name='MP',**kwargs):
    """urapping maxpooling2D layer"""
    def layer(x):
        x = keras.layers.MaxPooling2D(name=f'{name}')(x)
        return x
    return layer
blocks.MP = MP


def Upsample(size, interpolation, name, **kwargs):
    """ urapping UpSampling2D layers """
    def layer(x):
        return keras.layers.UpSampling2D(size, interpolation=interpolation, name=f'{name}')(x)
    return layer
blocks.Upsample = Upsample

def Sp(k=3, s=1):
    def layer(x):
       x = keras.layers.MaxPooling2D(pool_size=k, strides=s, padding='same')(x)
       return x
    return layer
blocks.Sp = Sp


@keras.utils.register_keras_serializable()
class ReOrg(keras.layers.Layer):
    """" reorganize the input tensor from (b, w, h, c) to (b, w/2, h/2, 4c)"""
    def __init__(self, name, dimension=-1, deploy=False, **kwargs):
        super(ReOrg, self).__init__(name=name, **kwargs)
    def call(self, x): # x(b,w,h,c) -> y(b,w/2,h/2,4c) 
        x =  tf.concat(
            [x[:, ::2, ::2,:], x[:, 1::2, ::2,:], x[:, ::2, 1::2, :], x[..., 1::2, 1::2,:]], -1)
        return x
 blocks.ReOrg = ReOrg


@keras.utils.register_keras_serializable()
def Concat(dimension=-1, name='_', **kwargs):
    """ concat a list of tensor in the filters dimension """
    def layer(x):
        x = keras.layers.Concatenate(name=f'{name}', axis=dimension)(x)
        return x
    return layer
blocks.Concat


@keras.utils.register_keras_serializable()
class Shortcut(keras.layers.Layer):
    """ It just merges tow tensor """
    def __init__(self, dimension=0, name='_', **kwargs):
        super(Shortcut, self).__init__(name=name, *kwargs)
    def call(self, x):
        return x[0]+x[1]
blocks.Shortcut



@keras.utils.register_keras_serializable()
class Conv(keras.layers.Layer):
    def __init__(self, filters, kernel_size=1, strides=1, padding=None, groups=1,
                 act=True, name='_', deploy=False, **kwargs):
        super(Conv, self).__init__(name=name, **kwargs)
       
        self.deploy = deploy
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.groups = groups
        
        if strides==1:
            self.conv = keras.layers.Conv2D(filters, kernel_size, strides, padding='same', 
                               groups=groups, use_bias=False, name=f'cv')
        else:
            self.conv = keras.Sequential([
                    Pad(autopad(kernel_size,None)), 
                    keras.layers.Conv2D(filters, kernel_size, strides, padding='Valid', 
                               groups=groups, use_bias=False, name=f'cv')
                ])
        self.bn = keras.layers.BatchNormalization(name=f'bn') if not deploy else None
        self.act = keras.activations.swish if act is True else (act if isinstance(act, keras.acivations) else tf.identity)
            
    def call(self, x):
        return self.act(self.fused_conv(x)) if self.deploy else self.act(self.bn(self.conv(x)))
    
    def get_config(self):
        config = super(Conv, self).get_config()
        config.update({'filters': self.filters, 'kernel_size': self.kernel_size, 'padding':self.padding,
                       'strides': self.strides, 'groups': self.groups})
        return config
    
blocks.Conv = Conv




@keras.utils.register_keras_serializable()
class DownC(keras.layers.Layer):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, filters, n=1, kernel_size=2, name='DownC', deploy=False, **kwargs):
        super(DownC, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.n = n
        self.kernel_size = kernel_size
    def build(self, input_shape):
        self.cv1 = Conv(input_shape[-1], 1, 1, name=f'cv1')
        self.cv2 = Conv(self.filters//2, 3, self.kernel_size, name=f'cv2')
        self.cv3 = Conv(self.filters//2, 1, 1, name=f'cv3')
        self.mp = keras.layers.MaxPooling2D(pool_size=self.kernel_size, strides=self.kernel_size, name=f'max_pool')
        
    def call(self, x):
        inputs = self.cv2(self.cv1(x))
        return tf.concat([inputs, self.cv3(self.mp(x))], -1)
    
    def get_config(self):
        config = super(DownC, self).get_config()
        config.update({'filters': self.filters, 'n': self.n, 'kernel_size':self.kernel_size})
        return config
blocks.DownC



@keras.utils.register_keras_serializable()
class SPPCSPC(keras.layers.Layer):
     # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, filters, n=1, shortcut=False, groups=1, e=0.5, kernel_size=(5, 9, 13), name='SPPCSPC', deploy=False, **kwargs):
        super(SPPCSPC, self).__init__(name=name, **kwargs)
        c_ = int(2 * filters * e)  # hidden channels
        self.filters = filters
        self.n = n
        self.groups = groups
        self.e = e
        self.kernel_size = kernel_size
        self.deploy = deploy
        
        self.cv1 = Conv(c_, 1, 1, deploy=deploy, groups=groups, name=f'cv1')
        self.cv2 = Conv(c_, 1, 1, deploy=deploy, groups=groups, name=f'cv2')
        self.cv3 = Conv(c_, 3, 1, deploy=deploy, groups=groups, name=f'cv3')
        self.cv4 = Conv(c_, 1, 1, deploy=deploy, groups=groups, name=f'cv4')
        self.cv5 = Conv(c_, 1, 1, deploy=deploy, groups=groups, name=f'cv5')  # transition layer
        self.cv6 = Conv(c_, 3, 1, deploy=deploy, groups=groups, name=f'cv6')
        self.cv7 = Conv(filters, 1, 1, deploy=deploy, groups=groups, name=f'cv7') # transition layer
        self.m = [keras.layers.MaxPooling2D(pool_size=n, strides=1, padding='same', name=f'max_pool_{i}') for i, n in enumerate(kernel_size)]
    def call(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        spp_output = tf.concat([x1] + [m(x1) for m in self.m], axis=-1)
        csp_inp1 = self.cv6(self.cv5(spp_output)) # transition
        csp_inp2 = self.cv2(x)
        return self.cv7(tf.concat([csp_inp1, csp_inp2], axis=-1)) # concatenation and transition
    
    def get_config(self):
        config = super(SPPCSPC, self).get_config()
        config.update({'filters': self.filters, 'n': self.n, 'groups': self.groups,
                       'e': self.e, 'kernel_size':self.kernel_size})
        return config

blocks.SPPCSPC = SPPCSPC



##### yolor #####

@keras.utils.register_keras_serializable()
class ImplicitA(keras.layers.Layer):
    def __init__(self, mean=0., std=.02, name='ImplicitA', **kwargs):
        super(ImplicitA, self).__init__(name=name, **kwargs)
        self.mean = mean
        self.std = std
       
    def build(self, input_shape):    
        self.implicit = tf.Variable(
            initial_value=tf.random_normal_initializer(
                mean=self.mean, stddev=self.std)(shape=(1, 1, 1, input_shape[-1])),
            trainable=True, name=self.name)
        
    def call(self, x):
        return tf.cast(x, self.implicit.dtype) + self.implicit
    
    def get_config(self):
        config = super(ImplicitA, self).get_config()
        config.update({'mean': self.mean, 'std':self.std})
        return config
blocks.ImplicitA = ImplicitA


@keras.utils.register_keras_serializable()
class ImplicitM(keras.layers.Layer):
    def __init__(self, filters, mean=0., std=.02, name='ImplicitM', **kwargs):
        super(ImplicitM, self).__init__(name=name, **kwargs)
        self.filters= filters
        self.mean = mean
        self.std = std
        
        self.implicit = tf.Variable(
            initial_value=tf.random_normal_initializer(
                mean=self.mean, stddev=self.std)(shape=(1, 1, 1, filters)),
            trainable=True, name=name)

    def call(self, x):
        return tf.cast(x, self.implicit.dtype) * self.implicit
    
    def get_config(self):
        config = super(ImplicitM, self).get_config()
        config.update({'filters': self.filters, 'mean': self.mean, 'std': self.std})
        return config
blocks.ImplicitM = ImplicitM    

    
##### end of yolor #####


##### repvgg #####
@keras.utils.register_keras_serializable()
class RepConv(keras.layers.Layer):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697
    def __init__(self, filters, kernel_size=3, strides=1, padding=None, groups=1,
                 act=True, deploy=False, name='RepConv', **kwargs):
        super(RepConv, self).__init__(name=name, **kwargs)
        assert kernel_size == 3
        self.deploy = deploy
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.groups = groups
        
        self.act = keras.activations.swish if act is True else (act if isinstance(act, keras.Activations) else tf.Identity)
       
        
        if deploy:
            self.rbr_reparam = keras.layers.Conv2D(filters, kernel_size, strides, padding='same', 
                                            groups=groups, use_bias=True, name=f'{name}_conv3x3_reparam')
        else:
            self.rbr_dense = keras.Sequential([
                keras.layers.Conv2D(filters, kernel_size, strides, padding='same', 
                                            groups=groups, use_bias=False, name=f'{name}_conv3x3'),
                keras.layers.BatchNormalization(name=f'{name}_bn1')], name=f'conv3x3_bn1')

            self.rbr_1x1 = keras.Sequential([
                keras.layers.Conv2D(filters, 1, strides, padding='valid', groups=groups, use_bias=False, 
                name=f'{name}_conv1x1'),
                keras.layers.BatchNormalization(name=f'{name}_bn2')], name=f'conv1x1_bn')
    def build(self, input_shape):
            self.rbr_identity = keras.layers.BatchNormalization(name=f'identity') if self.filters==input_shape[-1] else None
            
    def call(self, x):    
        if self.deploy:
            return self.act(self.rbr_reparam(x))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)
        return self.act(self.rbr_dense(x) + self.rbr_1x1(x) + id_out)
    
    def get_config(self):
        config = super(RepConv, self).get_config()
        config.update({'filters': self.filters, 'kernel_size':self.kernel_size,
                       'strides': self.strides, 'groups': self.groups})
        return config
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return tf.pad(kernel1x1, [[1,1],[1,1],[0,0],[0,0]])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, keras.Sequential):
            kernel = branch.layers[0].kernel
            moving_mean = branch.layers[1].moving_mean
            moving_var = branch.layers[1].moving_variance
            gamma = branch.layers[1].gamma
            beta = branch.layers[1].beta
            eps = branch.layers[1].epsilon
        else:
            assert isinstance(branch, keras.layers.BatchNormalization)
            if not hasattr(self, "id_tensor"):
                _,_,in_channels,_  = self.rbr_dense.layers[0].kernel.shape
                
                input_dim = in_channels // self.groups
                kernel_value = np.zeros(
                    (3, 3, in_channels, input_dim), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = tf.convert_to_tensor(kernel_value)
            kernel = self.id_tensor
            moving_mean = branch.running_mean
            moving_var = branch.running_variance
            gamma = branch.gamma
            beta = branch.beta
            eps = branch.epsilon
        
        std = tf.sqrt(moving_var + eps)
        t = tf.reshape(gamma / std, (1, 1, 1, -1))
        return kernel * t, beta - moving_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = keras.layers.Conv2D(self.filters, self.rbr_dense.layers[0].kernel_size,
                                     self.rbr_dense.layers[0].strides,
                                     padding=self.rbr_dense.layers[0].padding, use_bias=True)
       
        self.rbr_reparam(tf.ones((1, 10, 10, self.rbr_dense.layers[0].kernel.shape[-2])))
        self.rbr_reparam.kernel.assign(kernel)
        self.rbr_reparam.bias.assign(bias)
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True
        print('RepConv fussed')

blocks.RepConv
