import tensorflow as tf
from tensorflow import keras
from . import common
import cfg


@keras.utils.register_keras_serializable()
class Detect(keras.layers.Layer):
    def __init__(self, nc=80, anchors=(), name='Detect', training=True, deploy=False, **kwargs):  # detection layer
        super(Detect, self).__init__(name=name, **kwargs)
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [tf.zeros(1)] * self.nl  # init grid
        a = tf.cast(tf.reshape(tf.constant(anchors), (self.nl, -1, 2)), tf.float32)
        self.anchors = tf.Variable(a, trainable=False, name='anchors')  # shape(nl,na,2)
        self.anchor_grid = tf.Variable(tf.reshape(a, (self.nl, 1, 1, 1, -1, 2)), trainable=False, name='anchors_grid')  # shape(nl,1,1,1,na,2)
        self.m = [keras.layers.Conv2D(self.no * self.na, 1, name=f'cv{i}') for i in range(self.nl)]  # output conv  # output conv
        self.training = training
        
    def call(self, x):
        
        z = [] 
        outputs = []
        for i in range(self.nl):
            output = self.m[i](x[i])  # conv
            bs, ny, nx, _ = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            output = tf.reshape(output, (-1, ny, nx, self.na, self.no))
            outputs.append(output)
            if not self.training:  # inference
                if self.grid[i].shape[1:3] != x[i].shape[1:3]:
                    self.grid[i] = self._make_grid(nx, ny)
                y = tf.nn.sigmoid(output)
    
                y = tf.concat([(y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i],  #xy
                (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i], y[..., 4:]], axis=-1)  # wh
                z.append(tf.reshape(y, (-1, ny*nx*self.na, self.no)))
        
        return tuple(outputs) if self.training else tuple(z)
    
    
    @staticmethod
    def _make_grid(nx=20, ny=20):
        grid_xy = tf.meshgrid(tf.range(ny), tf.range(nx))
        grid = tf.cast(tf.expand_dims(tf.stack(grid_xy, axis=-1), 2), tf.float32)
        return tf.cast(grid, tf.float32)
blocks.Detect = Detect

    
@keras.utils.register_keras_serializable()
class IDetect(keras.layers.Layer):
    stride = None
    def __init__(self, nc=80, anchors=(), name='IDetect', training=True, deploy=False, **kwargs):  # detection layer
        super(IDetect, self).__init__(name=name, **kwargs)
        self.deploy = deploy
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = int(len(anchors[0])//2)#int(len(anchors[0]) // 2)  # number of anchors
        self.grid = [tf.zeros((1))] * self.nl  # init grid
        a = tf.reshape(tf.cast(tf.constant(anchors), tf.float32), (self.nl, -1, 2))
        self.anchors = tf.Variable(a, trainable=False, name='anchors')  # shape(nl,na,2)
        self.anchor_grid = tf.Variable(tf.reshape(a, (self.nl, 1, 1, 1, -1, 2)), trainable=False, name='anchors_grid')  # shape(nl,1,1,1,na,2)
        self.m = [keras.layers.Conv2D(self.no * self.na, 1, name=f'cv{i}') for i in range(self.nl)]  # output conv  # output conv
        
        self.ia = [ImplicitA(name=f'ImplicitA{i}') for i in range(self.nl)]
        self.im = [ImplicitM(self.no * self.na, name=f'ImplicitM{i}') for i in range(self.nl)]
        self.training = training
         
    def call(self, x):
        z = []  # inference output
        outputs = [] # training output
        for i in range(self.nl):
            if self.deploy:
                output = self.m[i](x[i])
            else:
                output = self.m[i](self.ia[i](x[i]))  
                output = self.im[i](output)
            bs, ny, nx, _ = output.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            output = tf.cast(tf.reshape(output, (-1, ny, nx, self.na, self.no)), tf.float32)
            outputs.append(output)
            if not self.training:  # inference
                if self.grid[i].shape[2:4] != output.shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny)

                y =  tf.nn.sigmoid(output)
                y = tf.concat([(y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i],  #xy
                (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i], y[..., 4:]], axis=-1)  # wh
                z.append(tf.reshape(y, (-1, ny*nx*self.na, self.no))) 
        return tuple(outputs) if self.training else tuple(z)
    
    def get_config(self):
        config = super(IDetect, self).get_config()
        config.update({'nc': self.nc, 'anchors':self.anchors.numpy().reshape(self.nl, self.na*2)})
        return config
    def switch_to_deploy(self):
        
        # fuse ImplicitA and Convolution
        for i in range(self.nl):
            kernel = tf.squeeze(self.m[i].kernel) # (1, 1, c1, c2(num_cls*a)) => (c1, c2)
            kernel = tf.transpose(kernel, [1,0]) # (c1, c2) # => (c2, c1)
            implicit_ia = tf.squeeze(self.ia[i].implicit)[...,None]  # (1, 1, 1, c1) => (c1, 1)
            fused_conv_bias = tf.matmul(kernel, implicit_ia) # (c2, 1)
            
            self.m[i].bias.assign_add(tf.squeeze(fused_conv_bias)) # add fused_conv_bias to the bias vector
            
        # fuse ImplicitM and Convolution
        for i in range(self.nl):
            implicit_m = tf.squeeze(self.im[i].implicit)
            self.m[i].bias.assign(self.m[i].bias * implicit_m)
            self.m[i].kernel.assign(self.m[i].kernel * self.im[i].implicit)
        
        self.__delattr__('im')
        self.__delattr__('ia')
        print("IDetect fused")
    @staticmethod
    def _make_grid(nx=20, ny=20):
        grid_xy = tf.meshgrid(tf.range(ny), tf.range(nx))
        grid = tf.cast(tf.expand_dims(tf.stack(grid_xy, axis=-1), 2), tf.float32)
        return tf.cast(grid, tf.float32)
blocks.IDetect = IDetect


@keras.utils.register_keras_serializable()
class IAuxDetect(keras.layers.Layer):
    stride = None
    def __init__(self, nc=80, anchors=(), deploy=False, training=True, name='IAuxDetect'):  # detection layer
        super(IAuxDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.deploy = deploy
        self.grid = [tf.zeros((1))] * self.nl  # init grid
        a = tf.reshape(tf.cast(tf.constant(anchors), tf.float32), (self.nl, -1, 2))
        self.anchors = tf.Variable(a, trainable=False)  # shape(nl,na,2)
        self.anchor_grid = tf.Variable(tf.reshape(a, (self.nl, 1, 1, 1,-1, 2)), trainable=False)  # shape(nl,1,na,1,1,2)
        self.m = [keras.layers.Conv2D(self.no * self.na, 1, name=f'm{i}') for i in range(self.nl)]  # output conv  # output conv
        self.m2 = [keras.layers.Conv2D(self.no * self.na, 1, name=f'm{i}') for i in range(self.nl)]

        self.ia = [ImplicitA(name=f'ImplicitA{i}') for i in range(self.nl)]
        self.im = [ImplicitM(self.no * self.na, name=f'ImplicitM{i}') for i in range(self.nl)]
        self.training=training
    def call(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        outputs = []
        aux_outputs = []
        
        for i in range(self.nl):
            if self.deploy:
                output = self.m[i](x[i])
            else:
                output = self.m[i](self.ia[i](x[i]))  # conv
                output = self.im[i](output)
            bs, ny, nx, _ = output.shape  # x(bs,20,20,255) to x(bs,20,20,3,85)
            output = tf.reshape(output, (-1, ny, nx, self.na, self.no))
            outputs.append(output)
            if not self.deploy:
                aux_output = self.m2[i](x[i+self.nl])
                aux_output = tf.reshape(aux_output, (-1, ny, nx, self.na, self.no))
                aux_outputs.append(aux_output)
            
            if not self.training:  # inference
                if self.grid[i].shape[2:4] != output.shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny)

                y = tf.nn.sigmoid(output)
                
                xy, wh, conf = tf.split(y, (2, 2, self.nc + 1), axis=4)
                xy = y[..., :2] * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                wh = y[..., 2:4] ** 2 * (4 * self.anchor_grid[i])  # new wh
                y = tf.concat((xy, wh, y[..., 4:]), axis=-1)
                z.append(tf.reshape(y, (-1, ny*nx*self.na, self.no)))

        return (*outputs, *aux_outputs) if self.training else (*z,*[tf.zeros_like(z[0]) for i in range(len(z))])


    def get_config(self):
        config = super(IAuxDetect, self).get_config()
        config.update({'nc': self.nc, 'anchors':self.anchors.numpy().reshape(self.nl, self.na*2)})
        return config
    
    def switch_to_deploy(self):
        
        # fuse ImplicitA and Convolution
        for i in range(self.nl):
            kernel = tf.squeeze(self.m[i].kernel) # (1, 1, c1, c2(num_cls*a)) => (c1, c2)
            kernel = tf.transpose(kernel, [1,0]) # (c1, c2) # => (c2, c1)
            implicit_ia = tf.squeeze(self.ia[i].implicit)[...,None]  # (1, 1, 1, c1) => (c1, 1)
            fused_conv_bias = tf.matmul(kernel, implicit_ia) # (c2, 1)
            
            self.m[i].bias.assign_add(tf.squeeze(fused_conv_bias)) # add fused_conv_bias to the bias vector
            
        # fuse ImplicitM and Convolution
        for i in range(self.nl):
            implicit_m = tf.squeeze(self.im[i].implicit)
            self.m[i].bias.assign(self.m[i].bias * implicit_m)
            self.m[i].kernel.assign(self.m[i].kernel * self.im[i].implicit)
        
        self.__delattr__('im')
        self.__delattr__('ia')
        print("IDetect fused")

    @staticmethod
    def _make_grid(nx=20, ny=20):
        grid_xy = tf.meshgrid(tf.range(ny), tf.range(nx))
        grid = tf.cast(tf.expand_dims(tf.stack(grid_xy, axis=-1), 2), tf.float32)
        return tf.cast(grid, tf.float32)
    def convert(self, z):
        z = tf.concat(z, axis=1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = tf.constant([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=tf.float32)
        box @= convert_matrix                          
        return (box, score)
blocks.IAuxDetect = IAuxDetect


def build_model(cfg, training=True, input_shape=(640,640), deploy=False, custom_model=False, name='yolo_model'):
    cfg['head'][-1] = [cfg['head'][-1][0], cfg['head'][-1][1], cfg['head'][-1][2], [cfg['nc'], cfg['anchors']]]
    
    input_ = keras.Input(shape=(*input_shape,3))
    output = []
    for idx, values, in enumerate(cfg['backbone'] + cfg['head']):
        
        f, n, block, arg = values
        output.append(
            eval(f'{common.blocks}.{block}')(*arg, name=f'{idx}_{block}', deploy=deploy)(
                [output[i] for i in f] if isinstance(f, list) else (input_ if len(output)==0 else output[f])))
        
    model = custom_model( inputs=input_, outputs=output[-1], name=name) if custom_model else keras.Model(
        inputs=input_, outputs=[output[-1]], name=name)
    
    stride = [input_shape[0]/x.shape[2]  for x in output[-1]]
    model.layers[-1].stride = stride
    model.stride = tf.Variable(stride, trainable=False)
    if not training:
        for layer in model.layers:
            layer.training = False
    
    return model
