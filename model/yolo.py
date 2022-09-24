import tensorflow as tf
from tensorflow import keras
from models.common import blocks
from loss import SigmoidBin
import cfg

class Detect(keras.layers.Layer):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), n=3, **kwargs):  # detection layer
        super(Detect, self).__init__(**kwargs)
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [tf.zeros(1)] * self.nl  # init grid
        a = tf.cast(tf.reshape(tf.constant(anchors), (self.nl, -1, 2)), tf.float32)
        self.anchors = a  # shape(nl,na,2)
        self.anchor_grid = tf.reshape(a, (self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = [keras.layers.Conv2D(self.no * self.na, 1) for x in range(n)]  # output conv

    def call(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = tf.transpose(tf.reshape(x[i], (bs, ny, nx, self.na, self.no)), (0, 3, 1, 2, 4))

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny)
                y = tf.nn.sigmoid(x[i])
    
                xy = y[..., :2] * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                wh = wh[..., 2:4] ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                y = tf.concat([xy, wh, y[...,4:5]], 4)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = tf.concat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = tf.concat(z, 1)
        else:
            out = (tf.concat(z, 1), x)

        return out

    @staticmethod
    def _make_grid(nx=20, ny=20):
        nx, ny = tf.meshgrid(tf.range(nx), tf.range(ny))
        grid = tf.reshape(tf.stack([ny, nx], axis=2), (1, 1, ny, nx, 2))
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


class IDetect(keras.layers.Layer):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), n=3):  # detection layer
        super(IDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [tf.zeros((1))] * self.nl  # init grid
        a = tf.reshape(tf.cast(tf.constant(anchors), tf.float32), (self.nl, -1, 2))
        self.ancohors = tf.Variable(a, trainable=False)  # shape(nl,na,2)
        self.anchor_grid = tf.Variable(tf.reshape(a, (self.nl, 1, -1, 1, 1, 2)), trainable=False)  # shape(nl,1,na,1,1,2)
        self.m = [keras.layers.Conv2D(self.no * self.na, 1) for x in range(n)]  # output conv  # output conv
        
        self.ia = [blocks.ImplicitA() for x in range(n)]
        self.im = [blocks.ImplicitM(self.no * self.na) for _ in range(n)]

    def call(self, x):
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, ny, nx, _ = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = tf.transpose(tf.reshape(x[i], (bs, ny, nx, self.na, self.no)), (0, 3, 1, 2, 4))

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny)

                y =  tf.nn.sigmoid(x[i])
                y = tf.concat([(y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i],  #xy
                (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i], y[..., 4:]], axis=-1)  # wh
                z.append(tf.reshape(y, (bs, -1, self.no)))

        return x if self.training else (tf.concat(z, 1), x)
    
    def fuse(self):
        print("IDetect.fuse")
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            _,_, c2,c1 = self.m[i].weight[0].shape
            _,_, c2_,c1_ = self.ia[i].implicit.shape
            self.m[i].weights[-1].assign_add(tf.squeeze(tf.matmul(tf.reshape(self.m[i].weight[1], (c1,c2),self.ia[i].implicit.reshape(c2_,c1_)))))

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1,c2, _,_ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0,1)
            
    @staticmethod
    def _make_grid(nx=20, ny=20):
        nx, ny = tf.meshgrid(tf.range(nx), tf.range(ny))
        grid = tf.reshape(tf.stack([ny, nx], axis=2), (1, 1, ny, nx, 2))
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



class IKeypoint(keras.layers.Layer):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), nkpt=17, n=3, inplace=True, dw_conv_kpt=False):  # detection layer
        super(IKeypoint, self).__init__()
        self.nc = nc  # number of classes
        self.nkpt = nkpt
        self.dw_conv_kpt = dw_conv_kpt
        self.no_det=(nc + 5)  # number of outputs per anchor for box and class
        self.no_kpt = 3*self.nkpt ## number of outputs per anchor for keypoints
        self.no = self.no_det+self.no_kpt
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.flip_test = False
        self.grid = [tf.zeros((1))] * self.nl  # init grid
        a = tf.reshape(tf.cast(tf.constant(anchors), tf.float32), (self.nl, -1, 2))
        self.ancohors = tf.Variable(a, trainable=False)  # shape(nl,na,2)
        self.anchor_grid = tf.Variable(tf.reshape(a, (self.nl, 1, -1, 1, 1, 2)), trainable=False)  # shape(nl,1,na,1,1,2)
        self.m = [keras.layers.Conv2D(self.no_det * self.na, 1) for x in range(n)]  # output conv  # output conv
        
        self.ia = [blocks.ImplicitA() for i in range(n)]
        self.im = [blocks.ImplicitM(self.no_det * self.na) for _ in range(n)]

        
        if self.nkpt is not None:
            if self.dw_conv_kpt: #keypoint head is slightly more complex
                self.m_kpt = [keras.Sequential(
                                blocks.DWConv(k=3), blocks.Conv(x,x),
                                blocks.DWConv(k=3), blocks.Conv(x, x),
                                blocks.DWConv(k=3), blocks.Conv(x,x),
                                blocks.DWConv(k=3), blocks.Conv(x, x),
                                blocks.DWConv(k=3), blocks.Conv(x, x),
                                blocks.DWConv(k=3), keras.layers.Conv2D(x, self.no_kpt * self.na, 1)) for x in ch]
    
            else: #keypoint head is a single convolution
                self.m_kpt = [keras.layers.Conv2D(self.no_kpt * self.na, 1) for x in range(n)]


    def call(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            if self.nkpt is None or self.nkpt==0:
                x[i] = self.im[i](self.m[i](self.ia[i](x[i])))  # conv
            else :
                x[i] = tf.concat((self.im[i](self.m[i](self.ia[i](x[i]))), self.m_kpt[i](x[i])), axis=-1)

            bs, ny, nx, _ = x[i].shape  # x(bs,20,20,256) to x(bs,3,20,20,85)
            x[i] = tf.transpose(tf.reshape(x[i], (bs, ny, nx, self.na, self.no)), (0, 3, 1, 2, 4))
            x_det = x[i][..., :6]
            x_kpt = x[i][..., 6:]

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny)
                kpt_grid_x = self.grid[i][..., 0:1]
                kpt_grid_y = self.grid[i][..., 1:2]

                if self.nkpt == 0:
                    y = tf.nn.sigmoid(x[i])
                else:
                    y = tf.nn.sigmoid(x_det)

              
                xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                if self.nkpt != 0:
                    y[..., 6:] = (y[..., 6:] * 2. - 0.5 + self.grid[i].repeat((1,1,1,1,self.nkpt))) * self.stride[i]  # xy
                y = tf.concat((xy, wh, y[..., 4:]), -1)

                z.append(tf.reshape(y, (bs, -1, self.no)))

        return x if self.training else (tf.concat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        nx, ny = tf.meshgrid(tf.range(nx), tf.range(ny))
        grid = tf.reshape(tf.stack([ny, nx], axis=2), (1, 1, ny, nx, 2))
        return tf.cast(grid, tf.float32)



class IAuxDetect(keras.layers.Layer):
    stride = None  # strides computed during build
    export = False  # onnx export
    end2end = False
    include_nms = False
    concat = False

    def __init__(self, nc=80, anchors=(), n=3):  # detection layer
        super(IAuxDetect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [tf.zeros((1))] * self.nl  # init grid
        a = tf.reshape(tf.cast(tf.constant(anchors), tf.float32), (self.nl, -1, 2))
        self.ancohors = tf.Variable(a, trainable=False)  # shape(nl,na,2)
        self.anchor_grid = tf.Variable(tf.reshape(a, (self.nl, 1, -1, 1, 1, 2)), trainable=False)  # shape(nl,1,na,1,1,2)
        self.m1 = [keras.layers.Conv2D(self.no * self.na, 1) for x in range(self.nl)]  # output conv  # output conv
        self.m2 = [keras.layers.Conv2D(self.no * self.na, 1) for x in range(self.nl)]

        self.ia = [blocks.ImplicitA(x) for x in range(n)]
        self.im = [blocks.ImplicitM(self.no * self.na) for _ in range(n)]

    def call(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = tf.transpose(tf.reshape(x[i], (bs, ny, nx, self.na, self.no)), (0, 3, 1, 2, 4))
            
            x[i+self.nl] = self.m2[i](x[i+self.nl])
            x[i+self.nl] = tf.transpose(tf.reshape(x[i+self.nl], (bs, ny, nx, self.na, self.no)), (0, 3, 1, 2, 4))

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny)

                y = tf.nn.sigmoid(x[i])
                
                xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                xy = y[..., :2] * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                wh = y[..., 2:4] ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                y = tf.concat((xy, wh, y[..., 4:5]), axis=-1)
                z.append(tf.reshape(y, (bs, -1, self.no)))

        return x if self.training else (tf.concat(z, 1), x[:self.nl])

    
    
    def fuse(self):
        print("IAuxDetect.fuse")
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            _,_, c2,c1 = self.m[i].kernel.shape
            _,_, c2_,c1_ = self.ia[i].implicit.shape
            self.m[i].bias.assign_add(tf.squeeze(tf.matmul(tf.reshape(self.m[i].bias, (c1,c2),self.ia[i].implicit.reshape(c2_,c1_)))))


        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            _,_,c2,c1 = self.im[i].implicit.shape
            self.m[i].bias.assign(self.m[i].bias * tf.reshape(self.im[i].implicit, (c2)))
            self.m[i].kernel.assign(self.m[i].kernel * tf.transpose(self.im[i].implicit, (0,1)))

    @staticmethod
    def _make_grid(nx=20, ny=20):
        nx, ny = tf.meshgrid(tf.range(nx), tf.range(ny))
        grid = tf.reshape(tf.stack([ny, nx], axis=2), (1, 1, ny, nx, 2))
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


class IBin(keras.layers.Layer):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), n=3, bin_count=21):  # detection layer
        super(IBin, self).__init__()
        self.nc = nc  # number of classes
        self.bin_count = bin_count
        self.w_bin_sigmoid = SigmoidBin(bin_count=self.bin_count, min=0.0, max=4.0)
        self.h_bin_sigmoid = SigmoidBin(bin_count=self.bin_count, min=0.0, max=4.0)
        # classes, x,y,obj
        self.no = nc + 3 + \
            self.w_bin_sigmoid.get_length() + self.h_bin_sigmoid.get_length()   # w-bce, h-bce
            # + self.x_bin_sigmoid.get_length() + self.y_bin_sigmoid.get_length()
        
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [tf.zeros(1)] * self.nl  # init grid
        a = tf.cast(tf.reshape(tf.constant(anchors), (self.nl, -1, 2)), tf.float32)
        self.anchors = a  # shape(nl,na,2)
        self.anchor_grid = tf.reshape(a, (self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = [keras.layers.Conv2D(self.no * self.na, 1) for x in range(n)]  # output conv
        
        self.ia = [blocks.ImplicitA(x) for x in range(n)]
        self.im = [blocks.ImplicitM(self.no * self.na) for _ in range(n)]

    def call(self, x):

        #self.x_bin_sigmoid.use_fw_regression = True
        #self.y_bin_sigmoid.use_fw_regression = True
        self.w_bin_sigmoid.use_fw_regression = True
        self.h_bin_sigmoid.use_fw_regression = True
        
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = tf.transpose(tf.reshape(x[i], (bs, ny, nx, self.na, self.no)), (0, 3, 1, 2, 4))

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny)

                y = x[i].sigmoid()
                xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                #y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                

                #px = (self.x_bin_sigmoid.forward(y[..., 0:12]) + self.grid[i][..., 0]) * self.stride[i]
                #py = (self.y_bin_sigmoid.forward(y[..., 12:24]) + self.grid[i][..., 1]) * self.stride[i]

                pw = self.w_bin_sigmoid.call(y[..., 2:24]) * self.anchor_grid[i][..., 0]
                ph = self.h_bin_sigmoid.call(y[..., 24:46]) * self.anchor_grid[i][..., 1]

                #y[..., 0] = px
                #y[..., 1] = py
                y[..., 2] = pw
                y[..., 3] = ph
                
                y = tf.concat([xy, pw, ph, y[..., 46:]], dim=-1)
                
                z.append(tf.reshape(y, (bs, -1, y.shape[-1])))

        return x if self.training else (tf.concat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        nx, ny = tf.meshgrid(tf.range(nx), tf.range(ny))
        grid = tf.reshape(tf.stack([ny, nx], axis=2), (1, 1, ny, nx, 2))
        return tf.cast(grid, tf.float32)


input_ = keras.Input(shape=(224,224,3))
output = []
for idx in range(list(cfg.keys())[-1]):
    f, n, block, arg = cfg[idx]
    output.append(
        eval(f'blocks.{block}')(*arg, name=f'{idx}_{block}')(
            [output[i] for i in f] if isinstance(f, list) else (output[f] if f else input_))
    )
model = keras.Model(inputs=input_, outputs=output[-1])
