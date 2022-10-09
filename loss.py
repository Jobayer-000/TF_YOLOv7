@keras.utils.register_keras_serializable()
class ComputeLoss(keras.losses.Loss):
    def __init__(self, hyp, anchors, nc, name='YoloLoss', **kwargs):
        super(ComputeLoss, self).__init__(**kwargs)
        self.anchors = anchors
        self.hyp = hyp
        self.nl = len(anchors)
        self.na = len(anchors[0])
        self.nc = nc
        self.balance = [4.0, 1.0, .4] if len(self.anchors) == 3 else [4.0, 1.0, 0.4, 0.1]
        self.bce_conf = self._WCEWithLogist(label_smoothing=self.hyp['label_smoothing'],
                                            pos_weight=self.hyp['pos_weight'])
        self.bce_class = self._WCEWithLogist(label_smoothing=self.hyp['label_smoothing'],
                                            pos_weight=self.hyp['pos_weight'])
    def call(self, y_true, y_pred):
        iou_loss_all = obj_loss_all = class_loss_all = 0.
        balance = self.balance  # P3-5 or P3-6
        tcls, tbox, indices, anchors = self.targets(y_true, y_pred)
        #return true_class, true_box, anchors
        for i, pred in enumerate(y_pred):
            tobj = tf.zeros_like(pred[...,4])
            ps = tf.gather_nd(pred, tf.stack(indices[i],1))
            pxy = tf.nn.sigmoid(ps[:, :2]) * 2. - 0.5
            pwh = (tf.nn.sigmoid(ps[:, 2:4]) * 2) ** 2 * anchors[i]
            pred_box = tf.concat([pxy, pwh], -1)
          
            if len(indices[i][0])>1:
                if self.nc > 1:
                    t = tf.one_hot(tf.cast(tcls[i], tf.int32), depth=self.nc)
                    class_loss_all += self.bce_class(t, ps[..., 5:])
                iou = bbox_iou(pred_box, tbox[i], xyxy=False, ciou=True)
               
                iou_loss_all += tf.reduce_mean(1 - iou) 
                tobj = tf.tensor_scatter_nd_update(tobj, tf.stack(indices[i],1), iou)
              
                conf_loss = self.bce_conf(tobj, pred[...,4])
                obj_loss_all += conf_loss * self.balance[i]
               # to balance the 3 loss
      
        if self.reduction == 'auto':
            return iou_loss_all*self.hyp['box'] + obj_loss_all*self.hyp['obj'] + class_loss_all*self.hyp['cls']
                
        else:
            return (iou_loss_all*self.hyp['box'] + obj_loss_all*self.hyp['obj']+
                class_loss_all*self.hyp['cls']) * tf.cast(tf.shape(tobj)[0], tf.float32)
        
    
    
    def _WCEWithLogist(self, label_smoothing=0, pos_weight=1):
        def smooth_labels(labels, factor=0.1):
            labels *= (1. - factor)
            labels += (factor / tf.cast(tf.shape(labels)[1], tf.float32))
            return labels
        def loss(y_true, y_pred):
            y_true = smooth_labels(y_true, label_smoothing)
            return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight=1))
        return loss
    
    def targets(self, labels, p):
        shape =  tf.shape(labels)
        b, n, c = shape[0], shape[1], shape[2]
        labels = tf.reshape(
            tf.concat(
                [tf.tile(tf.range(b, dtype=tf.float32)[...,None], [1, n])[...,None], labels], -1), [-1, c+1])
        labels = labels[labels[...,3]!=0] # filter padded labels
        labels = tf.tile(tf.expand_dims(labels, 0), [self.na, 1, 1]) # => n_anchor * n_gt * 6
        a  = tf.tile(
                    tf.range(self.na, dtype=tf.float32)[...,None],
                    (1, tf.shape(labels)[1]) # => n_anchor * n_gt * 1, holds anchors indices
                )[...,None]
        labels = tf.concat([labels, a], -1) # append anchors indices
        tcls, tbox, indices, anch =  [],[],[],[]
      
        gain = tf.ones(7, tf.float32)
        off = tf.constant([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], tf.float32)
        g = 0.5  # offset
        for i in range(self.nl):
            anchors = self.anchors[i]
            grid_size = tf.cast(p[i].shape[1], tf.int32)
            y_true = tf.zeros([grid_size, grid_size, self.na, 6], tf.float32)
            gain = tf.tensor_scatter_nd_update(gain, [[1], [2], [3], [4]], [grid_size] * 4)
        
            t = labels * gain  # label coordinator now is the same with anchors
            
            if len(labels)>0:
                gt_wh = t[..., 3:5]  # n_anchor * n_gt * 2
                if self.hyp['assign_method'] == 'wh':
                    assert self.hyp['anchor_t'] > 1, 'threshold is totally different for wh and iou assign'
                    anchors = tf.expand_dims(anchors, 1)  # => n_anchor * 1 * 2
                    r = gt_wh / anchors  # => n_anchor * n_gt * 2
                    j = tf.reduce_max(tf.math.maximum(r, 1 / r),
                                       axis=2) < self.hyp['anchor_t']  # => n_anchor * n_gt
       
                elif self.hyp['assign_method'] == 'iou':
                    assert self.hyp['assign_method'] < 1, 'threshold is totally different for wh and iou assign'
                    #box_wh = tf.expand_dims(gt_wh, 0)  # => 1 * n_gt * 2
                    box_area = box_wh[..., 0] * box_wh[..., 1]  # => 1 * n_gt
                    anchors = tf.cast(anchors, tf.float32)  # => n_anchor * 2
                    anchors = tf.expand_dims(anchors, 1)  # => n_anchor * 1 * 2
                    anchors_area = anchors[..., 0] * anchors[..., 1]  # => n_anchor * 1
                    inter = tf.math.minimum(anchors[..., 0], box_wh[..., 0]) * tf.math.minimum(anchors[..., 1],
                                                                                   box_wh[..., 1])  # n_gt * n_anchor
                    iou = inter / (anchors_area + box_area - inter + 1e-9)
                    j = iou > self.hyp['anchor_t']
                else:
                    raise ValueError
                
                t = t[j] # filters
                
                gxy = t[..., 1:3]  # n_matched * 2
                matched = (gxy % 1. < g) & (gxy > 1.)
                j, k = matched[:, 0], matched[:, 1]
                matched = ((gain[1:3] - gxy) % 1. < g) & (gxy < tf.expand_dims(gain[1:3], 0) - 1.)
                l, m = matched[:, 0], matched[:, 1]
                t = tf.concat([t, t[j], t[k], t[l], t[m]], axis=0)
                offset = tf.zeros_like(gxy)
                offset = g * tf.concat(
                    [ offset,   offset[j] + off[1], offset[k] + off[2], offset[l] + off[3], offset[m] + off[4]],
                    axis=0)

            else:
                offset= tf.zeros_like(t[:, 1:3])
                t = labels[0]
            b, xy, wh, clss, a = tf.split(t, (1,2,2,1,1), axis=-1)
            gij = tf.cast(xy - offset, tf.int32)
            indices.append((tf.cast(t[...,0], tf.int32), tf.clip_by_value(gij[:, 1], 0, grid_size-1),
                                    tf.clip_by_value(gij[:, 0], 0, grid_size-1), tf.cast(t[...,-1], tf.int32)))
            tbox.append(tf.concat([xy - tf.cast(gij, tf.float32), wh], -1))
            tcls.append(tf.squeeze(clss))
            anch.append(tf.gather(anchors[:,0,:], tf.cast(t[...,-1], tf.int32)))
            
        return  tcls, tbox, indices, anch
    def get_config(self):
        config = super(ComputeLoss, self).get_config()
        config.update({'hyp': self.hyp, 'anchors': self.anchors, 'nc': self.nc})
        return config
