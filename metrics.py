import tensorflow as tf

def dice_coefficient(logits, target, loss_type='jaccard', smooth=1e-5):
    logits = tf.reshape(logits, shape=[-1])
    target = tf.reshape(target, shape=[-1])

    inse = tf.reduce_sum(logits * target)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(logits * logits)
        r = tf.reduce_sum(target * target)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(logits)
        r = tf.reduce_sum(target)
    else:
        raise Exception("Unknow loss_type")

    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice, name='dice_coefficient')
    return dice
