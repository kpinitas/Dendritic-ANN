import cupy as cp

def categorical_cross_entropy(x,y, epsilon = 10**(-13)):
    x=cp.clip(x,epsilon,1.-epsilon)
    N = x.shape[0]

    return -cp.sum(y*cp.log(x+0.00001))/N

def loss(name='cre',x=None,y=None):
    assert x.all().tolist()!= None and y.all().tolist()!= None,"At least one \"None\" in input data"
    if name=='cre':
        return categorical_cross_entropy(x,y)
    