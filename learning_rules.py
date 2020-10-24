
import cupy as cp
from activations import activation


######### Learning strategiess ##########

def error_minimization(W,b,zeta,a,prev_layer,activation_func,den_activation, y,w=None,d=None,y_pred=None):
    dW={}
    dB={}
    delta={}
    try:
         batch_size = y.shape[1]
    except IndexError:
         batch_size=1
         y=cp.reshape(y,(y.shape[0],batch_size))
      
    is_last_layer  = (type(w) == type(d) )and (type(d) == type(None))
    
    if is_last_layer:
         
         delta['s'] = cp.subtract(a['s'],y)
         dB['s'] = (1/batch_size)*cp.sum(delta['s'],axis=1)
         dB['s']= cp.reshape(dB['s'],(dB['s'].shape[0],1,1))
         
         delta['s'] = cp.reshape(delta['s'],(delta['s'].shape[0],1,delta['s'].shape[1]) )
         
         dW['s']=(1/batch_size)* cp.einsum('nik,kjn->nij',delta['s'],a['d'].T) 
         
    else:
         w=cp.array(w)
         
         deltaW = cp.einsum('nik,kij->nj',w.T,d)
         deltaW=cp.reshape(deltaW,(deltaW.shape[0],1,deltaW.shape[1]))
         a_der = activation(str(activation_func)+'_der',zeta['s'])
         
         delta['s']=cp.multiply(deltaW,a_der)
         dB['s'] = (1/batch_size)*cp.sum(delta['s'].squeeze(), axis=1)
         dB['s']= cp.reshape(dB['s'],(dB['s'].shape[0],1,1))
         dW['s']=(1/batch_size)* cp.einsum('nik,kjn->nij',delta['s'],a['d'].T)
         
    
    deltaW=cp.einsum('nik,kij->knj',W['s'].T,delta['s']) 
    a_der = activation(den_activation+'_der',zeta['d'])
    delta['d']=cp.multiply(deltaW,a_der)
    dB['d'] = (1/batch_size)*cp.sum(delta['d'], axis=2)
    dB['d']= cp.reshape(dB['d'],(dB['d'].shape[0],dB['d'].shape[1],1))
    dW['d']=(1/batch_size)*cp.dot(delta['d'],prev_layer.T)
    return [dW,dB,delta]


def hebbian_rule(W,b,zeta,a,prev_layer,activation_func,den_activation, y,w=None,d=None):
    dW={}
    dB={}
    delta = None
    
    try:
         batch_size = y.shape[1]
    except IndexError:
         batch_size=1
         y=cp.reshape(y,(y.shape[0],batch_size))
      
    y = cp.argmax(y,axis=0).reshape((1,y.shape[1]))
    
    a['s'] = cp.reshape(a['s'], (a['s'].shape[0],1,a['s'].shape[1]))
    out_in = cp.einsum('nij,nkj->nik', a['s'],a['d'])
    out_w = cp.einsum('nik,nij->nkj',a['s'],W['s'])
    out_w_out = cp.einsum('nik,nji->njk', out_w,a['s'])
    dW['s'] = (1/batch_size)*(out_in-out_w_out) 
    
    out_b = cp.einsum('nik,nij->nkj',a['s'],b['s'])
    out_b_out = cp.einsum('nik,nji->njk', out_b,a['s'])
    
    dB['s'] = (1/batch_size)*cp.sum(y, axis=1)
    dB['s']= cp.reshape(dB['s'],(dB['s'].shape[0],1,1))

    
    # prev_layer = cp.reshape(prev_layer,(prev_layer.shape[0],1,prev_layer.shape[1]))
    
    out_in = cp.einsum('nij,kj->nik',a['d'],prev_layer)
    out_w = cp.einsum('nik,nij->nkj',a['d'],W['d'])
    out_w_out = cp.einsum('nik,nji->njk', out_w,a['d'])
    dW['d'] = (1/batch_size)*(out_in-out_w_out) 
    
    out_b = cp.einsum('nik,nij->nkj',a['d'],b['d'])
    out_b_out = cp.einsum('nik,nji->njk', out_b,a['d'])
    dB['d'] =(out_in-out_b_out) 
    dB['d'] = (1/batch_size)*cp.sum(dB['d'], axis=2)
    dB['d']= cp.reshape(dB['d'],(dB['d'].shape[0],dB['d'].shape[1],1))
    
    return [dW,dB,delta] 


########## Update Strategies ########## 

def  gradient_descent(lr,W,b,dW,dB):
    for key in b.keys():
        
        b[key]-=lr*dB[key]
        W[key]-=lr*dW[key]       
        
    return [W,b]





########## Wrapper ##########

learning_rule={
                'classic' :{'calc':error_minimization,
                            'update': gradient_descent
                            },
                
                'hebbian' :{ 'calc': hebbian_rule,
                             'update': gradient_descent
                          }
                 
    
    }