# -*- coding: utf-8 -*-

from maprop.energy.qpthlocal.qp import QPFunction
from maprop.energy.qpthlocal.qp import QPSolvers
from maprop.energy.qpthlocal.qp import make_gurobi_model

from maprop.energy.sgd_learner import *
from sklearn.metrics import mean_squared_error as mse

from maprop.energy.ip_model_whole import IPOfunc

import datetime
from maprop.energy.KnapsackSolving import *
import pandas as pd
from collections import defaultdict
import numpy as np
import random
import signal
from contextlib import contextmanager
from scipy.linalg import LinAlgError
import torch
from torch import nn, optim


class TimeoutException(Exception):
    pass


class MultilayerRegression(nn.Module):
    def __init__(self,input_size, hidden_size,target_size=1,
     num_layers=1):
        super(MultilayerRegression, self).__init__()
        if num_layers>1:

            net_layers = [nn.Linear(input_size, hidden_size), nn.Dropout()]#,
             # nn.ReLU()]
            for hidden in range(num_layers-2):
                net_layers.append(nn.Linear(hidden_size, hidden_size))
                net_layers.append(nn.Dropout())
                #net_layers.append(nn.ReLU())
            net_layers.append(nn.Linear(hidden_size,target_size))
            #net_layers.append(nn.ReLU())
            self.net = nn.Sequential(*net_layers)
        else:
            self.net = nn.Linear(input_size, target_size)

    def forward(self, x):
        return self.net(x)


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def make_matrix_qp(nbMachines,nbTasks,nbResources,MC,U,D,E,L,P,idle,up,down,q,**h):
    # nbMachines: number of machine
    # nbTasks: number of task
    # nb resources: number of resources
    # MC[m][r] resource capacity of machine m for resource r 
    # U[f][r] resource use of task f for resource r
    # D[f] duration of tasks f
    # E[f] earliest start of task f
    # L[f] latest end of task f
    # P[f] power use of tasks f
    # idle[m] idle cost of server m
    # up[m] startup cost of server m
    # down[m] shut-down cost of server m
    # q time resolution
    # timelimit in seconds

    # print("number of machines %d, number of tasks %d, number of resources %d"%(nbMachines,nbTasks,nbResources))
    Machines = range(nbMachines)
    Tasks = range(nbTasks)
    Resources = range(nbResources)
    N = 1440//q

    ### G and h
    G1 = torch.zeros((nbMachines*N ,nbTasks*nbMachines*N))
    h1 = torch.zeros(nbMachines*N)
    F = torch.zeros((N,nbTasks*nbMachines*N))
    for m in Machines:
        for t in range(N):
            h1[m*N+t] = MC[m][0]
            for f in Tasks:
                c_index = (f*nbMachines+m)*N 
                G1[t + m*N, (c_index+max(0,t-D[f]+1)):(c_index+(t+1))] =1
                F [t,(c_index+max(0,t-D[f]+1)):(c_index+(t+1))  ] = P[f]
    G2 = torch.eye((nbTasks*nbMachines*N))
    G3 = -1*torch.eye((nbTasks*nbMachines*N))
    h2 = torch.ones(nbTasks*nbMachines*N)
    h3 = torch.zeros(nbTasks*nbMachines*N)

    G = torch.cat((G1,G2,G3)) 
    h = torch.cat((h1,h2,h3))

    ### A and b
    A1 = torch.zeros((nbTasks, nbTasks*nbMachines*N))
    A2 = torch.zeros((nbTasks, nbTasks*nbMachines*N))
    A3 = torch.zeros((nbTasks, nbTasks*nbMachines*N))

    for f in Tasks:
        A1 [f,(f*N*nbMachines):((f+1)*N*nbMachines) ] = 1
        A2 [f,(f*N*nbMachines):(f*N*nbMachines + E[f]) ] = 1
        A3 [f,(f*N*nbMachines+L[f]-D[f]+1):((f+1)*N*nbMachines) ] = 1
    b = torch.cat((torch.ones(nbTasks),torch.zeros(2*nbTasks)))
    A = torch.cat((A1,A2,A3))    
    return A,b,G,h,torch.transpose(F, 0, 1)

def make_matrix_intopt(nbMachines,nbTasks,nbResources,MC,U,D,E,L,P,idle,up,down,q,**h):
    # nbMachines: number of machine
    # nbTasks: number of task
    # nb resources: number of resources
    # MC[m][r] resource capacity of machine m for resource r 
    # U[f][r] resource use of task f for resource r
    # D[f] duration of tasks f
    # E[f] earliest start of task f
    # L[f] latest end of task f
    # P[f] power use of tasks f
    # idle[m] idle cost of server m
    # up[m] startup cost of server m
    # down[m] shut-down cost of server m
    # q time resolution
    # timelimit in seconds

    # print("number of machines %d, number of tasks %d, number of resources %d"%(nbMachines,nbTasks,nbResources))
    Machines = range(nbMachines)
    Tasks = range(nbTasks)
    Resources = range(nbResources)
    N = 1440//q

    ### G and h
    G = torch.zeros((nbMachines*N ,nbTasks*nbMachines*N))
    h = torch.zeros(nbMachines*N)
    F = torch.zeros((N,nbTasks*nbMachines*N))
    for m in Machines:
        for t in range(N):
            h[m*N+t] = MC[m][0]
            for f in Tasks:
                c_index = (f*nbMachines+m)*N 
                G[t + m*N, (c_index+max(0,t-D[f]+1)):(c_index+(t+1))] =1
                F [t,(c_index+max(0,t-D[f]+1)):(c_index+(t+1))  ] = P[f]
    ### A and b
    A1 = torch.zeros((nbTasks, nbTasks*nbMachines*N))
    A2 = torch.zeros((nbTasks, nbTasks*nbMachines*N))
    A3 = torch.zeros((nbTasks, nbTasks*nbMachines*N))

    for f in Tasks:
        A1 [f,(f*N*nbMachines):((f+1)*N*nbMachines) ] = 1
        A2 [f,(f*N*nbMachines):(f*N*nbMachines + E[f]) ] = 1
        A3 [f,(f*N*nbMachines+L[f]-D[f]+1):((f+1)*N*nbMachines) ] = 1
    b = torch.cat((torch.ones(nbTasks),torch.zeros(2*nbTasks)))
    A = torch.cat((A1,A2,A3))    
    return A,b,G,h,torch.transpose(F, 0, 1)

#self.sol_validation = ICON_solution(param = param,y = y_validation, relax = self.validation_relax,n_items = self.n_items)
def ICON_solution(param,y,relax,n_items):
    clf = Gurobi_ICON(relax=relax,method=-1,reset=True,presolve=True,**param)
    clf.make_model()

    n_knap = len(y)//n_items
    sol_result = {}
    for kn_nr in range(n_knap):
        kn_start = kn_nr*n_items
        kn_stop = kn_start+n_items

        # [48]
        V = y[kn_start:kn_stop]
        assert len(V.shape) == 1 and V.shape[0] == 48
        
        sol, _ = clf.solve_model(V)
        assert len(sol.shape) == 1 and sol.shape[0] == 48

        # print(sol.shape)
    
        sol_result[kn_nr] = sol

    return sol_result
def ICON_obj(param,y,relax,n_items):
    clf = Gurobi_ICON(relax=relax,method=-1,reset=True,presolve=True,**param)
    clf.make_model()
    

    n_knap = len(y)//n_items
    sol_list = []
    for kn_nr in range(n_knap):
        kn_start = kn_nr*n_items
        kn_stop = kn_start+n_items
        V = y[kn_start:kn_stop]
        
        sol,_ = clf.solve_model(V)

        sol_list.append( sum(V*(sol)))
    return np.median(sol_list)

def validation_module(param,n_items,epoch=None, batch=None, 
    model_time = None,run_time=None,
    y_target_validation=None,sol_target_validation=None, y_pred_validation=None,
    y_target_test=None,sol_target_test=None,y_pred_test=None,
    relax=False,**kwargs):

    clf = Gurobi_ICON(relax=relax,method=-1,reset=True,presolve=True,**param)
    clf.make_model()
    def regret(y_target,sol_target,y_pred,relax= relax,**kwargs):
        n_knap = len(y_pred)//n_items
        regret_list= []

        for kn_nr in range(n_knap):
            kn_start = kn_nr*n_items
            kn_stop = kn_start+n_items
            V = y_pred[kn_start:kn_stop]
            V_target = y_target[kn_start:kn_stop]
            # print('V', V.shape)
            sol,_ = clf.solve_model(V)
            regret_list.append(sum(V_target*(sol-sol_target[kn_nr])))
        return np.median(regret_list), mse(y_target,y_pred)

    dict_validation = {}
    
    if (y_pred_validation is not None) and (y_target_validation is not None) and (sol_target_validation is not None):
        #print("validation",y_pred_validation.shape,y_target_validation.shape)
        validation_result = regret(y_target_validation,sol_target_validation,
         y_pred_validation,relax = relax)
        dict_validation['validation_regret'] = validation_result[0]
        dict_validation['validation_mse'] = validation_result[1]
    if (y_pred_test is not None) and (y_target_test is not None) and (sol_target_test is not None):
        #print("test ",y_pred_test.shape,y_target_test.shape)
        test_result = regret(y_target_test,sol_target_test,y_pred_test,
            relax = False)
        dict_validation['test_regret'] = test_result[0]
        dict_validation['test_mse'] = test_result[1]
        
    if batch is not None:
        dict_validation['batch'] = batch
    if epoch is not None:
        dict_validation['epoch'] = epoch
    if model_time is not None:
        dict_validation['Modeltime'] = model_time
    if run_time is not None:
        dict_validation['time'] = run_time
    return dict_validation

class intopt_energy:
    def __init__(self,param,
        input_size,hidden_size,num_layers,target_size=1,
        doScale= True,n_items=48,epochs=1,batchsize= 24,
        verbose=False,validation_relax=True,
        optimizer=optim.Adam,model_save=False,model_name=None,
        problem_timelimit= 50,model=None,store_validation=False,
        method =1,mu0=None,smoothing=False,thr = None,max_iter=None,
        damping=1e-3,clip=0.1,warmstart= False,
        **hyperparams):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.param = param
        self.doScale = doScale
        self.n_items = n_items
        self.epochs = epochs
        self.batchsize = batchsize

        self.verbose = verbose
        self.validation_relax = validation_relax
        #self.test_relax = test_relax        
        self.optimizer = optimizer
        self.model_save = model_save
        self.model_name = model_name
        self.smoothing = smoothing
        self.thr = thr
        self.damping = damping
        self.hyperparams = hyperparams
        self.max_iter = max_iter
        self.warmstart = warmstart
        self.method = method
        self.mu0 = mu0
        self.clip = clip
        self.problem_timelimit = problem_timelimit
        self.model = model
        self.store_validation = store_validation

        self.model = MultilayerRegression(input_size= input_size, hidden_size=hidden_size,target_size=target_size,num_layers=num_layers)

        print('Model state:')
        for param_tensor in self.model.state_dict():
            print(f'\t{param_tensor}\t{self.model.state_dict()[param_tensor].size()}')

        self.optimizer = optimizer(self.model.parameters(), **hyperparams)

    def fit(self,X,y,X_validation=None,y_validation=None,X_test=None,y_test=None):
        self.model_time = 0.
        runtime = 0.
        
        validation_time = 0
        test_time = 0
        # if validation true validation and tets data should be provided
        
        validation = (X_validation is not None) and (y_validation is not None)
        test = (X_test is not None) and (y_test is not None)
        param = self.param
        if self.doScale:
            self.scaler = preprocessing.StandardScaler().fit(X)
            X = self.scaler.transform(X)
        if validation:
            start_validation = time.time()
            
            if self.doScale:
                X_validation = self.scaler.transform(X_validation)
            end_validation = time.time()
            validation_time += end_validation - start_validation

        if test:
            start_test = time.time()
            
            if self.doScale:
                X_test = self.scaler.transform(X_test)
            end_test = time.time()
            test_time+=  end_test - start_test


        validation_relax = self.validation_relax
        n_items = self.n_items
        epochs = self.epochs
        batchsize = self.batchsize
        n_batches = X.shape[0]//(batchsize*n_items)
        n_knapsacks = X.shape[0]//n_items   
        subepoch= 0
       
        validation_result =[]
        shuffled_batches = [i for i in range(n_batches)]
        
        max_iter = self.max_iter
        # init_params = {el:None for el in range(n_knapsacks)}

        A, b, G, h, F = make_matrix_intopt(**param)
        logging.info("Started Intopt Optimization with method {} threshold {}".format(self.method,self.thr)) 
       
        for e in range(epochs):
            np.random.shuffle(shuffled_batches)
            for i in range(n_batches):
                start = time.time()
                self.optimizer.zero_grad()
                batch_list = random.sample([j for j in range(batchsize)], batchsize)

                for j in batch_list:
                    n_start = (batchsize*shuffled_batches[i] + j)*n_items
                    n_stop = n_start + n_items

                    z = torch.tensor(y[n_start:n_stop], dtype=torch.float)
                    X_tensor = torch.tensor(X[n_start:n_stop, :], dtype=torch.float)

                    print('Input example', 'X[start:stop]', X_tensor.shape, 'y[start:stop]', y[n_start:n_stop].shape)

                    c_true = torch.mm(F, torch.tensor(y[n_start:n_stop], dtype=torch.float).unsqueeze(1)).squeeze()

                    print('BBB', 'c_true', c_true.shape, 'F', F.shape, 'y[start:stop]', y[n_start:n_stop].shape)

                    model_out = self.model(X_tensor)
                    c_pred = torch.mm(F, model_out).squeeze()

                    print('BBB', 'c_pred', c_pred.shape, 'model_out', model_out.shape)

                    logging.info("c shape {}".format(c_pred.shape))

                    try:
                        with time_limit(self.problem_timelimit):
                            x = IPOfunc(A,b,G,h,pc = True,max_iter=self.max_iter, 
                                #init_val= init_params[(batchsize*shuffled_batches[i] + j)],
                                smoothing=self.smoothing,thr=self.thr,method= self.method,
                                mu0 = self.mu0,damping= self.damping)(c_pred)

                            print('CCC', 'x', x.shape, 'c_true', c_true.shape)

                            loss = (x*c_true).mean()

                            print('x', x.shape, 'c_true', c_true.shape, 'loss', loss.shape,
                                  'model_out', model_out.shape, 'F', F.shape)

                            c_pred.retain_grad()     
                            loss.backward()
                            # torch.nn.utils.clip_grad_norm_(self.lstm_layer.parameters(), 
                            #     self.clip)                                
                        forward_solved = IPOfunc.forward_solved()
                        self.model_time += IPOfunc.Runtime()
                        # print("solving cplt",datetime.datetime.now())
                        # print("solved",sum(x),x.shape)
                    except TimeoutException as msg:
                        forward_solved = False
                        logging.info("timelimitlimit exceeded")
                        print("Epoch[{}::{}] timelimitlimit exceeded\
                        If you see if often consider increasing \
                            problem_timelimit".format(e+1,i+1 ))
                    except LinAlgError as msg:
                        raise
                    except Exception as msg:
                        forward_solved = False
                        logging.info(msg)

                    if forward_solved:
                        logging.info("backward done {} {} {}".format(e,i,j))
                    else:
                        print("Epoch[{}/{}] fwd pass not solved".format(e+1,i+1 ))
                    

                self.optimizer.step()
                end = time.time()
                runtime += end -start
                logging.info("step done {} {}".format(e,i))
                # logging.info("--Model parameters--")
                # for  modelparam in self.lstm_layer.parameters():
                #     logging.info(modelparam)
                # logging.info("--******--")                
                if forward_solved:
                    logging.info("fwd not solved")
                    # if any(torch.isnan(c_pred.grad).tolist()):
                    #     logging.info("nan in c-gradient")
                    #     logging.info("smoothing is %s"%self.smoothing)

                
                subepoch += 1
                print('Epoch[{}/{}], loss(train):{:.2f} @ {:%Y-%m-%d %H:%M:%S} '.format(e+1, 
                            i+1, loss.item(),datetime.datetime.now() ))
                if ((i+1)%7==0)|((i+1)%n_batches==0):
                    if self.model_save:
                        torch.save(self.model.state_dict(), 
                            str(self.model_name+"_Epoch"+str(e)+"_"+str(i)+".pth"))

                    if self.store_validation:
                        if validation:
                            y_pred_validation= self.predict(X_validation,doScale= False)
                            if not hasattr(self, 'sol_validation'):
                                self.sol_validation = ICON_solution(param = param,y = y_validation,
                                    relax = self.validation_relax,n_items = self.n_items)
                        else:
                            self.sol_validation = None
                            y_pred_validation= None
                        if test:
                            y_pred_test= self.predict(X_test,doScale =False)
                            if not hasattr(self, 'sol_test'):
                                self.sol_test = ICON_solution(param = param,y =y_test,
                                    relax = False,n_items = self.n_items)
                        else:
                            self.sol_test = None
                            y_pred_test= None

                        dict_validation = validation_module(param = param,n_items= self.n_items,
                            run_time= runtime,epoch= e, batch=i, 
                            model_time = self.model_time,
                    y_target_validation= y_validation,
                    sol_target_validation= self.sol_validation,
                    y_pred_validation= y_pred_validation,
                    y_target_test= y_test,sol_target_test= self.sol_test ,
                    y_pred_test= y_pred_test,
                    relax= self.validation_relax)
                        validation_result.append(dict_validation)
        if self.store_validation:
            #return test_result
            dd = defaultdict(list)
            for d in validation_result:
                for key, value in d.items():
                    dd[key].append(value)
            df = pd.DataFrame.from_dict(dd)
            #self.logger.info('Completion Time %s \n' %str(datetime.datetime.now()) )
            logging.info('Completion Time %s \n' %str(datetime.datetime.now()) )

            return df

    def validation_result(self,X_validation,y_validation, scaler=None,doScale = True):
        if doScale:
            if scaler is None:
                try: 
                    scaler = self.scaler
                except:
                    raise Exception("you asked to do scaler but no StandardScaler found" )
            X_validation = scaler.transform(X_validation)
        model = self.model
        model.eval()
        X_tensor= torch.tensor(X_validation,dtype=torch.float)
        y_pred = model(X_tensor).detach().numpy().squeeze()
        model.train()
        sol_validation = ICON_solution(param = self.param,y = y_validation,
                                relax = False,n_items = self.n_items)
        validation_rslt = validation_module(param = self.param,n_items= self.n_items,
                    y_target_test= y_validation,sol_target_test= sol_validation ,
                    y_pred_test= y_pred)
        return validation_rslt['test_regret'] , validation_rslt['test_mse']        

    def predict(self,X,scaler=None,doScale = True):
        
        if doScale:
            if scaler is None:
                try: 
                    scaler = self.scaler
                except:
                    raise Exception("you asked to do scaler but no StandardScaler found" )
            X1 = scaler.transform(X)
        else:
            X1 = X
        model = self.model
        model.eval()
        X_tensor= torch.tensor(X1,dtype=torch.float)
        y_pred = model(X_tensor).detach().numpy().squeeze()
        model.train()
        return y_pred 
                             

                             

class qptl_energy:
    def __init__(self,param,
        input_size,hidden_size,num_layers,target_size=1,
        tau=20000,doScale= True,n_items=48,epochs=1,batchsize= 24,
        verbose=False,validation_relax=True,
        optimizer=optim.Adam,model_save=False,model_name=None,
        model=None,store_validation=False,problem_timelimit=500, **hyperparams):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.param = param
        self.tau = tau
        self.doScale = doScale
        self.n_items = n_items
        self.epochs = epochs
        self.batchsize = batchsize

        self.verbose = verbose
        self.validation_relax = validation_relax
        #self.test_relax = test_relax        
        self.optimizer = optimizer
        self.model_save = model_save
        self.model_name = model_name
        self.hyperparams = hyperparams
        self.problem_timelimit = problem_timelimit
        self.store_validation = store_validation

        self.model = MultilayerRegression(input_size= input_size, 
            hidden_size=hidden_size,target_size=target_size,num_layers=num_layers)
        self.optimizer = optimizer(self.model.parameters(), **hyperparams)

    def fit(self,X,y,X_validation=None,y_validation=None,X_test=None,y_test=None):

        self.model_time = 0.
        runtime = 0.
        validation_time = 0
        test_time = 0
        # if validation true validation and tets data should be provided
        
        validation = (X_validation is not None) and (y_validation is not None)
        test = (X_test is not None) and (y_test is not None)
        param = self.param
        if self.doScale:
            self.scaler = preprocessing.StandardScaler().fit(X)
            X = self.scaler.transform(X)
        if validation:
            start_validation = time.time()
            
            if self.doScale:
                X_validation = self.scaler.transform(X_validation)
            end_validation = time.time()
            validation_time += end_validation -start_validation

        if test:
            start_test = time.time()
            
            if self.doScale:
                X_test = self.scaler.transform(X_test)
            end_test = time.time()
            test_time+=  end_test - start_test

        validation_relax = self.validation_relax
        n_items = self.n_items
        epochs = self.epochs
        batchsize = self.batchsize
        n_batches = X.shape[0]//(batchsize*n_items)
        n_knapsacks = X.shape[0]//n_items   
        subepoch= 0
       
        validation_result =[]
        shuffled_batches = [i for i in range(n_batches)]

        A,b,G,h,F = make_matrix_qp(**param)
        Q = torch.eye(F.shape[0])/self.tau
        model_params_quad = make_gurobi_model(G.detach().numpy(),h.detach().numpy(),A.detach().numpy(), 
                                              b.detach().numpy(), Q.detach().numpy())
        self.gurobi_model = model_params_quad                
        for e in range(epochs):
            np.random.shuffle(shuffled_batches)
            for i in range(n_batches):
                start = time.time()
                self.optimizer.zero_grad()
                batch_list = random.sample([j for j in range(batchsize)], batchsize)
                for j in batch_list:
                    n_start =  (batchsize*shuffled_batches[i] + j)*n_items
                    n_stop = n_start + n_items
                    z = torch.tensor(y[n_start:n_stop],dtype=torch.float ) 
                    X_tensor= torch.tensor(X[n_start:n_stop,:],dtype=torch.float)
                    c_true= torch.mm(F,
                        torch.tensor(y[n_start:n_stop],dtype=torch.float ).unsqueeze(1)).squeeze()
                    
                    c_pred  = torch.mm(F,self.model(X_tensor)).squeeze()

              
                    try:
                        with time_limit(self.problem_timelimit):
                            solver = QPFunction(verbose=False, 
                                solver=QPSolvers.GUROBI, model_params=model_params_quad)
                            x = solver(Q.expand(1, *Q.shape),
                                c_pred.squeeze(), G.expand(1, *G.shape), 
                                h.expand(1, *h.shape), 
                                A.expand(1, *A.shape),b.expand(1, *b.shape))
                        forward_solved =True
                        # print("solving cplt",datetime.datetime.now())
                        # print("solved",sum(x),x.shape)
                    except TimeoutException as msg:
                        forward_solved = False
                        logging.info("timelimitlimit exceeded")
                        print("Epoch[{}::{}] timelimitlimit exceeded\
                        If you see if often consider increasing \
                            problem_timelimit".format(e+1,i+1 ))
                    except Exception as msg:
                        forward_solved = False
                        logging.info(msg)
                    
                    if forward_solved:
                        
                        loss = (x.squeeze()*c_true.squeeze()).mean()    
                        loss.backward()
                        # print("backward done")
                    else:
                        print("Epoch[{}/{}] fwd pass not solved".format(e+1,i+1 ))
                    

                self.optimizer.step()
                end = time.time()
                runtime += end -start                
                subepoch += 1
                print('Epoch[{}/{}], loss(train):{:.2f} @ {:%Y-%m-%d %H:%M:%S} '.format(e+1, 
                            i+1, loss.item(),datetime.datetime.now() ))
                if ((i+1)%7==0)|((i+1)%n_batches==0):
                    
                    if self.model_save:
                        torch.save(self.model.state_dict(), 
                            str(self.model_name+"_Epoch"+str(e)+"_"+str(i)+".pth"))

                    if self.store_validation:
                        if validation:
                            y_pred_validation= self.predict(X_validation,doScale= False)
                            if not hasattr(self, 'sol_validation'):
                                self.sol_validation = ICON_solution(param = param,y = y_validation,
                                    relax = self.validation_relax,n_items = self.n_items)
                        else:
                            self.sol_validation = None
                            y_pred_validation= None
                        if test:
                            y_pred_test= self.predict(X_test,doScale =False)
                            if not hasattr(self, 'sol_test'):
                                self.sol_test = ICON_solution(param = param,y =y_test,
                                    relax = False,n_items = self.n_items)
                        else:
                            self.sol_test = None
                            y_pred_test= None

                        dict_validation = validation_module(param = param,n_items= self.n_items,
                            run_time= runtime,epoch= e, batch=i, 
                            model_time = self.model_time,
                    y_target_validation= y_validation,
                    sol_target_validation= self.sol_validation, 
                    y_pred_validation= y_pred_validation,
                    y_target_test= y_test,sol_target_test= self.sol_test ,
                    y_pred_test= y_pred_test,
                    relax= self.validation_relax)
                        validation_result.append(dict_validation)
        if self.store_validation :
            #return test_result
            dd = defaultdict(list)
            for d in validation_result:
                for key, value in d.items():
                    dd[key].append(value)
            df = pd.DataFrame.from_dict(dd)
            #self.logger.info('Completion Time %s \n' %str(datetime.datetime.now()) )
            logging.info('Completion Time %s \n' %str(datetime.datetime.now()) )

            return df

    def validation_result(self,X_validation,y_validation, scaler=None,doScale = True):
        if doScale:
            if scaler is None:
                try: 
                    scaler = self.scaler
                except:
                    raise Exception("you asked to do scaler but no StandardScaler found" )
            X_validation = scaler.transform(X_validation)
        model = self.model
        model.eval()
        X_tensor= torch.tensor(X_validation,dtype=torch.float)
        y_pred = model(X_tensor).detach().numpy().squeeze()
        model.train()
        sol_validation = ICON_solution(param = self.param,y = y_validation,
                                relax = False,n_items = self.n_items)

        

        validation_rslt = validation_module(param = self.param,n_items= self.n_items,
                    y_target_test= y_validation,sol_target_test= sol_validation ,
                    y_pred_test= y_pred)
        return validation_rslt['test_regret'] , validation_rslt['test_mse']        

    def predict(self,X,scaler=None,doScale = True):
        
        if doScale:
            if scaler is None:
                try: 
                    scaler = self.scaler
                except:
                    raise Exception("you asked to do scaler but no StandardScaler found" )
            X1 = scaler.transform(X)
        else:
            X1 = X
        model = self.model
        model.eval()
        X_tensor= torch.tensor(X1,dtype=torch.float)
        y_pred = model(X_tensor).detach().numpy().squeeze()
        model.train()
        return y_pred 

class twostage_energy:
    def __init__(self,param,
        input_size,hidden_size,num_layers,target_size=1,
        doScale= True,n_items=48,epochs=1,batchsize= 24,
        verbose=False,validation_relax=True,
        optimizer=optim.Adam,model_save=False,model_name=None,
        model=None,store_validation=False, scheduler=False,
        **hyperparams):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.param = param
        self.doScale = doScale
        self.n_items = n_items
        self.epochs = epochs
        self.batchsize = batchsize

        self.verbose = verbose
        self.validation_relax = validation_relax
        #self.test_relax = test_relax        
        self.optimizer = optimizer
        self.model_save = model_save
        self.model_name = model_name
        self.hyperparams = hyperparams
        self.store_validation = store_validation
        self.scheduler = scheduler

        self.model = MultilayerRegression(input_size=input_size,hidden_size=hidden_size, target_size=target_size, num_layers=num_layers)
        self.optimizer = optimizer(self.model.parameters(), **hyperparams)
    def fit(self,X,y,X_validation=None,y_validation=None,X_test=None,y_test=None):

        self.model_time = 0.
        runtime = 0.
        
        validation_time = 0
        test_time = 0
        # if validation true validation and tets data should be provided
        
        validation = (X_validation is not None) and (y_validation is not None)
        test = (X_test is not None) and (y_test is not None)
        param = self.param
        if self.doScale:
            self.scaler = preprocessing.StandardScaler().fit(X)
            X = self.scaler.transform(X)
        if validation:
            start_validation = time.time()
            
            if self.doScale:
                X_validation = self.scaler.transform(X_validation)
            end_validation = time.time()
            validation_time += end_validation -start_validation

        if test:
            start_test = time.time()
            
            if self.doScale:
                X_test = self.scaler.transform(X_test)
            end_test = time.time()
            test_time+=  end_test - start_test
        validation_relax = self.validation_relax

        criterion = nn.MSELoss(reduction='mean')
        n_items = self.n_items
        epochs = self.epochs
       
        batchsize = self.batchsize
        n_batches = X.shape[0]//(batchsize*n_items)
        n_knapsacks = X.shape[0]//n_items   

        subepoch= 0
        validation_result =[]
        shuffled_batches = [i for i in range(n_batches)]

        n_train = len(y)
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda x: 1 if x < 2 else 0.95 ** x)

        for e in range(epochs):
            logging.info('Epoch %d'%e )
            np.random.shuffle(shuffled_batches)
            for i in range(n_batches):
                start = time.time()

                n_start = (batchsize*shuffled_batches[i] * n_items)
                n_stop = (batchsize*(shuffled_batches[i] + 1) * n_items)

                # [1152, 8], 1152 = 24 * 48
                X_tensor = torch.from_numpy(X[n_start:n_stop,:]).float()

                # [1152, 1], 1152 = 24 * 48
                y_target = torch.from_numpy(y[n_start:n_stop][:, np.newaxis]).float()

                self.optimizer.zero_grad()

                # [1152, 1], 1152 = 24 * 48
                y_pred = self.model(X_tensor)

                # []
                loss = criterion(y_pred, y_target)

                # print('loss', loss.shape)

                loss.backward()
                self.optimizer.step()
                end = time.time()
                runtime += end -start
                subepoch += 1
                print('Epoch[{}/{}], loss(train):{:.2f} @ {:%Y-%m-%d %H:%M:%S} '.format(e+1, 
                            i+1, loss.item(),datetime.datetime.now() ))
                if ((i+1)%7==0)|((i+1)%n_batches==0):
                    if self.model_save:
                        torch.save(self.model.state_dict(), 
                            str(self.model_name+"_Epoch"+str(e)+"_"+str(i)+".pth"))

                    if self.store_validation:
                        if validation:
                            y_pred_validation= self.predict(X_validation,doScale= False)
                            if not hasattr(self, 'sol_validation'):
                                self.sol_validation = ICON_solution(param = param,y = y_validation,
                                    relax = self.validation_relax,n_items = self.n_items)
                        else:
                            self.sol_validation = None
                            y_pred_validation= None
                        if test:
                            y_pred_test= self.predict(X_test,doScale =False)
                            if not hasattr(self, 'sol_test'):
                                self.sol_test = ICON_solution(param = param,y =y_test,
                                    relax = False,n_items = self.n_items)
                        else:
                            self.sol_test = None
                            y_pred_test= None

                        dict_validation = validation_module(param = param,n_items= self.n_items,
                            run_time= runtime,epoch= e, batch=i, 
                            model_time = self.model_time,
                    y_target_validation= y_validation,
                    sol_target_validation= self.sol_validation, 
                    y_pred_validation= y_pred_validation,
                    y_target_test= y_test,sol_target_test= self.sol_test ,
                    y_pred_test= y_pred_test,
                    relax= self.validation_relax)

                        print(dict_validation)

                        validation_result.append(dict_validation)
        if self.store_validation :
            #return test_result
            dd = defaultdict(list)
            for d in validation_result:
                for key, value in d.items():
                    dd[key].append(value)
            df = pd.DataFrame.from_dict(dd)
            #self.logger.info('Completion Time %s \n' %str(datetime.datetime.now()) )
            logging.info('Completion Time %s \n' %str(datetime.datetime.now()) )

            return df
    def validation_result(self,X_validation,y_validation, scaler=None,doScale = True):
        if doScale:
            if scaler is None:
                try: 
                    scaler = self.scaler
                except:
                    raise Exception("you asked to do scaler but no StandardScaler found" )
            X_validation = scaler.transform(X_validation)
        model = self.model
        model.eval()
        X_tensor= torch.tensor(X_validation,dtype=torch.float)
        y_pred = model(X_tensor).detach().numpy().squeeze()
        model.train()
        sol_validation = ICON_solution(param = self.param,y = y_validation,
                                relax = False,n_items = self.n_items)

        

        validation_rslt = validation_module(param = self.param,n_items= self.n_items,
                    y_target_test= y_validation,sol_target_test= sol_validation ,
                    y_pred_test= y_pred)
        return validation_rslt['test_regret'] , validation_rslt['test_mse']        

    def predict(self,X,scaler=None,doScale = True):
        
        if doScale:
            if scaler is None:
                try: 
                    scaler = self.scaler
                except:
                    raise Exception("you asked to do scaler but no StandardScaler found" )
            X1 = scaler.transform(X)
        else:
            X1 = X
        model = self.model
        model.eval()
        X_tensor= torch.tensor(X1,dtype=torch.float)
        y_pred = model(X_tensor).detach().numpy().squeeze()
        model.train()
        return y_pred 
                             
  


class SPO_energy:
    def __init__(self,param,
       input_size,hidden_size,num_layers,target_size=1,
        doScale= True,n_items=48,epochs=1,batchsize= 24,
        verbose=False,validation_relax=True,
        optimizer=optim.Adam,model_save=False,model_name=None,
        model=None,store_validation=False, scheduler=False,
        **hyperparams):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.param = param
        self.doScale = doScale
        self.n_items = n_items
        self.epochs = epochs
        self.batchsize = batchsize

        self.verbose = verbose
        self.validation_relax = validation_relax
        #self.test_relax = test_relax        
        self.optimizer = optimizer
        self.model_save = model_save
        self.model_name = model_name
        self.hyperparams = hyperparams
        self.store_validation = store_validation
        self.scheduler = scheduler

        self.model = MultilayerRegression(input_size=input_size, hidden_size=hidden_size, target_size=target_size, num_layers=num_layers)

        print('Model state:')
        for param_tensor in self.model.state_dict():
            print(f'\t{param_tensor}\t{self.model.state_dict()[param_tensor].size()}')

        self.optimizer = optimizer(self.model.parameters(), **hyperparams)

    def fit(self,X,y,X_validation=None,y_validation=None,X_test=None,y_test=None):
        self.model_time = 0.
        runtime = 0.
        validation_time = 0
        test_time = 0
        
        validation = (X_validation is not None) and (y_validation is not None)
        test = (X_test is not None) and (y_test is not None)
        param = self.param
        if self.doScale:
            self.scaler = preprocessing.StandardScaler().fit(X)
            X = self.scaler.transform(X)
        if validation:
            start_validation = time.time()
            
            if self.doScale:
                X_validation = self.scaler.transform(X_validation)
            end_validation = time.time()
            validation_time += end_validation - start_validation

        if test:
            start_test = time.time()
            
            if self.doScale:
                X_test = self.scaler.transform(X_test)
            end_test = time.time()
            test_time+=  end_test - start_test

        validation_relax = self.validation_relax
        n_items = self.n_items
        epochs = self.epochs
        batchsize = self.batchsize
        n_batches = X.shape[0]//(batchsize*n_items)
        n_knapsacks = X.shape[0]//n_items   
        subepoch= 0
       
        validation_result =[]
        shuffled_batches = [i for i in range(n_batches)]
        clf = Gurobi_ICON(relax=True,method=-1,reset=True,presolve=True,**param)
        clf.make_model()

        self.sol_train = ICON_solution(param=param, y=y, relax=True, n_items=self.n_items)

        for e in range(epochs):
            np.random.shuffle(shuffled_batches)
            start = time.time()
            for i in range(n_batches):
                start = time.time()
                self.optimizer.zero_grad()
                batch_list = random.sample([j for j in range(batchsize)], batchsize)
                for j in batch_list:
                    n_start = (batchsize*shuffled_batches[i] + j) * n_items
                    n_stop = n_start + n_items

                    # [48]
                    x_actual = self.sol_train[(batchsize*shuffled_batches[i] + j)]

                    z = torch.tensor(y[n_start:n_stop],dtype=torch.float )

                    # [48, 8]
                    X_tensor = torch.tensor(X[n_start:n_stop, :], dtype=torch.float)

                    # [48]
                    c_true = y[n_start:n_stop]

                    # [48]
                    y_pred = self.model(X_tensor).squeeze()
                    c_pred = y_pred.detach().numpy()

                    # [48]
                    c_spo = (2*c_pred - c_true)

                    # [48]
                    x_spo, _ = clf.solve_model(c_spo)

                    # print('SOLUTION', 'x_spo', x_spo.shape, 'c_spo', c_spo.shape, 'c_pred', c_pred.shape,
                    #       'X_ex', X_tensor.shape, 'y_ex', y_pred.shape, 'x_actual', x_actual.shape, 'c_true', c_true.shape)

                    grad = torch.from_numpy(x_actual - x_spo).float()
                    y_pred.backward(gradient=grad)

                self.optimizer.step()
                end = time.time()
                runtime += end -start
                logging.info("step done {} {}".format(e,i))
                subepoch += 1
                print('Epoch[{}/{}] @ {:%Y-%m-%d %H:%M:%S} '.format(e+1, 
                            i+1,datetime.datetime.now() ))
                if ((i+1)%7==0)|((i+1)%n_batches==0):
                    
                    if self.model_save:
                        torch.save(self.model.state_dict(), 
                            str(self.model_name+"_Epoch"+str(e)+"_"+str(i)+".pth"))
                    if self.store_validation:
                        if validation:
                            y_pred_validation= self.predict(X_validation,doScale= False)
                            if not hasattr(self, 'sol_validation'):
                                self.sol_validation = ICON_solution(param = param,y = y_validation,
                                    relax = self.validation_relax,n_items = self.n_items)
                        else:
                            self.sol_validation = None
                            y_pred_validation= None
                        if test:
                            y_pred_test= self.predict(X_test,doScale =False)
                            if not hasattr(self, 'sol_test'):
                                self.sol_test = ICON_solution(param = param,y =y_test,
                                    relax = False,n_items = self.n_items)
                        else:
                            self.sol_test = None
                            y_pred_test= None

                        dict_validation = validation_module(param = param,n_items= self.n_items,
                            run_time= runtime,epoch= e, batch=i, 
                            model_time = self.model_time,
                    y_target_validation= y_validation,
                    sol_target_validation= self.sol_validation, 
                    y_pred_validation= y_pred_validation,
                    y_target_test= y_test,sol_target_test= self.sol_test ,
                    y_pred_test= y_pred_test,
                    relax= self.validation_relax)
                        validation_result.append(dict_validation)
        if self.store_validation :
            #return test_result
            dd = defaultdict(list)
            for d in validation_result:
                for key, value in d.items():
                    dd[key].append(value)
            df = pd.DataFrame.from_dict(dd)
            #self.logger.info('Completion Time %s \n' %str(datetime.datetime.now()) )
            logging.info('Completion Time %s \n' %str(datetime.datetime.now()) )

            return df

    def validation_result(self,X_validation,y_validation, scaler=None,doScale = True):
        if doScale:
            if scaler is None:
                try: 
                    scaler = self.scaler
                except:
                    raise Exception("you asked to do scaler but no StandardScaler found" )
            X_validation = scaler.transform(X_validation)
        model = self.model
        model.eval()
        X_tensor= torch.tensor(X_validation,dtype=torch.float)
        y_pred = model(X_tensor).detach().numpy().squeeze()
        model.train()
        sol_validation = ICON_solution(param = self.param,y = y_validation,
                                relax = False,n_items = self.n_items)

        

        validation_rslt = validation_module(param = self.param,n_items= self.n_items,
                    y_target_test= y_validation,sol_target_test= sol_validation ,
                    y_pred_test= y_pred)
        return validation_rslt['test_regret'] , validation_rslt['test_mse']        
    def predict(self,X,scaler=None,doScale = True):
        
        if doScale:
            if scaler is None:
                try: 
                    scaler = self.scaler
                except:
                    raise Exception("you asked to do scaler but no StandardScaler found" )
            X1 = scaler.transform(X)
        else:
            X1 = X
        model = self.model
        model.eval()
        X_tensor= torch.tensor(X1,dtype=torch.float)
        y_pred = model(X_tensor).detach().numpy().squeeze()
        model.train()
        return y_pred 
                             
