# -*- coding: utf-8 -*-

import numpy as np

from gurobipy import *


def data_reading(filename):
    with open(filename) as f:
        mylist = f.read().splitlines()
    
    q= int(mylist[0])
    nbResources = int(mylist[1])
    nbMachines =int(mylist[2])
    idle = [None]*nbMachines
    up = [None]*nbMachines
    down = [None]*nbMachines
    MC = [None]*nbMachines
    for m in range(nbMachines):
        l = mylist[2*m+3].split()
        idle[m] = int(l[1])
        up[m] = float(l[2])
        down[m] = float(l[3])
        MC[m] = list(map(int, mylist[2*(m+2)].split()))
    lines_read = 2*nbMachines + 2
    nbTasks = int(mylist[lines_read+1])

    # print(nbTasks)

    U = [None]*nbTasks
    D=  [None]*nbTasks
    E=  [None]*nbTasks
    L=  [None]*nbTasks
    P=  [None]*nbTasks
    for f in range(nbTasks):
        l = mylist[2*f + lines_read+2].split()
        D[f] = int(l[1])
        E[f] = int(l[2])
        L[f] = int(l[3])
        P[f] = float(l[4])
        U[f] = list(map(int, mylist[2*f + lines_read+3].split()))

    # print(D)

    return {"nbMachines":nbMachines,
                "nbTasks":nbTasks,"nbResources":nbResources,
                "MC":MC,
                "U":U,
                "D":D,
                "E":E,
                "L":L,
                "P":P,
                "idle":idle,
                "up":up,
                "down":down,
                "q":q}
class Gurobi_ICON:
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
    def __init__(self,nbMachines,nbTasks,nbResources,MC,U,D,E,L,P,idle,up,down,q,reset,presolve, relax=False,
        verbose=False,warmstart=False,method=-1,**h):
        self.nbMachines  = nbMachines
        self.nbTasks = nbTasks
        self.nbResources = nbResources
        self.MC = MC
        self.U =  U
        self.D = D
        self.E = E
        self.L = L
        self.P = P
        self.idle = idle
        self.up = up
        self.down = down
        self.q= q
        self.relax = relax
        self.verbose = verbose
        
        self.method = method
        #self.obj_cut = obj_cut 
        self.sol_hist = []
        self.presolve = presolve
        self.reset = reset
        self.warmstart = warmstart
        #-1 : no objective cut; 0: cut for predictions only 'true' solution; 1: use sol_hist
        
    def make_model(self):
        Machines = range(self.nbMachines)
        Tasks = range(self.nbTasks)
        Resources = range(self.nbResources)

        MC = self.MC
        U =  self.U
        D = self.D
        E = self.E
        L = self.L
        P = self.P
        idle = self.idle
        up = self.up
        down = self.down
        relax = self.relax
        q= self.q
        N = 1440//q

        M = Model("icon")
        if not self.verbose:
            M.setParam('OutputFlag', 0)
        if relax:
            x = M.addVars(Tasks, Machines, range(N), lb=0., ub=1., vtype=GRB.CONTINUOUS, name="x")
        else:
            x = M.addVars(Tasks, Machines, range(N), vtype=GRB.BINARY, name="x")

        # x = {}
        # for f in Tasks:
        #     for m in Machines:
        #         for t in range(N):
        #             if relax:
        #                 x[(f,m,t)] = M.addVar(0.,1.,name= "x"+str(f)+"_"+str(m)+"_"+str(t))
        #             else:
        #                 x[(f,m,t)] = M.addVar(vtype=GRB.BINARY,name= "x"+str(f)+"_"+str(m)+"_"+str(t))
                    
        # earliest start time constraint
        #M.addConstrs( (x[(f,m,t)]==0 )  for f in Tasks for m in Machines for t in range(E[f]) )
        # latest end time constraint
        #M.addConstrs( (x[(f,m,t)]==0 )   for f in Tasks for m in Machines for t in range(L[f]-D[f] +1,N) )
        M.addConstrs( x.sum(f,'*',range(E[f])) == 0 for f in Tasks)
        M.addConstrs( x.sum(f,'*',range(L[f]-D[f]+1,N)) == 0 for f in Tasks)



        M.addConstrs(( quicksum(x[(f,m,t)] for t in range(N) for m in Machines) == 1  for f in Tasks))

        # capacity requirement
        for r in Resources:
            for m in Machines:
                for t in range(N):
                    M.addConstr( quicksum( quicksum(x[(f,m,t1)]  for t1 in range(max(0,t-D[f]+1),t+1) )*
                                   U[f][r] for f in Tasks) <= MC[m][r])   
        if self.presolve:
            M = M.presolve()
        else:
            M.update()
        self.model = M
        #self.x =x
        # recreate dictionary
        self.x = dict()
        for var in M.getVars():
            name = var.varName
            if name.startswith('x['):
                (f,m,t) = map(int, name[2:-1].split(','))
                self.x[(f,m,t)] = var
        #assert(len(x) == len(M.getVars()))

    def solve_model(self,price,timelimit=None):
        Model = self.model
        MC = self.MC
        U =  self.U
        D = self.D
        E = self.E
        L = self.L
        P = self.P
        idle = self.idle
        up = self.up
        down = self.down
        q= self.q
        N = 1440//q  
        newcut = None
        if self.reset:
            Model.reset()

        verbose = self.verbose
        x =  self.x
        nbMachines = self.nbMachines
        nbTasks = self.nbTasks
        nbResources = self.nbResources
        Machines = range(nbMachines)
        Tasks = range(nbTasks)
        Resources = range(nbResources)
        obj_expr = quicksum( [x[(f,m,t)]*np.sum(price[t:t+D[f]])*P[f]*q/60 
            for f in Tasks for t in range(N-D[f]+1) for m in Machines if (f,m,t) in x] )

        
        if self.warmstart:
            bestval = np.inf # MINIMISATION problem!
            if len(self.sol_hist)>0:
                for i,sol in enumerate(self.sol_hist):
                    (pvars,dcons,vbasis,cbasis,sol_vec) = sol
                    val = sum( (sum(sol_vec[f,m,t] for m in Machines)*np.sum(price[t:t+D[f]])*P[f]*q/60) 
                        for f in Tasks for t in range(N-D[f]+1))
                    if  val< bestval:
                        val =bestval
                        ind = i
                        
                (pvars,dcons,vbasis,cbasis,sol_vec) = self.sol_hist[ind]
                for i,var in enumerate(Model.getVars()):
                    var.Pstart = pvars[i]
                    var.VBasis = vbasis[i]
                for i,cons in enumerate(Model.getConstrs()):
                    cons.Dstart = dcons[i]
                    cons.CBasis = cbasis[i]
        #print("######################### Value before optimization ################")
        #print(x)



        # this use of 'warmstart' objcuts undoes the benefit of native warmstarts...
        # if False \
        #    and warmstart is not None and (self.obj_cut>=0):
        #     (pvars,dcons,sol) = warmstart
        #     bestval = sum( (sum(sol[f,m,t] for m in Machines)*np.sum(price[t:t+D[f]])*P[f]*q/60) 

        #             for f in Tasks for t in range(N-D[f]+1))

        #     # OLD: add Pstart/Dstart, not sure this works, make sure presolve still runs!
        #     '''for i,var in enumerate(Model.getVars()):
        #         var.Pstart = pvars[i]
        #     for i,cons in enumerate(Model.getConstrs()):
        #         cons.Dstart = dcons[i]'''
        # if (self.obj_cut>=1) and (len(self.sol_hist) > 0):
        #     for sol in self.sol_hist:
        #          #val = np.sum(sol*price)
        #          val =  sum( (sum(sol[f,m,t] for m in Machines)*np.sum(price[t:t+D[f]])*P[f]*q/60) 
        #             for f in Tasks for t in range(N-D[f]+1)) 

        #          if (bestval is None) or (val < bestval):
        #             bestval = val
        # if bestval is not  None:
        #     # Minimization; make sure to remove the cut again below!!
        #     newcut = Model.addConstr(obj_expr <= bestval, name="objcut")   
                                     
        
        Model.setObjective(obj_expr, GRB.MINIMIZE)
        #Model.setObjective( quicksum( (x[(f,m,t)]*P[f]*quicksum([price[t+i] for i in range(D[f])])*q/60) for f in Tasks
        #                for m in Machines for t in range(N-D[f]+1)), GRB.MINIMIZE)
        if timelimit:
            Model.setParam('TimeLimit', timelimit)
        #if relax:
        #    Model = Model.relax()
        Model.setParam('Method', self.method)
        #logging.info("Number of constraints%d",Model.NumConstrs)
        Model.optimize()
        #print("###################  Value after optimization #########################")
        #print(x)

        solver = np.zeros(N)
        if Model.status in [GRB.Status.OPTIMAL,9]:
            try:
                #task_on = Model.getAttr('x',x)
                task_on = np.zeros( (nbTasks,nbMachines,N) )
                for ((f,m,t),var) in x.items():
                    try:
                        task_on[f,m,t] = var.X
                    except AttributeError:
                        task_on[f,m,t] = 0.
                        print("AttributeError: b' Unable to retrieve attribute 'X'")
                        print("__________Something WRONG___________________________")


                if verbose:
                    # for k,val in task_on.items():
                    #     if int(val)>0:
                    #         print("Task_%d starts on machine_%d at %d"%(k[0],k[1],k[2]))
                    print('\nCost: %g' % Model.objVal)
                    print('\nExecution Time: %f' %Model.Runtime)

                # remove cut again (modifies model)
                # if newcut is not None:
                #     Model.remove(newcut)
                #     newcut = None
                
                for t in range(N):        
                    solver[t] = np.sum( np.sum(task_on[f,:,max(0,t-D[f]+1):t+1])*P[f] for f in Tasks )  
                solver = solver*q/60  
                # if self.obj_cut>=1:
                #     self.sol_hist.append(task_on)
                #     if len(self.sol_hist)>(self.obj_cut):
                #         self.sol_hist.pop(0)

                if self.warmstart:
                    pvars = Model.getAttr(GRB.Attr.X, Model.getVars())
                    dcons = Model.getAttr(GRB.Attr.Pi, Model.getConstrs())
                    vbasis  = Model.getAttr(GRB.Attr.VBasis, Model.getVars())
                    cbasis = Model.getAttr(GRB.Attr.CBasis,Model.getConstrs())
                    self.sol_hist.append( (pvars,dcons,vbasis,cbasis,task_on) )
                    if len(self.sol_hist)>10:
                        self.sol_hist.pop(0)

                    #return solver,(pvars,dcons,vbasis,cbasis,task_on),Model.Runtime
                    #schedule = np.zeros((nbTasks,nbMachines,N))
                    # for f in Tasks:
                    #     for m in Machines:
                    #         for t in range(N):
                    #             schedule[f,m,t] = task_on[(f,m,t)]
                    # return solver,schedule,Model.Runtime
                return solver,Model.Runtime
            except NameError:
                print("\n__________Something wrong_______ \n ")
                # make sure cut is removed! (modifies model)
                if newcut is not None:
                    Model.remove(newcut)
                    newcut = None
                return solver,Model.Runtime

        elif Model.status == GRB.Status.INF_OR_UNBD:
            print('Model is infeasible or unbounded')
        elif Model.status == GRB.Status.INFEASIBLE:
            print('Model is infeasible')
        elif Model.status == GRB.Status.UNBOUNDED:
            print('Model is unbounded')
        else:
            print('Optimization ended with status %d' % Model.status)

        # make sure cut is removed! (modifies model)
        # if newcut is not None:
        #     Model.remove(newcut)

        return solver,Model.Runtime   


def ICON_scheduling(price,nbMachines,nbTasks,nbResources,MC,U,D,E,L,P,idle,up,down,q,
    verbose=False,scheduling=False,warmstart=None, timelimit=None):
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

    Machines = range(nbMachines)
    Tasks = range(nbTasks)
    Resources = range(nbResources)
    N = 1440//q

    M = Model("icon")
    if not verbose:
        M.setParam('OutputFlag', 0)

    x = {}
    for f in Tasks:
        for m in Machines:
            for t in range(N):
                x[(f,m,t)] = M.addVar(vtype=GRB.BINARY,name= "x"+str(f)+"_"+str(m)+"_"+str(t))
                if warmstart is not None:
                    x[(f,m,t)].start = warmstart[f,m,t]



    # earliest start time constraint
    M.addConstrs( (x[(f,m,t)]==0 )  for f in Tasks for m in Machines for t in range(E[f]) )
    # latest end time constraint
    M.addConstrs( (x[(f,m,t)]==0 )   for f in Tasks for m in Machines for t in range(L[f]-D[f] +1,N) )
    M.addConstrs(( quicksum(x[(f,m,t)] for t in range(N) for m in Machines) == 1  for f in Tasks))
    # capacity requirement
    for r in Resources:
        for m in Machines:
            for t in range(N):
                M.addConstr( sum( sum(x[(f,m,t1)]  for t1 in range(max(0,t-D[f]+1),t+1) )*
                                   U[f][r] for f in Tasks) <= MC[m][r])                
    M.setObjective( sum( (x[(f,m,t)]*P[f]*sum([price[t+i] for i in range(D[f])])*q/60) for f in Tasks
                        for m in Machines for t in range(N-D[f]+1)), GRB.MINIMIZE)
    if timelimit:
        M.setParam('TimeLimit', timelimit)
    M.optimize()
    if M.status == GRB.Status.OPTIMAL:
        task_on = M.getAttr('x',x)
        if verbose:
            for k,val in task_on.items():
                if int(val)>0:
                    print("Task_%d starts on machine_%d at %d"%(k[0],k[1],k[2]))
            print('\nCost: %g' % M.objVal)
            print('\nExecution Time: %f' %M.Runtime)

        solver = np.zeros(N)
        '''
        for t in range(N+1):
        solver[t] = sum(task[(f,m,t)]*P[f] for f in Tasks for m in Machines)*q/60 + \
        sum( machine_run[(m,t)]*idle[m] for m in  Machines)*q/60 + \
        sum(m_on[(m,t)]*up[m] for m in Machines)/price[t] + \
        sum(m_off[(m,t)]*down[m] for m in Machines)/price[t]
        '''
        for t in range(N):        
            solver[t] = sum( sum(task_on[(f,m,t1)]  for t1 in range(max(0,t-D[f]+1),t+1) )*P[f] 
                            for f in Tasks for m in Machines  ) 
        solver = solver*q/60        
        
        if scheduling:
            schedule = np.zeros((nbTasks,nbMachines,N))
            for f in Tasks:
                for m in Machines:
                    for t in range(N):
                        schedule[f,m,t] = task_on[(f,m,t)]
            return solver,schedule
        return solver
    elif M.status == GRB.Status.INF_OR_UNBD:
        print('Model is infeasible or unbounded')

    elif M.status == GRB.Status.INFEASIBLE:
        print('Model is infeasible')
    elif M.status == GRB.Status.UNBOUNDED:
        print('Model is unbounded')
    elif M.status ==9:
        try:
            task_on = M.getAttr('x',x)
            solver = np.zeros(N)
            for t in range(N):
                solver[t] = sum( sum(task_on[(f,m,t1)]  for t1 in range(max(0,t-D[f]+1),t+1) )*P[f] 
                            for f in Tasks for m in Machines  ) 
            solver = solver*q/60
            if scheduling:
                schedule = np.zeros((nbTasks,nbMachines,N))
                for f in Tasks:
                    for m in Machines:
                        for t in range(N):
                            schedule[f,m,t] = task_on[(f,m,t)]
                return solver,schedule
            return solver
        except:
             print("__________Something went wrong_______") 
    else:
        print('Optimization ended with status %d' % M.status)   
    return None


def ICON_scheduling_relaxation(price,nbMachines,nbTasks,nbResources,MC,U,D,E,L,P,idle,up,down,q,
    verbose=False,scheduling=False,warmstart=None, timelimit=None):
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

    Machines = range(nbMachines)
    Tasks = range(nbTasks)
    Resources = range(nbResources)
    N = 1440//q

    M = Model("icon")
    if not verbose:
        M.setParam('OutputFlag', 0)
    lb =0.0
    ub =1.0  
    x = {}
    for f in Tasks:
        for m in Machines:
            for t in range(N):
                x[(f,m,t)] = M.addVar(lb,ub,name= "x"+str(f)+"_"+str(m)+"_"+str(t))
                if warmstart is not None:
                    x[(f,m,t)].start = warmstart[f,m,t]

    # earliest start time constraint
    M.addConstrs( (x[(f,m,t)]==0 )  for f in Tasks for m in Machines for t in range(E[f]) )
    # latest end time constraint
    M.addConstrs( (x[(f,m,t)]==0 )   for f in Tasks for m in Machines for t in range(L[f]-D[f] +1,N) )

   

    M.addConstrs(( quicksum(x[(f,m,t)] for t in range(N) for m in Machines) == 1  for f in Tasks))
    # capacity requirement
    for r in Resources:
        for m in Machines:
            for t in range(N):
                M.addConstr( sum( sum(x[(f,m,t1)]  for t1 in range(max(0,t-D[f]+1),t+1) )*
                                   U[f][r] for f in Tasks) <= MC[m][r])                
                

    M.setObjective( sum( (x[(f,m,t)]*P[f]*sum([price[t+i] for i in range(D[f])])*q/60) for f in Tasks
                        for m in Machines for t in range(N-D[f]+1)), GRB.MINIMIZE)     
    
    if timelimit:
        M.setParam('TimeLimit', timelimit)    
    M.optimize()
    schedule ={}
    if M.status == GRB.Status.OPTIMAL:
        task_on = M.getAttr('x',x)

        if verbose:
            for k,val in task_on.items():
                if int(val)>0:
                    print("Task_%d starts on machine_%d at %d"%(k[0],k[1],k[2]))
            print('\nCost: %g' % M.objVal)
        solver = np.zeros(N)
        '''
        for t in range(N+1):
        solver[t] = sum(task[(f,m,t)]*P[f] for f in Tasks for m in Machines)*q/60 + \
        sum( machine_run[(m,t)]*idle[m] for m in  Machines)*q/60 + \
        sum(m_on[(m,t)]*up[m] for m in Machines)/price[t] + \
        sum(m_off[(m,t)]*down[m] for m in Machines)/price[t]
        '''
        for t in range(N):        
            solver[t] = sum( sum(task_on[(f,m,t1)]  for t1 in range(max(0,t-D[f]+1),t+1) )*P[f] 
                            for f in Tasks for m in Machines  ) 
        solver = solver*q/60        
        
        if scheduling:
            solver = np.zeros((nbTasks,nbMachines,N))
            for f in Tasks:
                for m in Machines:
                    for t in range(N):
                        solver[f,m,t] = task_on [(f,m,t)]
            return solver,schedule
        return solver
    elif M.status ==9:
        try:
            task_on = M.getAttr('x',x)
            solver = np.zeros(N)
            for t in range(N):
                solver[t] = sum( sum(task_on[(f,m,t1)]  for t1 in range(max(0,t-D[f]+1),t+1) )*P[f] 
                            for f in Tasks for m in Machines  ) 
            solver = solver*q/60
            if scheduling:
                schedule = np.zeros((nbTasks,nbMachines,N))
                for f in Tasks:
                    for m in Machines:
                        for t in range(N):
                            schedule[f,m,t] = task_on[(f,m,t)]
                return solver,schedule
            return solver
        except:
             print("__________Something went wrong_______") 

    elif M.status == GRB.Status.INF_OR_UNBD:
        print('Model is infeasible or unbounded')

    elif M.status == GRB.Status.INFEASIBLE:
        print('Model is infeasible')
    elif M.status == GRB.Status.UNBOUNDED:
        print('Model is unbounded')

    else:
        print('Optimization ended with status %d' % M.status)

    return None

 
                

def optimal_value(price,schedule,nbMachines,nbTasks,nbResources,MC,U,D,E,L,P,idle,up,down,q):
    if schedule:
        m_on = schedule["y"]
        c_up = 0
        for k,val in m_on.items():
            if int(val)>0:
                c_up += up[k[0]]
        m_off = schedule["z"]

        c_down=0
        for k,val in m_off.items():
            if int(val)>0:
                c_down += down[k[0]]
        task = schedule["x1"]
        cJ =0
        for k,val in task.items():
            if int(val)>0:
                cJ += P[k[0]]* price[k[2]]
        cJ = (cJ*q)/60
        machine_run = schedule["v"]
        cM=0
        for k,val in machine_run.items():
            if int(val)>0:
                cM += idle[k[0]]*price[k[1]]
        cM = (cM*q)/60
        return c_up+c_down+cJ+cM
    else:
        print("No feasible Solution")
        return math.inf


def main():
    from maprop.energy.get_energy import get_energy
    (X_1gtrain, y_train, X_1gtest, y_test) = get_energy()
    dirct = 'load1'
    fileList = sorted(os.listdir(dirct))
    day_cnt=0
    for file in fileList:
        nbMachines,nbTasks,nbResources,MC,U,D,E,L,P,idle,up,down,q = data_reading(dirct+"/"+file)
        price = y_train[(day_cnt*48):(1+(day_cnt+1)*48)]
        sch = ICON_scheduling(nbMachines,nbTasks,nbResources,MC,U,D,E,L,P,idle,up,down,q,price,verbose=False)
        print(optimal_value(nbMachines,nbTasks,nbResources,MC,U,D,E,L,P,idle,up,down,q,price,sch))
        day_cnt+=1
    
    
        
        
        
  
  
if __name__== "__main__":
  main()



