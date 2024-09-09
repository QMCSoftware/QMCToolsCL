from _problems import * 

import numpy as np
import pandas as pd
import argparse 

parser = argparse.ArgumentParser(prog='Global Size Experiment')
parser.add_argument('-q','--qrproblem',type=str,nargs='?',default="lattice",help='quasi-random problem type in %s'%list(map_run_problem.keys()))
parser.add_argument('-t','--tag',type=str,nargs='?',default="debug",help='experiment tag')
parser.add_argument('-m','--np2',type=int,nargs='?',default=1,help='problems from n = 2^0,...,2^m')
parser.add_argument('-k','--dp2',type=int,nargs='?',default=2,help='problems from d = 2^0,...,2^k')
parser.add_argument('-s','--skip',type=int,nargs='?',default=0,help='initial runs skipped for each problem, default 0')
parser.add_argument('-r','--runs',type=int,nargs='?',default=1,help='runs after skipped runs which count for each problem, default 1')
parser.add_argument('-p','--platform',type=int,nargs='?',default=0,help='platform index, default 0')
parser.add_argument('-d','--device',type=int,nargs='?',default=0,help='device index within platform, default 0')
parser.add_argument('-f','--force',type=bool,nargs='?',default=False,help='if True, overwrite previous tag results directory')
args = parser.parse_args()

experiment_dir = "gs_%s.%s"%(args.qrproblem,args.tag)
run_problem = map_run_problem[args.qrproblem]
n = 2**args.np2 
d = 2**args.dp2 
gs_n_pows2 = np.arange(0,args.np2+1) 
gs_d_pows2 = np.arange(0,args.dp2+1)
gs_ns = 2**gs_n_pows2
gs_ds = 2**gs_d_pows2

kwargs_c,kwargs_cl = setup_speed_tests(args.platform,args.device)
remake_dir(experiment_dir,force=args.force)

df_c_perf,df_c_process = np.zeros((args.np2+1,args.dp2+1),dtype=np.float64),np.zeros((args.np2+1,args.dp2+1),dtype=np.float64) 
df_cl_perf,df_cl_process = np.zeros((args.np2+1,args.dp2+1),dtype=np.float64),np.zeros((args.np2+1,args.dp2+1),dtype=np.float64)

print("logging gs_n_pows2 up to %d and gs_d_pows2 up to %d"%(args.np2,args.dp2))
for i in range(args.np2+1):
    print("n_pows2 = %d: d_pows2 = "%gs_n_pows2[i],end='',flush=True)
    gs_n = gs_ns[i]
    for j in range(args.dp2+1):
        print("%d, "%gs_d_pows2[j],end='',flush=True)
        gs_d = gs_ds[j]
        kwargs_cl["global_size"] = (1,gs_n,gs_d)
        for t in range(args.skip+args.runs):
            c_perf,c_process = run_problem(n=n,d=d,kwargs=kwargs_c)
            cl_perf,cl_process = run_problem(n=n,d=d,kwargs=kwargs_cl)
            if t>=args.skip:
                df_c_perf[i,j] += c_perf 
                df_c_process[i,j] += c_process
                df_cl_perf[i,j] += cl_perf 
                df_cl_process[i,j] += cl_process
    print()
print()
df_c_perf /= args.runs
df_c_process /= args.runs
df_cl_perf /= args.runs 
df_cl_process /= args.runs

pd.DataFrame(df_c_perf,index=gs_ns,columns=gs_ds).to_csv("%s/gs_%s.c_perf.csv"%(experiment_dir,args.qrproblem))
pd.DataFrame(df_c_process,index=gs_ns,columns=gs_ds).to_csv("%s/gs_%s.c_process.csv"%(experiment_dir,args.qrproblem))
pd.DataFrame(df_cl_perf,index=gs_ns,columns=gs_ds).to_csv("%s/gs_%s.cl_perf.csv"%(experiment_dir,args.qrproblem))
pd.DataFrame(df_cl_process,index=gs_ns,columns=gs_ds).to_csv("%s/gs_%s.cl_process.csv"%(experiment_dir,args.qrproblem))
