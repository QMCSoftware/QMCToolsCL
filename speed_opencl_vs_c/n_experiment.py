from problems import * 

import numpy as np 
from matplotlib import pyplot
import os 

pyplot.style.use('seaborn-v0_8-whitegrid')

problem_name = "halton"
tag = "save"
d = 1000
min_pow2 = 0
max_pow2 = 10
plot_min_pow2 = 6

run_problem = map_run_problem[problem_name]
pows2 = np.arange(min_pow2,max_pow2+1)
ns = 2**pows2

experiment_dir = "%s/n_%s.%s"%(THISDIR,problem_name,tag)
assert not os.path.isdir(experiment_dir), "existing experiment_dir = %s"%experiment_dir
os.mkdir(experiment_dir) 

df = {}
df["n"] = ns
keys = [
    "c_perf",
    "cl_perf_n_d",
    "cl_perf_n2_d",
    "cl_perf_n_d2",
    "cl_perf_n2_d2",
    "c_process",
    "cl_process_n_d",
    "cl_process_n2_d",
    "cl_process_n_d2",
    "cl_process_n2_d2",
]
for key in keys: df[key] = np.zeros(max_pow2-min_pow2+1,dtype=np.float64) 

print("logging pow2 from %d to %d: "%(min_pow2,max_pow2),end='',flush=True)
for i in range(max_pow2-min_pow2+1):
    pow2 = pows2[i]
    print("%d, "%pow2,end='',flush=True)
    n = ns[i]
    df["c_perf"][i],df["c_process"][i] = run_problem(n=n,d=d,kwargs=kwargs_c)
    kwargs_cl["global_size"] = (1,n,d)
    df["cl_perf_n_d"][i],df["cl_process_n_d"][i] = run_problem(n=n,d=d,kwargs=kwargs_cl)
    kwargs_cl["global_size"] = (1,max(1,n//2),d)
    df["cl_perf_n2_d"][i],df["cl_process_n2_d"][i] = run_problem(n=n,d=d,kwargs=kwargs_cl)
    kwargs_cl["global_size"] = (1,n,max(1,d//2))
    df["cl_perf_n_d2"][i],df["cl_process_n_d2"][i] = run_problem(n=n,d=d,kwargs=kwargs_cl)
    kwargs_cl["global_size"] = (1,max(1,n//2),max(1,d//2))
    df["cl_perf_n2_d2"][i],df["cl_process_n2_d2"][i] = run_problem(n=n,d=d,kwargs=kwargs_cl)
print("\n")

df = pd.DataFrame(df)
df.to_csv("%s/n_%s.csv"%(experiment_dir,problem_name),index=False)

df = pd.read_csv("%s/n_%s.csv"%(experiment_dir,problem_name))
df = df.iloc[(plot_min_pow2-min_pow2):]
fig,ax = pyplot.subplots(nrows=1,ncols=2,figsize=(12,5),sharey=True)
for i,timing in enumerate(["process","perf"]):
    ax[i].plot(df["n"],df["c_%s"%timing],marker="o",linestyle="solid",color="xkcd:green",label="C")
    ax[i].plot(df["n"],df["cl_%s_n_d"%timing],marker="o",linestyle="solid",color="xkcd:blue",label="OpenCL(n,d)")
    ax[i].plot(df["n"],df["cl_%s_n2_d"%timing],marker="o",linestyle="dashdot",color="xkcd:blue",label="OpenCL(n/2,d)")
    ax[i].plot(df["n"],df["cl_%s_n_d2"%timing],marker="o",linestyle="dashed",color="xkcd:blue",label="OpenCL(n,d/2)")
    ax[i].plot(df["n"],df["cl_%s_n2_d2"%timing],marker="o",linestyle="dotted",color="xkcd:blue",label="OpenCL(n/2,d/2)")
    ax[i].set_xscale('log',base=2)
    ax[i].set_yscale('log',base=10)
    ax[i].set_xlabel(r"$n$ points")
ax[0].legend(loc='lower center',bbox_to_anchor=(1.15,-0.25),frameon=False,shadow=True,ncol=5)
ax[0].set_title("process time")
ax[1].set_title("wall clock time")
ax[0].set_ylabel(r"time [sec]")
fig.suptitle(r"%s sequence with $d=%d$ dimensions"%(problem_name,d))
fig.savefig("%s/n_%s.png"%(experiment_dir,problem_name),dpi=256,bbox_inches="tight")

