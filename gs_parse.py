from _problems import *

import numpy as np 
import pandas as pd 
from matplotlib import pyplot,cm,colors
import argparse 

parser = argparse.ArgumentParser(prog='Global Size Experiment')
parser.add_argument('-q','--qrproblem',type=str,nargs='?',default="lattice",help='quasi-random problem type in %s'%list(map_run_problem.keys()))
parser.add_argument('-t','--tag',type=str,nargs='?',default="debug",help='experiment tag')
args = parser.parse_args()

experiment_dir = "gs_%s.%s"%(args.qrproblem,args.tag)
df_c_perf = pd.read_csv("%s/gs_%s.c_perf.csv"%(experiment_dir,args.qrproblem),index_col=0)
df_c_process = pd.read_csv("%s/gs_%s.c_process.csv"%(experiment_dir,args.qrproblem),index_col=0)
df_cl_perf = pd.read_csv("%s/gs_%s.cl_perf.csv"%(experiment_dir,args.qrproblem),index_col=0)
df_cl_process = pd.read_csv("%s/gs_%s.cl_process.csv"%(experiment_dir,args.qrproblem),index_col=0)

perf_plt = df_c_perf.values/df_cl_perf.values
process_plt = df_c_process.values/df_cl_process.values
vmax = max(perf_plt.max(),process_plt.max())

cmap = cm.gist_yarg
textcol_leq1 = "k"
textcol_gt1 = "r" 
rows,cols = perf_plt.shape 
fig,ax = pyplot.subplots(nrows=1,ncols=2,figsize=(2*cols,rows+1),sharey=True)
norm = colors.Normalize(vmin=0,vmax=vmax)
ax[0].set_title("wall clock time")
ax[0].imshow(perf_plt,cmap=cmap,norm=norm)
ax[1].set_title("process time")
ax[1].imshow(process_plt,cmap=cmap,norm=norm)
ax[0].set_ylabel(r"global size $n$")
ax[0].set_yticks(np.arange(rows),labels=[r"$2^{%d}$"%p for p in np.log2(df_c_perf.index)])
for i in range(2):
    ax[i].set_xlabel(r"global size $d$")
    ax[i].set_xticks(np.arange(cols),labels=[r"$2^{%d}$"%np.log2(int(v)) for v in df_c_perf.columns])
    #pyplot.setp(ax[i].get_xticklabels(),rotation=45,ha="right",rotation_mode="anchor")
for i in range(rows):
    for j in range(cols):
        text = ax[0].text(j,i,"%.2f"%perf_plt[i,j],ha="center",va="center",color=textcol_leq1 if perf_plt[i,j]<=1 else textcol_gt1)
        text = ax[1].text(j,i,"%.1f"%process_plt[i,j],ha="center",va="center",color=textcol_leq1 if process_plt[i,j]<=1 else textcol_gt1)
cax = fig.add_axes([ax[0].get_position().x0,ax[0].get_position().y1+.1,ax[1].get_position().x1-ax[0].get_position().x0,0.05])
fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),cax,location="top",label="speedup factor")
fig.savefig("%s/gs_%s.png"%(experiment_dir,args.qrproblem),dpi=256,bbox_inches="tight")
