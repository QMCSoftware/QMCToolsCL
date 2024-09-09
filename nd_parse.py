from _problems import *

import numpy as np 
import pandas as pd 
from matplotlib import pyplot,cm,colors
import argparse 

parser = argparse.ArgumentParser(prog='Global Size Experiment')
parser.add_argument('-q','--qrproblem',type=str,nargs='?',default="lattice",help='quasi-random problem type in %s'%list(map_run_problem.keys()))
parser.add_argument('-t','--tag',type=str,nargs='?',default="debug",help='experiment tag')
parser.add_argument('-m','--np2min',type=int,nargs='?',default=0,help='only plot n>=2^m')
parser.add_argument('-k','--dp2min',type=int,nargs='?',default=0,help='only plot d>=2^k')
parser.add_argument('-c','--cmap',type=str,nargs='?',default="gist_yarg",help='colormap from matplotlib.cm')
parser.add_argument('-x','--cleq1',type=str,nargs='?',default="k",help='text color when speedup<=1')
parser.add_argument('-y','--cgt1',type=str,nargs='?',default="b",help='text color when speedup>1')
parser.add_argument('-z','--cperf',type=str,nargs='?',default="r",help='text color for perf plot')
args = parser.parse_args()

experiment_dir = "nd_%s.%s"%(args.qrproblem,args.tag)
df_c_perf = pd.read_csv("%s/nd_%s.c_perf.csv"%(experiment_dir,args.qrproblem),index_col=0)
df_cl_perf = pd.read_csv("%s/nd_%s.cl_perf.csv"%(experiment_dir,args.qrproblem),index_col=0)

perf_cl_plt = df_cl_perf.values[args.np2min:,args.dp2min:]
speedup_plt = df_c_perf.values[args.np2min:,args.dp2min:]/perf_cl_plt

rows,cols = perf_cl_plt.shape 
fig,ax = pyplot.subplots(nrows=1,ncols=2,figsize=(2*cols,rows+1),sharey=True)
norm_perf = colors.LogNorm(vmin=perf_cl_plt.min(),vmax=perf_cl_plt.max())
norm_speedup = colors.Normalize(vmin=0,vmax=speedup_plt.max())
ax[0].imshow(perf_cl_plt,cmap=args.cmap,norm=norm_perf)
ax[1].imshow(speedup_plt,cmap=args.cmap,norm=norm_speedup)
ax[0].set_ylabel(r"$n$")
ax[0].set_yticks(np.arange(rows),labels=[r"$2^{%d}$"%p for p in np.log2(df_c_perf.index)[args.np2min:]])
for i in range(2):
    ax[i].set_xlabel(r"$d$")
    ax[i].set_xticks(np.arange(cols),labels=[r"$2^{%d}$"%np.log2(int(v)) for v in df_c_perf.columns[args.dp2min:]])
    #pyplot.setp(ax[i].get_xticklabels(),rotation=45,ha="right",rotation_mode="anchor")
for i in range(rows):
    for j in range(cols):
        text = ax[0].text(j,i,"%.0e"%perf_cl_plt[i,j],ha="center",va="center",color=args.cperf)
        text = ax[1].text(j,i,"%.1f"%speedup_plt[i,j],ha="center",va="center",color=args.cleq1 if speedup_plt[i,j]<=1 else args.cgt1)
cax_perf = fig.add_axes([ax[0].get_position().x0,ax[0].get_position().y1+.025,ax[0].get_position().x1-ax[0].get_position().x0,0.025])
fig.colorbar(cm.ScalarMappable(norm=norm_perf,cmap=args.cmap),cax_perf,location="top",label="wall clock time")
cax_speedup = fig.add_axes([ax[1].get_position().x0,ax[1].get_position().y1+.025,ax[1].get_position().x1-ax[1].get_position().x0,0.025])
fig.colorbar(cm.ScalarMappable(norm=norm_speedup,cmap=args.cmap),cax_speedup,location="top",label="speedup factor")
fig.savefig("%s/nd_%s.png"%(experiment_dir,args.qrproblem),dpi=256,bbox_inches="tight")