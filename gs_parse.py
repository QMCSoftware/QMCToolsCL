from _problems import *

import numpy as np 
import pandas as pd 
from matplotlib import pyplot,cm,colors
import argparse 

parser = argparse.ArgumentParser(prog='Global Size Experiment')
parser.add_argument('-q','--qrproblem',type=str,nargs='?',default="lattice",help='quasi-random problem type %s')
parser.add_argument('-t','--tag',type=str,nargs='?',default="debug",help='experiment tag')
parser.add_argument('-c','--cmap',type=str,nargs='?',default="gist_yarg",help='colormap from matplotlib.cm')
parser.add_argument('-x','--colorleq1',type=str,nargs='?',default="k",help='text color when speedup<=1')
parser.add_argument('-y','--colorgt1',type=str,nargs='?',default="r",help='text color when speedup>1')
parser.add_argument('-z','--colorperf',type=str,nargs='?',default="r",help='text color for perf plot')
args = parser.parse_args()

experiment_dir = "gs_%s.%s"%(args.qrproblem,args.tag)
df_c_process = pd.read_csv("%s/gs_%s.c_process.csv"%(experiment_dir,args.qrproblem),index_col=0)
df_cl_process = pd.read_csv("%s/gs_%s.cl_process.csv"%(experiment_dir,args.qrproblem),index_col=0)
df_cl_perf = pd.read_csv("%s/gs_%s.cl_perf.csv"%(experiment_dir,args.qrproblem),index_col=0)

perf_cl_plt = df_cl_perf.values
speedup_plt = df_c_process.values/df_cl_process.values

rows,cols = perf_cl_plt.shape 
fig,ax = pyplot.subplots(nrows=1,ncols=2,figsize=(2*cols,rows+1),sharey=True)
norm_perf = colors.LogNorm(vmin=np.nanmin(perf_cl_plt),vmax=np.nanmax(perf_cl_plt))
norm_speedup = colors.Normalize(vmin=0,vmax=np.nanmax(speedup_plt))
ax[0].imshow(perf_cl_plt,cmap=args.cmap,norm=norm_perf)
ax[1].imshow(speedup_plt,cmap=args.cmap,norm=norm_speedup)
ax[0].set_ylabel(r"global size $n$")
ax[0].set_yticks(np.arange(rows),labels=[r"$2^{%d}$"%p for p in np.log2(df_cl_perf.index)])
for i in range(2):
    ax[i].set_xlabel(r"global size $d$")
    ax[i].set_xticks(np.arange(cols),labels=[r"$2^{%d}$"%np.log2(int(v)) for v in df_cl_perf.columns])
    #pyplot.setp(ax[i].get_xticklabels(),rotation=45,ha="right",rotation_mode="anchor")
for i in range(rows):
    for j in range(cols):
        text = ax[0].text(j,i,"%.0e"%perf_cl_plt[i,j],ha="center",va="center",color=args.colorperf)
        text = ax[1].text(j,i,"%.1f"%speedup_plt[i,j],ha="center",va="center",color=args.colorleq1 if speedup_plt[i,j]<=1 else args.colorgt1)
cax_perf = fig.add_axes([ax[0].get_position().x0,ax[0].get_position().y1+.025,ax[0].get_position().x1-ax[0].get_position().x0,0.025])
fig.colorbar(cm.ScalarMappable(norm=norm_perf,cmap=args.cmap),cax_perf,location="top",label="wall clock time")
cax_speedup = fig.add_axes([ax[1].get_position().x0,ax[1].get_position().y1+.025,ax[1].get_position().x1-ax[1].get_position().x0,0.025])
fig.colorbar(cm.ScalarMappable(norm=norm_speedup,cmap=args.cmap),cax_speedup,location="top",label="speedup factor")
fig.savefig("%s/gs_%s.png"%(experiment_dir,args.qrproblem),dpi=256,bbox_inches="tight")
