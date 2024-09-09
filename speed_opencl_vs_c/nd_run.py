from problems import * 

import numpy as np 
from matplotlib import pyplot,cm,colors

problem_name = "lattice"
tag = "scratch"
n_min_pow2 = 0
n_max_pow2 = 17
n_plot_min_pow2 = 7
d_min_pow2 = 0
d_max_pow2 = 9
d_plot_min_pow2 = 0
discard_trials = 1
trials = 5
platform_id = 1 
device_id = 2

experiment_dir = "%s/nd_%s.%s"%(THISDIR,problem_name,tag)
run_problem = map_run_problem[problem_name]
n_pows2 = np.arange(n_min_pow2,n_max_pow2+1)
ns = 2**n_pows2
d_pows2 = np.arange(d_min_pow2,d_max_pow2+1)
ds = 2**d_pows2
d_mesh,n_mesh = np.meshgrid(ds,ns) 


kwargs_c,kwargs_cl = setup_speed_tests(platform_id,device_id)
remake_dir(experiment_dir,force=True)

df_c_perf,df_c_process = np.zeros_like(n_mesh,dtype=np.float64),np.zeros_like(n_mesh,dtype=np.float64) 
df_cl_perf,df_cl_process = np.zeros_like(n_mesh,dtype=np.float64),np.zeros_like(n_mesh,dtype=np.float64)

print("logging n_pows2 from %d to %d and d_pows2 from %d to %d"%(n_min_pow2,n_max_pow2,d_min_pow2,d_max_pow2))
for i in range(n_max_pow2-n_min_pow2+1):
    print("n_pows2 = %d: d_pows2 = "%n_pows2[i],end='',flush=True)
    n = ns[i]
    for j in range(d_max_pow2-d_min_pow2+1):
        print("%d, "%d_pows2[j],end='',flush=True)
        d = ds[j]
        kwargs_cl["global_size"] = (1,n,d)
        for t in range(discard_trials+trials):
            c_perf,c_process = run_problem(n=n,d=d,kwargs=kwargs_c)
            cl_perf,cl_process = run_problem(n=n,d=d,kwargs=kwargs_cl)
            if t>=discard_trials:
                df_c_perf[i,j] += c_perf 
                df_c_process[i,j] += c_process
                df_cl_perf[i,j] += cl_perf 
                df_cl_process[i,j] += cl_process
    print()
df_c_perf /= trials
df_c_process /= trials
df_cl_perf /= trials
df_cl_process /= trials

pd.DataFrame(df_c_perf,index=ns,columns=ds).to_csv("%s/nd_%s.c_perf.csv"%(experiment_dir,problem_name))
pd.DataFrame(df_c_process,index=ns,columns=ds).to_csv("%s/nd_%s.c_process.csv"%(experiment_dir,problem_name))
pd.DataFrame(df_cl_perf,index=ns,columns=ds).to_csv("%s/nd_%s.cl_perf.csv"%(experiment_dir,problem_name))
pd.DataFrame(df_cl_process,index=ns,columns=ds).to_csv("%s/nd_%s.cl_process.csv"%(experiment_dir,problem_name))

df_c_perf = pd.read_csv("%s/nd_%s.c_perf.csv"%(experiment_dir,problem_name),index_col=0)
df_c_process = pd.read_csv("%s/nd_%s.c_process.csv"%(experiment_dir,problem_name),index_col=0)
df_cl_perf = pd.read_csv("%s/nd_%s.cl_perf.csv"%(experiment_dir,problem_name),index_col=0)
df_cl_process = pd.read_csv("%s/nd_%s.cl_process.csv"%(experiment_dir,problem_name),index_col=0)

n_chop = n_plot_min_pow2-n_min_pow2
d_chop = d_plot_min_pow2-d_min_pow2
n_mesh_plt = n_mesh[n_chop:,d_chop:]
d_mesh_plt = d_mesh[n_chop:,d_chop:]
perf_plt = df_c_perf.values[n_chop:,d_chop:]/df_cl_perf.values[n_chop:,d_chop:]
process_plt = df_c_process.values[n_chop:,d_chop:]/df_cl_process.values[n_chop:,d_chop:]
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
ax[0].set_ylabel(r"$n$")
ax[0].set_yticks(np.arange(rows),labels=[r"$2^{%d}$"%p for p in n_pows2[n_chop:]])
for i in range(2):
    ax[i].set_xlabel(r"$d$")
    ax[i].set_xticks(np.arange(cols),labels=[r"$2^{%d}$"%p for p in d_pows2[d_chop:]])
    #pyplot.setp(ax[i].get_xticklabels(),rotation=45,ha="right",rotation_mode="anchor")
for i in range(rows):
    for j in range(cols):
        text = ax[0].text(j,i,"%.2f"%perf_plt[i,j],ha="center",va="center",color=textcol_leq1 if perf_plt[i,j]<=1 else textcol_gt1)
        text = ax[1].text(j,i,"%.1f"%process_plt[i,j],ha="center",va="center",color=textcol_leq1 if process_plt[i,j]<=1 else textcol_gt1)
cax = fig.add_axes([ax[0].get_position().x0,ax[0].get_position().y1+.1,ax[1].get_position().x1-ax[0].get_position().x0,0.05])
fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),cax,location="top",label="speedup factor")
fig.savefig("%s/nd_%s.png"%(experiment_dir,problem_name),dpi=256,bbox_inches="tight")
