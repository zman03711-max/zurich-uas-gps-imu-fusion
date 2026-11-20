"""Plot the separately defined common-position-observation protocol."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import ROOT

table=pd.read_csv(ROOT/'results/tables/position_only_fair_comparison.csv')
methods=['gps_only','fixed_gain','adaptive','kf_gins']
labels=['GPS position hold','Fixed fusion','Adaptive robust fusion','KF-GINS']
colors=['#8b949e','#d17a28','#117c83','#8457b5']
fig,axes=plt.subplots(1,3,figsize=(14,4.4))
for ax,family,xlabel in zip(axes,['dropout','noise','outlier'],['GPS outage (s)','GPS position noise std (m)','GPS offset (m)']):
    for method,label,color in zip(methods,labels,colors):
        s=table[(table.family==family)&(table.method==method)].sort_values('strength')
        ax.errorbar(s.strength,s.mean_rmse_m,yerr=s.std_rmse_m,marker='o',capsize=3,label=label,color=color)
    ax.set(xlabel=xlabel,ylabel='Horizontal RMSE (m)',title='Common GPS-position-only observations')
    ax.spines[['top','right']].set_visible(False)
axes[0].legend(fontsize=8)
fig.tight_layout()
for suffix in ['png','svg']:fig.savefig(ROOT/'figures'/f'09_position_only_comparison.{suffix}',dpi=300)
plt.close(fig)
print('FAIR_PROTOCOL_FIGURE_COMPLETE')
