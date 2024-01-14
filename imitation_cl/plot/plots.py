import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.patheffects as pe
from matplotlib import colors as plt_colors
from matplotlib.ticker import FixedLocator
from math import pi
from glob import glob
import torch
import pathlib
import seaborn as sns
import matplotlib.ticker as ticker

from imitation_cl.plot.experiments import create_results_df, script_to_cl_map, script_to_traj_map
from imitation_cl.data.utils import get_dtw_summary

def get_cl_metrics(arr):
    """
    Given an array of validation accuracies (each current task along the rows,
    and accuracy of the tasks in the columns), this function computes the 
    CL metrics according to https://arxiv.org/pdf/1810.13166.pdf
    These metrics are computed:
        - Accuracy,
        - Backward Transfer,
        - BWT+,
        - REM,
    """

    n = arr.shape[0]

    # Accuracy considers the average accuracy by considering the diagonal 
    # elements as well as all elements below it
    # This is equivalent to computing the sum of the lower traingular matrix
    # and dividing that sum by N(N+1)/2
    acc = np.sum(np.tril(arr))/(n*(n+1)/2.0)

    # Backward transfer (BWT) 
    bwt = 0.0
    for i in range(1, n):
        for j in range(0, i):
            bwt += (arr[i,j] - arr[j,j])
    bwt /= (n*(n-1)/2.0)   

    rem = 1.0 - np.abs(np.min([bwt, 0.0]))
    bwt_plus = np.max([bwt, 0.0])

    return acc, bwt, bwt_plus, rem


def get_cl_acc_arr(database_df, dataset, cl_type, traj_type, data_dim, explicit_time, num_iters, num_tasks, threshold, metric_name):

    # Select the relevant rows
    query = f'(dataset=="{dataset}") and (cl_type=="{cl_type}") and (traj_type=="{traj_type}") and (data_dim=={data_dim}) and (num_iters=={num_iters}) and (explicit_time=={explicit_time})'
    selection_df = database_df.query(query)

    # Find the unique seeds
    seeds = np.unique(selection_df['seed'].tolist())

    # Initialize the acc matrix
    acc_arr = np.zeros((num_tasks, num_tasks, len(seeds)))
    for train_task_id in range(num_tasks):
        for eval_task_id in range(train_task_id+1):
            for i,seed in enumerate(seeds):
                metric = selection_df.query(f'(train_task_id=={train_task_id}) and (eval_task_id=={eval_task_id}) and (seed=={seed})')[metric_name]
                metric = metric.to_numpy()
                acc = (metric < threshold).sum()/len(metric)
                acc_arr[train_task_id, eval_task_id, i] = acc

    return acc_arr, seeds


def get_te_ms_sss_fs(database_df, seed_idx, dataset, cl_type, traj_type, data_dim, explicit_time, num_iters, num_tasks, threshold, metric_name):

    # Select the relevant rows
    query = f'(dataset=="{dataset}") and (cl_type=="{cl_type}") and (traj_type=="{traj_type}") and (data_dim=={data_dim}) and (num_iters=={num_iters}) and (explicit_time=={explicit_time}) and (obs_id==0)'
    selection_df = database_df.query(query).query('train_task_id == eval_task_id')

    # Find the unique seeds
    seeds = np.unique(selection_df['seed'].tolist())

    # Rows for the current seed
    selection_df = selection_df.query(f'seed == {seeds[seed_idx]}')

    num_tasks = np.unique(selection_df['num_tasks'].tolist())
    assert len(num_tasks) == 1
    num_tasks = num_tasks[0]

    task_ids = selection_df['train_task_id'].tolist()
    train_times = selection_df['train_time'].tolist()
    model_param_cnt = selection_df['model_param_cnt'].tolist()

    # Compute time efficiency
    summed_time_ratio = 0.0
    for task_id, time in zip(task_ids, train_times):
        summed_time_ratio += train_times[0]/time
    summed_time_ratio /= num_tasks
    time_efficiency = min(1.0, summed_time_ratio)

    # Compute model size efficiency
    ms = 0
    for task_id in range(num_tasks):
        ms += model_param_cnt[0]/model_param_cnt[task_id]
    ms /= num_tasks
    ms = min(1.0, ms)

    # Compute final model size
    max_param_query = f'(dataset=="{dataset}") and (traj_type=="{traj_type}") and (data_dim=={data_dim}) and (num_iters=={num_iters}) and (explicit_time=={explicit_time}) and (obs_id==0)'
    max_param_df = database_df.query(max_param_query).query('train_task_id == eval_task_id')
    max_param_df = max_param_df.query(f'seed == {seeds[seed_idx]}')
    max_param_df = max_param_df.query(f'train_task_id == {task_id}')
    max_model_param_cnt = max(max_param_df['model_param_cnt'].tolist())
    fs = 1.0 - (model_param_cnt[task_id]/max_model_param_cnt)

    # Samples storage size efficiency
    if cl_type == 'REP':
        sss = 1.0 - min(1.0, ((num_tasks * (num_tasks+1))/(2*num_tasks))/num_tasks)
    else:
        sss = 1.0

    return time_efficiency, ms, sss, fs

def create_cl_df(database_df,cl_queries):
    cl_data = list()
    for cl_query in cl_queries:
        # print(f'Processing {cl_query}')
        acc_arr, seeds = get_cl_acc_arr(database_df, **cl_query)
        num_seeds = acc_arr.shape[-1]
        for seed_idx in range(num_seeds):

            # ACC, BWT, BWT_PLUS, REM
            acc, bwt, bwt_plus, rem = get_cl_metrics(acc_arr[:,:,seed_idx])

            # TE, MS, SSS
            time_efficiency, ms, sss, fs = get_te_ms_sss_fs(database_df, seed_idx, **cl_query)


            row = dict(dataset=cl_query['dataset'],
                    cl_type=cl_query['cl_type'],
                    traj_type=cl_query['traj_type'],
                    data_dim=cl_query['data_dim'],
                    explicit_time=cl_query['explicit_time'],
                    threshold=cl_query['threshold'],
                    metric_name=cl_query['metric_name'],
                    seed=seeds[seed_idx],
                    acc=acc,
                    bwt=bwt,
                    bwt_plus=bwt_plus,
                    rem=rem,
                    ms=ms,  
                    te=time_efficiency, 
                    fs=fs,
                    sss=sss,
                    )
            cl_data.append(row)
    cl_df = pd.DataFrame(cl_data)
    return cl_df

def create_cl_table(cl_df, dataset, traj_type, explicit_time):

    # METHOD ACC REM MS TE FS CLscore CLstability
    r_df = cl_df.query(f'(dataset=="{dataset}") and (traj_type=="{traj_type}") and (explicit_time=={explicit_time})').groupby(['cl_type']).mean(numeric_only=True)
    data_dim = r_df['data_dim'].tolist()[0]
    #r_df = r_df[['acc', 'bwt_plus', 'rem', 'ms', 'te', 'fs', 'sss']]
    r_df = r_df[['acc', 'rem', 'ms', 'te', 'fs', 'sss']]
    r_df['cl_score'] = (r_df['acc'] + r_df['rem'] +r_df['ms'] + r_df['te'] + r_df['fs'] + r_df['sss'])/6
    r_df['cl_stability'] = 1.0 - r_df[['acc', 'rem', 'ms', 'te', 'fs', 'sss']].std(axis=1)

    # String name for saving
    table_str_name = f'tbl_{dataset}_{int(data_dim)}D_t{explicit_time}_{traj_type}.tex'
    return r_df, table_str_name

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


class Plotter:

    def __init__(self,
                 logbase_dir,
                 log_comment,
                 dataset,
                 data_dim,
                 num_iters,
                 num_tasks,
                 cl_types,
                 traj_types,
                 explicit_times,
                 cl_threshold_pos,
                 cl_threshold_ori=None,
                 cl_metric_pos=None,
                 cl_metric_ori=None,
                 results_save_dir=None,
                 plots_save_dir=None,
                 force_logread=False,
                 results_save=False,
                 metric_label_map={'dtw':'DTW Error', 
                                   'frechet':'Frechet Error',
                                   'swept_area':'Swept Area Error',
                                   'quat_error':'Quat Error'},
                 cl_label_map={'SG':'SG', 
                               'FT':'FT',
                               'REP':'REP',
                               'SI':'SI',
                               'MAS':'MAS',
                               'HN':'HN',
                               'CHN':'CHN'},
                 traj_label_map={'NODE':'NODE', 
                                 'LSDDM': 'sNODE'},
                 colors_models_node={'FT': '#f58231', 
                                     'SI': '#911eb4', 
                                     'MAS': 'rosybrown',
                                     'HN': '#e6194B', 
                                     'SG': '#1E90FF', 
                                     'iFlow': '#800000',
                                     'CHN': '#3cb44b', 
                                     'REP': 'gray'},
                 color_models_lsddm={'FT': '#f79b45', 
                                     'SI': '#b41ea5', 
                                     'MAS': '#bca68f',
                                     'HN': '#fc5bb9', 
                                     'SG': '#47a5ed', 
                                     'iFlow': '#800000',
                                     'CHN': '#49e398', 
                                     'REP': '#ababa9'},
                 color_models_traj={'NODE': '#eefa14', 'LSDDM': '#3d3d3b'},
                 color_models_combined={
                                    'SG_LSDDM': '#1E90FF', 
                                    'SG_NODE': '#1E90FF', 
                                    'FT_LSDDM': '#f58231', 
                                    'FT_NODE': '#f58231', 
                                    'REP_LSDDM': 'gray',
                                    'REP_NODE': 'gray',
                                    'SI_LSDDM': '#911eb4', 
                                    'SI_NODE': '#911eb4', 
                                    'MAS_LSDDM': 'rosybrown',
                                    'MAS_NODE': 'rosybrown',
                                    'HN_LSDDM': '#e6194B', 
                                    'HN_NODE': '#e6194B', 
                                    'CHN_LSDDM': '#3cb44b', 
                                    'CHN_NODE': '#3cb44b', 
                                    },
                 ls_models={'NODE':'solid', 'LSDDM': 'solid'},  # Line styles for different traj types
                 marker_models={'NODE':'o', 'LSDDM': 'o'},      # Marker styles for different traj types
                 lw_models={'FT': 2,                          # Control line widths of different cl types
                            'SI': 2, 
                            'MAS': 2,
                            'HN': 2, 
                            'SG': 2, 
                            'iFlow': 2,
                            'CHN': 2, 
                            'REP': 2},
                 custom_file_suffix=''
                 ) -> None:
                
        self.logbase_dir = logbase_dir
        self.log_comment = log_comment
        self.dataset = dataset
        self.data_dim = data_dim
        self.num_iters = num_iters
        self.num_tasks = num_tasks
        self.cl_types = cl_types
        self.traj_types = traj_types
        self.explicit_times = explicit_times
        self.cl_metric_pos = cl_metric_pos
        self.cl_metric_ori = cl_metric_ori
        self.cl_threshold_pos = cl_threshold_pos
        self.cl_threshold_ori = cl_threshold_ori
        self.results_save_dir = results_save_dir
        self.plots_save_dir = plots_save_dir
        self.force_logread = force_logread
        self.results_save = results_save
        self.metric_label_map = metric_label_map
        self.cl_label_map = cl_label_map
        self.traj_label_map = traj_label_map
        self.colors_models_node = colors_models_node
        self.color_models_lsddm = color_models_lsddm
        self.color_models_traj = color_models_traj
        self.color_models_combined = color_models_combined
        self.ls_models = ls_models
        self.marker_models = marker_models
        self.lw_models = lw_models
        self.custom_file_suffix = custom_file_suffix

        # Read experimental logs
        self.read_log()

        # Create the continual learning data
        if self.cl_metric_pos is not None and self.cl_threshold_pos is not None:
            self.create_cl_pos_data()

        if self.cl_metric_ori is not None and self.cl_threshold_ori is not None:
            self.create_cl_ori_data()

        # For storing aggregated CL metrics
        # Key: Tuple of (pos_ori,traj_type)
        self.cl_agg_metrics_dict = dict()

        # To switch between 2 line plots and multiple line plots
        self.bicolor = False

    def read_log(self) -> None:
        self.save_file = f'results_{self.dataset}_{self.data_dim}D{self.custom_file_suffix}.csv'
        self.save_file_path = os.path.join(self.results_save_dir, self.save_file)

        # Obtain the experiment results
        if not os.path.exists(self.save_file_path) or self.force_logread:
            # If results file does not exist
            # or if forced to read log dir
            print(f'Reading logbase_dir: {self.logbase_dir}')
            self.results_df = create_results_df(base_logdir=self.logbase_dir, 
                                                insert_comment=self.log_comment,
                                                verbose=False)
            print(f'Found {len(self.results_df)} entries in {self.logbase_dir}')
            # Save if needed
            if self.results_save:
                print(f'Saving results_path: {self.save_file_path}')
                self.results_df.to_csv(self.save_file_path, index=False)
        else:
            # Read the existing results file
            print(f'Reading existing results_path: {self.save_file_path}')
            self.results_df = pd.read_csv(self.save_file_path, low_memory=False)
        cols = list()
        for c in self.results_df.columns:
            cols.append(c)
        print(f'Columns in dataframe: {cols}')

    def merge(self, plotter2):
        self.results_df = pd.concat([self.results_df, plotter2.results_df])

    def create_cl_pos_data(self) -> None:
        print('Creating CL results (pos)')
        cl_queries = list()

        for cl_type in self.cl_types:
            for traj_type in self.traj_types:
                for explicit_time in self.explicit_times:
                    cl_queries.append(dict(dataset=self.dataset, 
                                           cl_type=cl_type, 
                                           traj_type=traj_type, 
                                           data_dim=self.data_dim, 
                                           explicit_time=explicit_time, 
                                           num_iters=self.num_iters, 
                                           num_tasks=self.num_tasks, 
                                           threshold=self.cl_threshold_pos, 
                                           metric_name=self.cl_metric_pos)
                                      )
        # Create CL dataframe
        self.cl_pos_results_df = create_cl_df(self.results_df, cl_queries)
        print(f'Created {len(self.cl_pos_results_df)} records for CL (pos)')

    def create_cl_ori_data(self) -> None:
        print('Creating CL results (ori)')
        cl_queries = list()

        for cl_type in self.cl_types:
            for traj_type in self.traj_types:
                for explicit_time in self.explicit_times:
                    cl_queries.append(dict(dataset=self.dataset, 
                                           cl_type=cl_type, 
                                           traj_type=traj_type, 
                                           data_dim=self.data_dim, 
                                           explicit_time=explicit_time, 
                                           num_iters=self.num_iters, 
                                           num_tasks=self.num_tasks, 
                                           threshold=self.cl_threshold_ori, 
                                           metric_name=self.cl_metric_ori)
                                      )
        # Create CL dataframe
        self.cl_ori_results_df = create_cl_df(self.results_df, cl_queries)
        print(f'Created {len(self.cl_ori_results_df)} records for CL (ori)')

    def plot_cumulative_errors(self,
                               ax, 
                               explicit_time, 
                               cl_type, 
                               traj_type, 
                               metric_name, 
                               plot_log=True, 
                               label_type='combined', # Options: 'combined', 'cl', 'traj'
                               markersize=5.0):

        # Filter results for the dataset and explicit time
        filtered_database_df = self.results_df.loc[
            (self.results_df.dataset == self.dataset) &
            (self.results_df.explicit_time == explicit_time) &
            (self.results_df.data_dim == self.data_dim)
        ]

        filtered_database_df = filtered_database_df.reset_index()
        filtered_database_df.drop(['index'], axis=1, inplace=True)

        # Find the number of tasks
        num_tasks = np.unique(filtered_database_df.num_tasks.values)[0]

        # Create a dict to store metric values
        metric_list = list()

        for train_task_id in range(num_tasks):

            val = filtered_database_df.loc[
                (filtered_database_df.cl_type == cl_type) &
                (filtered_database_df.traj_type == traj_type) &
                (filtered_database_df.train_task_id == train_task_id) &
                (filtered_database_df.eval_task_id <= train_task_id)
            ][metric_name].values

            metric_list.append(val)

        # Create a dict to store medians and percentiles of metric values (each metric has a list containing arrays)
        metric_plot = list()
        metric_log10_plot = list()

        for val in metric_list:
            low = np.percentile(val, 25)
            mid = np.median(val)
            high = np.percentile(val, 75)
            metric_plot.append((low, mid, high))

            low_log10 = np.percentile(np.log10(val), 25)
            mid_log10 = np.median(np.log10(val))
            high_log10 = np.percentile(np.log10(val), 75)
            metric_log10_plot.append((low_log10, mid_log10, high_log10))


        if self.bicolor:
            color = self.color_models_traj[traj_type]
            plot_kwargs = dict(color=color,
                            marker=self.marker_models[traj_type],
                            markersize=markersize,
                            markeredgecolor=adjust_lightness(color, 0.5),
                            markeredgewidth=1.5,
                            lw=self.lw_models[cl_type],
                            ls=self.ls_models[traj_type])
        else:
            color = self.colors_models_node[cl_type] if traj_type == 'NODE' else self.color_models_lsddm[cl_type]
            plot_kwargs = dict(color=color,
                            marker=self.marker_models[traj_type],
                            markersize=markersize,
                            markeredgecolor=adjust_lightness(color, 0.5),
                            markeredgewidth=0.5,
                            lw=self.lw_models[cl_type],
                            ls=self.ls_models[traj_type])

        plot_data = metric_log10_plot if plot_log else metric_plot

        if label_type=='combined':
            label = f'{self.cl_label_map[cl_type]}_{self.traj_label_map[traj_type]}'
        elif label_type=='cl':
            label = self.cl_label_map[cl_type]
        elif label_type=='traj':
            label = self.traj_label_map[traj_type]
        else:
            raise NotImplementedError(f'Invalid label_type: {label_type}')
        ax.plot([m[1] for m in plot_data],
                label=label,
                path_effects=[pe.Stroke(linewidth=self.lw_models[cl_type]*2, 
                                        foreground=adjust_lightness(color, 0.8)), 
                                        pe.Normal()],
                **plot_kwargs)

        x = np.arange(0, num_tasks)
        ax.fill_between(x, [m[0] for m in plot_data], [m[2]
                        for m in plot_data], color=color, alpha=0.15)

    def plot_cumu_inter_cl(self,
                           ax,
                           cl_types,
                           traj_type,
                           ylim,
                           explicit_time=1,
                           metric_name='dtw',
                           plot_log=False,
                           label_type='combined',
                           fontsizes=dict(label=12,title=16,legend=12),
                           showlegend=True,
                           showlabel=dict(x=True, y=True),
                           title=None) -> None:
        """
        Cumulative plots comparing the different cl_types for a given traj_type
        """

        for cl_type in cl_types:
            self.plot_cumulative_errors(ax=ax, 
                                        explicit_time=explicit_time, 
                                        cl_type=cl_type,
                                        traj_type=traj_type, 
                                        metric_name=metric_name, 
                                        plot_log=plot_log, 
                                        label_type=label_type)

        ax.set_ylim(ylim)
        if showlabel['y']:
            ax.set_ylabel(self.metric_label_map[metric_name], fontsize=fontsizes['label'])
        if showlabel['x']:
            ax.set_xlabel('Current task', fontsize=fontsizes['label'])

        dtw_thresh = get_dtw_summary(dataset=self.dataset, data_dim=self.data_dim, norm=False, verbose=False)

        #ax.plot(dtw_thresh, color='black', ls='-.')
        if showlegend:
            ax.legend(ncol=4, loc='upper left', fontsize=fontsizes['legend'])
        ax.grid(True)

        if title is not None:
            ax.set_title(title, fontsize=fontsizes['title'])

    def plot_cumu_inter_traj(self,
                             ax,
                             cl_type,
                             ylim,
                             explicit_time=1,
                             metric_name='dtw',
                             plot_log=False,
                             label_type='combined',
                             fontsizes=dict(label=12,title=16,legend=12),
                             showlegend=True,
                             showlabel=dict(x=True, y=True),
                             title=None
                             ) -> None:
        """
        Cumulative plots comparing the different traj_types for a given cl_type
        """
        for traj_type in self.traj_types:
            cl_types = [cl_type]
            self.plot_cumu_inter_cl(ax=ax,
                    cl_types=cl_types,
                    traj_type=traj_type,
                    ylim=ylim,
                    explicit_time=explicit_time,
                    metric_name=metric_name,
                    plot_log=plot_log,
                    label_type=label_type,
                    fontsizes=fontsizes,
                    showlegend=showlegend,
                    showlabel=showlabel,
                    title=title)

    def plot_stability(self) -> None:
        pass

    def plot_first_last_eval(self) -> None:
        pass

    def plot_traj(self) -> None:
        pass

    def create_cl_table(self,
                        traj_type,
                        pos_ori,
                        save_latex=True,
                        dirname='tables',
                        filename='table.tex',
                        display=True,
                        notebook=True,
                        explicit_time=1,
                        cl_sort_map=dict(SG=0, FT=1, REP=2,
                                         SI=3, MAS=4, HN=5, CHN=6),
                        col_names_display=[
                            'MET', 'ACC', 'REM', 'MS', 'TE', 'FS', 'SSS', 'CL$_{sco}$', 'CL$_{stab}$'],
                        col_names_highlight=[
                            'ACC', 'REM', 'MS', 'TE', 'FS', 'SSS', 'CL$_{sco}$', 'CL$_{stab}$'],
                        highlight_props='cellcolor:{yellow}; bfseries: ;',
                        float_precision=3,
                        ) -> None:

        # Select position or orientation data
        cl_results_df = self.cl_pos_results_df if pos_ori=='pos' else self.cl_ori_results_df
        # Compute aggregated CL metrics
        cl_agg_metrics_df, _ = create_cl_table(cl_results_df, 
                                                    dataset=self.dataset, 
                                                    traj_type=traj_type, 
                                                    explicit_time=explicit_time)

        # Sort rows
        cl_agg_metrics_df = cl_agg_metrics_df.sort_values(by=['cl_type'], 
                                                        key=lambda x: x.map(cl_sort_map))

        # Change header names
        cl_agg_metrics_df.index.names = [col_names_display[0]]
        cl_agg_metrics_df.columns = col_names_display[1:]

        # Store for creating radar charts
        self.cl_agg_metrics_dict[(pos_ori,traj_type)] = cl_agg_metrics_df.copy()

        # Move MET to a regular column and create a numeric index
        cl_agg_metrics_df = cl_agg_metrics_df.reset_index()

        if display and not notebook:
            print(cl_agg_metrics_df)
        if display and notebook:
            from IPython.display import display
            display(cl_agg_metrics_df) 

        if save_latex:
            # Chained formatting for latex file:
            # Hide index
            # Highlight max numeric columns
            # Float precision
            # Save to latex
            cl_agg_metrics_df.style \
                .hide(axis="index") \
                .highlight_max(subset=col_names_highlight, axis=0, props=highlight_props) \
                .format(precision=float_precision) \
                .to_latex(os.path.join(dirname, filename), hrules=True)


    def get_cl_table(self,
                     traj_type,
                     save_latex=True,
                     display=True,
                     notebook=True,
                     explicit_time=1,
                     pos_ori='pos',
                     cl_sort_map = dict(SG=0, FT=1, REP=2, SI=3, MAS=4, HN=5, CHN=6),
                     col_names_display=['MET','ACC','REM','MS','TE','FS','SSS','CL$_{sco}$','CL$_{stab}$']) -> None:
    
        # Select position or orientation data
        cl_results_df = self.cl_pos_results_df if pos_ori=='pos' else self.cl_ori_results_df
        # Compute aggregated CL metrics
        cl_agg_metrics_df, _ = create_cl_table(cl_results_df, 
                                                    dataset=self.dataset, 
                                                    traj_type=traj_type, 
                                                    explicit_time=explicit_time)
        
        # Sort rows
        cl_agg_metrics_df = cl_agg_metrics_df.sort_values(by=['cl_type'], 
                                                                    key=lambda x: x.map(cl_sort_map))
        
        # Change header names
        cl_agg_metrics_df.index.names = [col_names_display[0]]
        cl_agg_metrics_df.columns = col_names_display[1:]


        if save_latex:

            # TODO: Apply bold for max of numeric columns                
            # Not working: https://flopska.com/highlighting-pandas-to_latex-output-in-bold-face-for-extreme-values.html

            # Different names for position or orientation
            self.table_label_name = f'{self.dataset}_{self.data_dim}D_{traj_type}_t{explicit_time}_{pos_ori}'
            self.table_file_name = f'{self.table_label_name}.tex'

            # Set the styler
            #styler = cl_agg_metrics_df.style
            # Format floats
            # Highlight max in each numeric column
            cl_agg_metrics_df.style.to_latex(os.path.join(self.plots_save_dir, self.table_file_name), 
                                                  label=self.table_label_name)
            
        # Store for creating radar charts
        self.cl_agg_metrics_dict[(pos_ori,traj_type)] = cl_agg_metrics_df
            
    def plot_cl_radar(self,
                      ax,
                      cl_type,
                      title,
                      pos_ori='pos',
                      traj_types=['NODE','LSDDM'],
                      fontsizes=dict(label=12,title=16,legend=12),
                      intra_slice_shift = 0.05,  # Controls the spacing between LSDDM and NODE for the same CL metric
                      width_ratio = 0.4,         # Controls the gap between different CL metrics
                      alpha=0.2,
                      hatch='.',
                      bordercolor='k',           # Border to separate CL metrics
                      ) -> None:
        """
        Adapted from: https://stackoverflow.com/questions/62938954/pie-radar-chart-in-python-pizza-slices
        """

        assert ax.name == 'polar', 'ax is not polar'

        hex_color_node = self.color_models_traj['NODE']
        hex_color_lsddm = self.color_models_traj['LSDDM']

        rgb_color_node = plt_colors.to_rgb(hex_color_node)
        rgb_color_lsddm = plt_colors.to_rgb(hex_color_lsddm)

        traj_kwargs = dict(NODE=dict(facecolor=[*rgb_color_node, alpha],
                                     edgecolor=rgb_color_node,
                                     linewidth=2,
                                     hatch=hatch),
                           LSDDM=dict(facecolor=[*rgb_color_lsddm, alpha],
                                      edgecolor=rgb_color_lsddm,
                                      linewidth=2,
                                      hatch=None))

        dfs = dict()
        for traj_type in traj_types:
            dfs[traj_type] = self.cl_agg_metrics_dict[(pos_ori,traj_type)]

        categories = list(dfs[traj_types[0]])[0:6]
        N = len(categories)
        angles = np.linspace(0, 2 * pi, N, endpoint=False)
        angles_mids = angles + (angles[1] / 2)

        ax.set_theta_offset(pi/2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles_mids)
        ax.set_xticklabels(categories)
        ax.xaxis.set_minor_locator(FixedLocator(angles))

        values = dict()
        for k,v in dfs.items():
            values[k] = v.loc[cl_type].values[:-2]

        ax.bar(angles_mids-pi/12+intra_slice_shift, 
            values['LSDDM'], 
            width=(angles[1] - angles[0])*width_ratio,
            facecolor=traj_kwargs['LSDDM']['facecolor'], 
            edgecolor=traj_kwargs['LSDDM']['edgecolor'], 
            linewidth=traj_kwargs['LSDDM']['linewidth'], 
            label='sNODE',
            hatch=traj_kwargs['LSDDM']['hatch'])

        ax.bar(angles_mids+pi/12-intra_slice_shift, 
            values['NODE'], 
            width=(angles[1] - angles[0])*width_ratio,
            facecolor=traj_kwargs['NODE']['facecolor'], 
            edgecolor=traj_kwargs['NODE']['edgecolor'], 
            linewidth=traj_kwargs['NODE']['linewidth'], 
            label='NODE', 
            hatch=traj_kwargs['NODE']['hatch'])

        # Draw ylabels
        ax.set_rlabel_position(0)
        ax.set_yticks([0.25,0.5,0.75,1.0])
        ax.set_yticklabels([0.25,0.5,0.75,1.0], fontsize=fontsizes['label'])
        ax.set_ylim(0, 1.05)

        ax.set_xticklabels(['ACC','REM','MS','TE','FS','SSS'], fontsize=fontsizes['label'])

        # For better segregating the CL metrics
        ax.grid(True, axis='x', which='minor', color=bordercolor, lw=2)

        ax.grid(False, axis='x', which='major')
        ax.grid(True, axis='y', which='major')
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=fontsizes['title'], loc='left')

        ax.spines['polar'].set_color(bordercolor)
        ax.spines['polar'].set_linewidth(2)

        ax.legend(loc='upper left', bbox_to_anchor=(0.9, 1), fontsize=fontsizes['legend'])

    def plot_cl_radar_inter_cl(
            self,
            ax=None,
            title=None,
            pos_ori='pos',
            cl_types=['SG', 'FT', 'REP', 'SI', 'MAS', 'HN', 'CHN'],
            fontsizes=dict(label=12,title=16,legend=12),
            intra_slice_shift = 0.05,  # Controls the spacing between LSDDM and NODE for the same CL metric
            width_ratio = 0.4,         # Controls the gap between different CL metrics
            alpha=0.2,
            hatch='.',
            bordercolor='k',           # Border to separate CL metrics
            ) -> None:

        #assert ax.name == 'polar', 'ax is not polar'

        rgb_color_sg = plt_colors.to_rgb(self.colors_models_node['SG'])
        rgb_color_ft = plt_colors.to_rgb(self.colors_models_node['FT'])
        rgb_color_rep = plt_colors.to_rgb(self.colors_models_node['REP'])
        rgb_color_si = plt_colors.to_rgb(self.colors_models_node['SI'])
        rgb_color_mas = plt_colors.to_rgb(self.colors_models_node['MAS'])
        rgb_color_hn = plt_colors.to_rgb(self.colors_models_node['HN'])
        rgb_color_chn = plt_colors.to_rgb(self.colors_models_node['CHN'])

        traj_kwargs = dict(SG=dict(facecolor=[*rgb_color_sg, alpha],
                                edgecolor=rgb_color_sg,
                                linewidth=2,
                                hatch=hatch),
                        FT=dict(facecolor=[*rgb_color_ft, alpha],
                                edgecolor=rgb_color_ft,
                                linewidth=2,
                                hatch=hatch),
                        REP=dict(facecolor=[*rgb_color_rep, alpha],
                                edgecolor=rgb_color_rep,
                                linewidth=2,
                                hatch=hatch),
                        SI=dict(facecolor=[*rgb_color_si, alpha],
                                edgecolor=rgb_color_si,
                                linewidth=2,                               hatch=hatch),
                        MAS=dict(facecolor=[*rgb_color_mas, alpha],
                                edgecolor=rgb_color_mas,
                                linewidth=2,
                                hatch=hatch),
                        HN=dict(facecolor=[*rgb_color_hn, alpha],
                                edgecolor=rgb_color_hn,
                                linewidth=2,
                                hatch=hatch),
                        CHN=dict(facecolor=[*rgb_color_chn, alpha],
                                edgecolor=rgb_color_chn,
                                linewidth=2,
                                hatch=hatch),
        )

        dfs = self.cl_agg_metrics_dict[(pos_ori,'LSDDM')]

        # The individual CL metrics
        categories = list(dfs.columns)[0:6]

        N = len(categories)
        angles = np.linspace(0, 2 * pi, N, endpoint=False)
        angles_mids = angles + (angles[1] / 2)

        ax.set_theta_offset(pi/2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles_mids)
        ax.set_xticklabels(categories)
        ax.xaxis.set_minor_locator(FixedLocator(angles))

        for i,cl_type in enumerate(cl_types):
            values = dfs.loc[cl_type, categories].values.flatten().tolist()
            ax.bar(
                angles_mids-((i)*pi/21)+intra_slice_shift,
                values,
                width=(angles[1] - angles[0])*width_ratio,
                facecolor=traj_kwargs[cl_type]['facecolor'], 
                edgecolor=traj_kwargs[cl_type]['edgecolor'], 
                linewidth=traj_kwargs[cl_type]['linewidth'], 
                label=cl_type,
                hatch=traj_kwargs[cl_type]['hatch'])

        # Draw ylabels
        ax.set_rlabel_position(0)
        ax.set_yticks([0.25,0.5,0.75,1.0])
        ax.set_yticklabels([0.25,0.5,0.75,1.0], fontsize=fontsizes['label'])
        ax.set_ylim(0, 1.05)

        ax.set_xticklabels(['ACC','REM','MS','TE','FS','SSS'], fontsize=fontsizes['label'])

        ax.grid(True, axis='x', which='minor')
        ax.grid(False, axis='x', which='major')

        # For better segregating the CL metrics
        ax.grid(True, axis='x', which='minor', color=bordercolor, lw=2)
        ax.spines['polar'].set_color(bordercolor)
        ax.spines['polar'].set_linewidth(2)

        ax.grid(True, axis='y', which='major')
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=fontsizes['title'], loc='left')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1.1), fontsize=fontsizes['legend'])

    def plot_cl_radar_inter_cl_traj(
        self,
        ax=None,
        title=None,
        pos_ori='pos',
        cl_traj_types=['CHN_LSDDM', 'CHN_NODE'],
        fontsizes=dict(label=12,title=16,legend=12),
        intra_slice_shift = 0.05,  # Controls the spacing between LSDDM and NODE for the same CL metric
        width_ratio = 0.4,         # Controls the gap between different CL metrics
        alpha=0.2,
        hatch='.',
        bordercolor='k',           # Border to separate CL metrics
        ) -> None:

        #assert ax.name == 'polar', 'ax is not polar'

        rgb_color_sg = plt_colors.to_rgb(self.colors_models_node['SG'])
        rgb_color_ft = plt_colors.to_rgb(self.colors_models_node['FT'])
        rgb_color_rep = plt_colors.to_rgb(self.colors_models_node['REP'])
        rgb_color_si = plt_colors.to_rgb(self.colors_models_node['SI'])
        rgb_color_mas = plt_colors.to_rgb(self.colors_models_node['MAS'])
        rgb_color_hn = plt_colors.to_rgb(self.colors_models_node['HN'])
        rgb_color_chn = plt_colors.to_rgb(self.colors_models_node['CHN'])

        traj_kwargs = dict(SG=dict(facecolor=[*rgb_color_sg, alpha],
                                edgecolor=rgb_color_sg,
                                linewidth=2,
                                hatch=hatch),
                        FT=dict(facecolor=[*rgb_color_ft, alpha],
                                edgecolor=rgb_color_ft,
                                linewidth=2,
                                hatch=hatch),
                        REP=dict(facecolor=[*rgb_color_rep, alpha],
                                edgecolor=rgb_color_rep,
                                linewidth=2,
                                hatch=hatch),
                        SI=dict(facecolor=[*rgb_color_si, alpha],
                                edgecolor=rgb_color_si,
                                linewidth=2,                               hatch=hatch),
                        MAS=dict(facecolor=[*rgb_color_mas, alpha],
                                edgecolor=rgb_color_mas,
                                linewidth=2,
                                hatch=hatch),
                        HN=dict(facecolor=[*rgb_color_hn, alpha],
                                edgecolor=rgb_color_hn,
                                linewidth=2,
                                hatch=hatch),
                        CHN=dict(facecolor=[*rgb_color_chn, alpha],
                                edgecolor=rgb_color_chn,
                                linewidth=2,
                                hatch=hatch),
        )

        # Change the index to cl_traj
        dfs1 = self.cl_agg_metrics_dict[(pos_ori,'LSDDM')].copy()
        dfs1.index = dfs1.index.map(lambda x: f'{x}_LSDDM')

        dfs2 = self.cl_agg_metrics_dict[(pos_ori,'NODE')].copy()
        dfs2.index = dfs2.index.map(lambda x: f'{x}_NODE')

        # Merge
        dfs = pd.concat([dfs1, dfs2])

        cl_scores = dict()
        for cl_traj_type in cl_traj_types:
            cl_scores[cl_traj_type] = dfs.loc[cl_traj_type, 'CL$_{sco}$']

        # The individual CL metrics
        categories = list(dfs.columns)[0:6]

        N = len(categories)
        angles = np.linspace(0, 2 * pi, N, endpoint=False)
        angles_mids = angles + (angles[1] / 2)

        ax.set_theta_offset(pi/2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles_mids)
        ax.set_xticklabels(categories)
        ax.xaxis.set_minor_locator(FixedLocator(angles))

        for i,cl_traj_type in enumerate(cl_traj_types):
            cl_type = cl_traj_type.replace('_NODE', '').replace('_LSDDM', '')
            values = dfs.loc[cl_traj_type, categories].values.flatten().tolist()
            bars = ax.bar(
                angles_mids-((i)*pi/21)+intra_slice_shift,
                values,
                width=(angles[1] - angles[0])*width_ratio,
                facecolor=traj_kwargs[cl_type]['facecolor'] if 'LSDDM' in cl_traj_type else (1,1,1,alpha), # white with alpha 
                edgecolor=traj_kwargs[cl_type]['edgecolor'], 
                linewidth=traj_kwargs[cl_type]['linewidth'], 
                hatch=hatch if 'NODE' in cl_traj_type else '', 
                label=cl_traj_type,
                )
            

        # Draw ylabels
        ax.set_rlabel_position(0)
        ax.set_yticks([0.25,0.5,0.75,1.0])
        ax.set_yticklabels([0.25,0.5,0.75,1.0], fontsize=fontsizes['label'])
        ax.set_ylim(0, 1.05)

        ax.set_xticklabels(['ACC','REM','MS','TE','FS','SSS'], fontsize=fontsizes['label'])

        ax.grid(True, axis='x', which='minor')
        ax.grid(False, axis='x', which='major')

        # For better segregating the CL metrics
        ax.grid(True, axis='x', which='minor', color=bordercolor, lw=2)
        ax.spines['polar'].set_color(bordercolor)
        ax.spines['polar'].set_linewidth(2)

        ax.grid(True, axis='y', which='major')
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=fontsizes['title'], loc='left')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1.1), fontsize=fontsizes['legend'])

        return cl_scores

    def box_all(
            self,
            ax,
            y_metric='dtw',
            ylim=(0,8_000),
            box_to_modify = [6, 8],
            to_keep=[
                "SG_LSDDM", 
                "FT_LSDDM", 
                "REP_LSDDM", 
                "SI_LSDDM", 
                "MAS_LSDDM",
                "HN_LSDDM", 
                "HN_NODE", 
                "CHN_LSDDM", 
                "CHN_NODE"
                ],
            sci=True,
            modify_box_config={'color':'white', 'hatch':'////', 'ls':'-'},
            y_ref_config={'y':10_000, 'ls':'--', 'lw':1.5, 'color':'darkgray'},
            width=0.5,
            linewidth=1.5,
            ):

        # Create a new derived column by combining cl and traj types
        self_results_cl_traj_df = self.results_df.copy()
        self_results_cl_traj_df['cl_traj_reg'] = self_results_cl_traj_df['cl_type'].astype(str) + '_' + \
            self_results_cl_traj_df['traj_type'].astype(str)    

        # Keep required combinations and discard the rest
        self_results_cl_traj_df = self_results_cl_traj_df.query('cl_traj_reg in @to_keep')

        # Create the main box plot
        sns.boxplot(ax=ax, 
                    data=self_results_cl_traj_df, 
                    x='cl_traj_reg', 
                    y=y_metric, 
                    fliersize=0, 
                    order=to_keep, 
                    palette=self.color_models_combined,
                    width=width,
                    linewidth=linewidth,
                    )

        # Find the boxplots
        box_container = ax.artists

        # How many lines in each boxplot
        lines_per_boxplot = len(ax.lines) // len(box_container)

        # Go through each boxplot and modify the ones necessary
        for box_id, (box, xtick) in enumerate(zip(box_container, ax.get_xticklabels())):
            if box_id in box_to_modify:
                box.set_facecolor(modify_box_config['color'])
                edgecolor = self.color_models_combined[to_keep[box_id]]
                box.set_edgecolor(edgecolor)
                box.set_ls(modify_box_config['ls'])
                box.set_hatch(modify_box_config['hatch'])
                lines = ax.lines[box_id * lines_per_boxplot: (box_id + 1) * lines_per_boxplot]
                for lid, lin in enumerate(lines):
                    # 0: lower whisker
                    # 1: upper whisker
                    # 2: lower foot
                    # 3: upper foot
                    # 4: median
                    # 5: grid line (don't modify)
                    if lid == 4:
                        lin.set_linestyle(modify_box_config['ls'])
                        lin.set_color(edgecolor)

        # Set the math notation for the y axis label
        if sci:
            mf = matplotlib.ticker.ScalarFormatter(useMathText=True)
            mf.set_powerlimits((3,3))
            ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
            ax.yaxis.set_major_formatter(mf)

        # Draw the reference y line
        if y_ref_config is not None:
            ax.axhline(y=y_ref_config['y'], ls=y_ref_config['ls'], lw=y_ref_config['lw'], color=y_ref_config['color'])

        ax.set_ylim(ylim)

    def box_all_grouped(
            self,
            ax,
            y_metric='dtw',
            ylim=(0,8_000),
            box_to_modify = [6, 8],
            to_keep=[],
            sci=True,
            modify_box_config={'color':'white', 'hatch':'////', 'ls':'-'},
            y_ref_config={'y':10_000, 'ls':'--', 'lw':1.5, 'color':'darkgray'},
            width=0.5,
            linewidth=1.5,
            ):
        
        palette=self.color_models_combined

        # Create a new derived column by combining cl and traj types
        self_results_cl_traj_df = self.results_df.copy()
        self_results_cl_traj_df['cl_traj_reg'] = self_results_cl_traj_df['cl_type'].astype(str) + '_' + \
            self_results_cl_traj_df['traj_type'].astype(str)    

        # Keep required combinations and discard the rest
        self_results_cl_traj_df = self_results_cl_traj_df.query('cl_traj_reg in @to_keep')

        ## Creating dummy rows for empty spaces
        # Copy the last row
        dummy_rows = list()
        for i in range(6):
            # Add the color of the dummy entry to the palette
            palette[f'empty_{i}'] = 'white'

            # Set the dummy entry
            row_template = self_results_cl_traj_df.iloc[0].copy()
            row_template['script_name'] = f'empty_{i}'
            row_template['cl_type'] = f'empty_{i}'
            row_template['traj_type'] = f'empty_{i}'
            row_template['stoch_reg_num'] = f'empty_{i}'
            row_template['cl_traj_reg'] = f'empty_{i}'
            # Set the numeric values of the dummy entry 
            # to a low value so that they do not show
            row_template['dtw'] = -1000.0  
            row_template['frechet'] = -1000.0
            row_template['swept'] = -1000.0
            row_template['quat_error'] = -1000.0

            # Append the modified row to the list
            dummy_rows.append(row_template)

        # Create a df from the dummy rows
        temp_df = pd.DataFrame(dummy_rows, columns=self_results_cl_traj_df.columns).reset_index()
        # Concatenate to the original df while ignoring the duplicate indices
        self_results_cl_traj_df = pd.concat([self_results_cl_traj_df, temp_df], ignore_index=True)

        # Create the main box plot
        sns.boxplot(ax=ax, 
                    data=self_results_cl_traj_df, 
                    x='cl_traj_reg', 
                    y=y_metric, 
                    fliersize=0, 
                    order=to_keep, 
                    palette=self.color_models_combined,
                    width=width,
                    linewidth=linewidth,
                    )

        # Find indexes of empty locators
        empty_idx = []
        for x in to_keep:
            if 'empty' in x:
                empty_idx.append(to_keep.index(x))

        # Find the boxplots
        box_container = ax.artists

        # How many lines in each boxplot
        lines_per_boxplot = len(ax.lines) // len(box_container)

        # Go through each boxplot and modify the ones necessary
        for box_id, box in enumerate(box_container):
            if box_id in box_to_modify:
                box.set_facecolor(modify_box_config['color'])
                edgecolor = self.color_models_combined[to_keep[box_id]]
                box.set_edgecolor(edgecolor)
                box.set_ls(modify_box_config['ls'])
                box.set_hatch(modify_box_config['hatch'])
                lines = ax.lines[box_id * lines_per_boxplot: (box_id + 1) * lines_per_boxplot]
                for lid, lin in enumerate(lines):
                    # 0: lower whisker
                    # 1: upper whisker
                    # 2: lower foot
                    # 3: upper foot
                    # 4: median
                    # 5: grid line (don't modify)
                    if lid == 4:
                        lin.set_linestyle(modify_box_config['ls'])
                        lin.set_color(edgecolor)


        # Set the math notation for the y axis label
        if sci:
            mf = matplotlib.ticker.ScalarFormatter(useMathText=True)
            mf.set_powerlimits((3,3))
            ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
            ax.yaxis.set_major_formatter(mf)

        # Draw the reference y line
        if y_ref_config is not None:
            ax.axhline(y=y_ref_config['y'], ls=y_ref_config['ls'], lw=y_ref_config['lw'], color=y_ref_config['color'])

        ax.set_ylim(ylim)

    def box_plots(self, 
                  ax, 
                  ylim,
                  y_metric='dtw',
                  cl_order=['SG', 'FT', 'REP', 'SI', 'MAS', 'HN', 'CHN'],
                  sci=True,
                  hue_order=['NODE', 'LSDDM']):


        sns.boxplot(data=self.results_df, 
                    x="cl_type", 
                    y=y_metric, 
                    hue="traj_type", 
                    ax=ax, fliersize=0, 
                    order=cl_order,
                    palette=self.color_models_traj,
                    notch=False,
                    medianprops={"color": "black"},
                    hue_order=hue_order)

        ax.set_ylim(ylim)
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.legend([],[], frameon=False)
        if sci:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    def box_plots_inter_cl(
            self, 
            ax, 
            ylim,
            traj_type,
            y_metric='dtw',
            cl_order=['SG', 'FT', 'REP', 'SI', 'MAS', 'HN', 'CHN'],
            sci=True,
            hue_order=['NODE', 'LSDDM']):


        sns.boxplot(data=self.results_df[self.results_df['traj_type']==traj_type], 
                    x="cl_type", 
                    y=y_metric, 
                    ax=ax, 
                    fliersize=0, 
                    order=cl_order,
                    palette=self.colors_models_node,
                    notch=False,
                    medianprops={"color": "black"},
                    hue_order=hue_order)

        ax.set_ylim(ylim)
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.legend([],[], frameon=False)
        if sci:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(3,3))

    def compute_model_sizes(self, seed, device):
        
        # Find dirs for each training script
        subdirs = glob(os.path.join(self.logbase_dir, "*", ""), recursive=False)

        size_list = list()
        for subdir in subdirs:
            if 'processed' not in subdir:
                globdir_node = os.path.join(subdir,f"*seed{seed}*", "models",f"node_{self.num_tasks-1}.pth")
                globdir_hnet = os.path.join(subdir,f"*seed{seed}*", "models",f"hnet_{self.num_tasks-1}.pth")

                # Path of the model
                if '_hn_' in subdir or '_chn_' in subdir:
                    models = glob(globdir_hnet)
                    assert len(models)>0, f'Model is empty for seed: {seed}, \n subdir:{subdir}, \n globdir_node:{globdir_node}, \n globdir_hnet:{globdir_hnet}'
                    model = models[0]
                else:
                    models = glob(globdir_node)
                    assert len(models)>0, f'Model is empty for seed: {seed}, \n subdir:{subdir}, \n globdir_node:{globdir_node}, \n globdir_hnet:{globdir_hnet}'
                    model = models[0]
                model_size = os.path.getsize(model)

                # Count number of total and trainable params
                model_loaded = torch.load(model, map_location=device)
                total_params = sum(p.numel() for p in model_loaded.parameters())
                train_params = sum(p.numel() for p in model_loaded.parameters() if p.requires_grad)

                # Find the parent directory, script name and cl and traj types
                parentdir = os.path.basename(os.path.normpath(subdir))
                script_name = parentdir[:parentdir.find(self.dataset)-1]
                cl_type = script_to_cl_map[script_name]
                traj_type = script_to_traj_map[script_name]

                # Calculate total size for SG
                if cl_type == 'SG':
                    model_size *= self.num_tasks
                    total_params *= self.num_tasks
                    train_params *= self.num_tasks

                size_list.append(dict(script_name=script_name,
                                      cl_type=cl_type,
                                      traj_type=traj_type,
                                      model_size=model_size,
                                      total_params=total_params,
                                      train_params=train_params))
                
        self.size_df = pd.DataFrame(size_list)

        if len(subdirs) == 0:
            raise Exception(f'Globbing {self.logbase_dir} unsuccessful!')

    def plot_model_sizes(self, 
                         fig, ax1, ax2,
                         cl_type_order, 
                         hspace=0.08, 
                         diag=0.01, 
                         upper_ylim=(5e6,20e6), 
                         lower_ylim=(1e6, 3e6),
                         fontsizes=dict(label=12,title=16,legend=12),
                         gap=0.5):
        # Adapted from: https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/broken_axis.html

        @ticker.FuncFormatter
        def million_formatter(x, pos):
            return "%.1f" % (x/1E6)

        # Plot the upper part
        bars1 = sns.barplot(ax=ax1, data=self.size_df, x='cl_type', y='total_params', gap=gap,
                    hue='traj_type', palette=self.color_models_traj, order=cl_type_order)

        # Plot the lower part
        bars2 = sns.barplot(ax=ax2, data=self.size_df, x='cl_type', y='total_params', gap=gap,
                    hue='traj_type', palette=self.color_models_traj, order=cl_type_order)

        # zoom-in / limit the view to different portions of the data
        ax1.set_ylim(upper_ylim)  # outliers only
        ax2.set_ylim(lower_ylim)  # most of the data

        # hide the spines between ax and ax2
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False)  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()

        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((-diag, +diag), (-diag, +diag), **kwargs)        # top-left diagonal
        ax1.plot((1 - diag, 1 + diag), (-diag, +diag), **kwargs)  # top-right diagonal

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-diag, +diag), (1 - diag, 1 + diag), **kwargs)  # bottom-left diagonal
        ax2.plot((1 - diag, 1 + diag), (1 - diag, 1 + diag), **kwargs)  # bottom-right diagonal

        # Remove legend from bottom part and remove axis labels
        ax2.get_legend().remove()
        ax1.set_ylabel('')
        ax2.set_ylabel('')
        ax1.set_xlabel('')
        ax2.set_xlabel('')
        # Recreate legend
        ax1.legend(loc='upper right', fontsize=fontsizes['legend'])

        # Set formatter for y axis
        ax1.yaxis.set_major_formatter(million_formatter)
        ax2.yaxis.set_major_formatter(million_formatter)

        # Set grid lines
        ax1.grid(True)
        ax2.grid(True)
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)

        for a in [ax1, ax2]:
            for t in a.get_xticklabels():
                t.set_fontsize(fontsizes['ticklabel'])
            for t in a.get_yticklabels():
                t.set_fontsize(fontsizes['ticklabel'])

        # Adjust spacing and set axis labels for both parts of plot
        fig.subplots_adjust(hspace=hspace)
        fig.supylabel('Parameters ('+r'$\times 10^6$'+')', fontsize=fontsizes['label'])
        fig.supxlabel('CL Types', fontsize=fontsizes['label'])

        return bars1, bars2

