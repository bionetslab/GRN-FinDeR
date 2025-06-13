import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

def boxplot_f1(df):
    # Plot with seaborn
    df = df[~df['num_non_tfs'].between(2, 9)]
    
    custom_palette = ['#FA8072',  # salmon-like
                      '#C7E8A9']  # creamy light green
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='num_non_tfs', y='F1_score', hue='F1_type', palette=custom_palette)

    # Tweak labels and title
    #plt.title('F1 scores over 17 tissues for varying number of non-TF clusters')
    plt.xlabel('Number of non-TF clusters')
    plt.ylabel('F1 Score')
    plt.legend(title='P-value Threshold')
    plt.xticks(rotation=45)  # if you have many x-axis values
    plt.legend(loc='center right')
    plt.tight_layout()

    plt.savefig('boxplot_nonTFs_medoid_f1.pdf')
    plt.show()

def lineplot_mae(df):
    plt.figure(figsize=(12, 6))

    # Lineplot for each tissue's MAE progression
    sns.lineplot(
        data=df,
        x='num_non_tfs',
        y='mae',
        hue='num_samples',
        hue_norm=(0, 500),
        estimator=None,  # important: plot actual values, not aggregated means
        markers=True
    )

    #plt.title('P-value errors on 17 tissues for varying numbers of non-TF clusters')
    plt.xlabel('Number of non-TF clusters')
    plt.ylabel('MAE')
    plt.xticks(rotation=45)
    plt.legend(title='Samples per tissue', bbox_to_anchor=(0.92, 0.6))#, loc='center right')
    plt.tight_layout()
    plt.savefig('lineplot_mae.pdf')
    plt.show()

def lineplot_total_runtime(df):

    df['total_runtime'] /= 3600.0
    plt.figure(figsize=(12, 6))

    # Lineplot for each tissue's MAE progression
    sns.lineplot(
        data=df,
        x='num_non_tfs',
        y='total_runtime',
        hue='num_samples',
        estimator=None,  # important: plot actual values, not aggregated means
        markers=True
    )

    #plt.title('Absolute saved runtime over 17 tissues for varying numbers of non-TF clusters')
    plt.xlabel('Number of non-TF clusters')
    plt.ylabel('Total runtime [hours]')
    plt.xticks(rotation=45)
    plt.legend(title='Samples per tissue') #bbox_to_anchor=(0.92, 0.6))#, loc='upper left')
    plt.tight_layout()
    plt.savefig('lineplot_total_runtime.pdf')
    plt.show()

def lineplot_time(df):

    df['abs_time_saving'] /= 3600.0
    plt.figure(figsize=(12, 6))

    # Lineplot for each tissue's MAE progression
    sns.lineplot(
        data=df,
        x='num_non_tfs',
        y='abs_time_saving',
        hue='num_samples',
        estimator=None,  # important: plot actual values, not aggregated means
        markers=True
    )

    #plt.title('Absolute saved runtime over 17 tissues for varying numbers of non-TF clusters')
    plt.xlabel('Number of non-TF clusters')
    plt.ylabel('Saved runtime [hours]')
    plt.xticks(rotation=45)
    plt.legend(title='Samples per tissue') #bbox_to_anchor=(0.92, 0.6))#, loc='upper left')
    plt.tight_layout()
    plt.savefig('lineplot_abs_time.pdf')
    plt.show()

def lineplot_rel_time(df):

    plt.figure(figsize=(12, 6))

    # Lineplot for each tissue's MAE progression
    sns.lineplot(
        data=df,
        x='num_non_tfs',
        y='rel_time_saving',
        hue='num_samples',
        estimator=None,  # important: plot actual values, not aggregated means
        markers=True
    )

    #plt.title('Relative saved runtime for 17 tissues for varying numbers of non-TF clusters')
    plt.xlabel('Number of non-TF clusters')
    plt.ylabel('Saved runtime factor')
    plt.xticks(rotation=45)
    plt.legend(title='Samples per tissue')#, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('lineplot_rel_time.pdf')
    plt.show()
    
def lineplot_silhouettes_TFs(df, subset_tissues):
    df = df[df['gene_type'] == 'tf']
    df = df[df['tissue'].isin(subset_tissues)]
    # Plot
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x='num_clusters',
        y='avg_silhouette',
        hue='num_samples',
        estimator=None,  # important: plot actual values, not aggregated means
        markers=True
    )

    #plt.title('Relative saved runtime for 17 tissues for varying numbers of non-TF clusters')
    plt.xlabel('Number of TF clusters')
    plt.ylabel('Averaged Silhouette Score')
    plt.xticks(rotation=45)
    plt.legend(title='Samples per tissue')#, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('lineplot_silhouettes_TFs.pdf')
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv('approximate_fdr_grns_medoid_results.csv')
    # Assuming your dataframe is called df
    # Melt the dataframe to long format for F1 scores
    df = df.rename(columns={'f1_005': 'alpha=0.05', 'f1_001': 'alpha=0.01'})
    df_melted = df.melt(
        id_vars=['num_non_tfs'],
        value_vars=['alpha=0.05', 'alpha=0.01'],
        var_name='F1_type',
        value_name='F1_score'
    )
    
    # Load sample sizes dictionary.
    with open('samples_per_tissue.pkl', 'rb') as f:
        samples = pickle.load(f)
    
    silhouettes_df = pd.read_csv('silhouettes_all_tissues.tsv', sep='\t', index_col=0)
    silhouettes_tissues = silhouettes_df['tissue']
    samples_list = [samples[tissue] for tissue in silhouettes_tissues]
    silhouettes_df['num_samples'] = samples_list
    print(silhouettes_df)
        
    # Append tissue sizes to DF.
    tissues_list = df['tissue']
    samples_list = [samples[tissue] for tissue in tissues_list]
    df['num_samples'] = samples_list
    subset_tissues = set(df['tissue'])

    #boxplot_f1(df_melted)
    #lineplot_rel_time(df)
    #lineplot_total_runtime(df)
    lineplot_silhouettes_TFs(silhouettes_df, subset_tissues)