library(decoupleR)
library(scMultiSim)
library(data.table)
library(ggplot2)
library(dplyr)



subsample_and_sparsify_network <- function(network, num_sources, max_out_degree) {
  # Convert to data.table for efficient manipulation
  network_dt <- as.data.table(network)

  # Get unique source and target nodes
  all_sources <- unique(network_dt$source)
  all_targets <- unique(network_dt$target)

  # Check if there are enough sources and targets to sample from
  if (length(all_sources) < num_sources) {
    stop("Not enough unique source nodes in the network to sample from.")
  }


  # Randomly select a subset of sources and targets
  sampled_sources <- sample(all_sources, num_sources, replace = FALSE)

  # Filter the network to keep only the sampled sources and targets
  subnetwork <- network_dt[source %in% sampled_sources]

  # Sparsify the subnetwork by limiting outgoing connections
  # We group by the source and keep at most `max_out_degree` connections.

    sparsified_subnetwork <- subnetwork %>%
    group_by(source) %>%
    slice_head(n = max_out_degree) %>%
    ungroup() %>%
    as.data.table()

    sparsified_subnetwork<-sparsified_subnetwork[!(source %in% sparsified_subnetwork$target)]

  return(sparsified_subnetwork)
}

create_datasets<-function(collectri_net, outpath, num_sources = 5, max_out_degree=20, n_datasets = 10, n_cells = 1000){

  for(i in 1:n_datasets){
    net_sub <- subsample_and_sparsify_network(
      network = collectri_net,
      num_sources = num_sources,
      max_out_degree = max_out_degree
    )

    net_sub$mor<-rnorm(nrow(net_sub), mean=5, sd = 1)
    net_sub<-as.data.table(net_sub)
    network_folder<-file.path(outpath, 'nets')
    if(!dir.exists(network_folder)){
      dir.create(network_folder, recursive = T)
    }

    network_file<- file.path(network_folder, paste0('network_', i, '.tsv' ))
    print(network_file)
    # save network into file
    fwrite(net_sub, network_file, sep = '\t')

    # take true simulated counts, here this should be sufficient.
    results <- sim_true_counts(list(
      # required options
      GRN = net_sub,
      tree = Phyla1(), # 1 cluster
      num.cells = n_cells,
      # optional options
      num.cif = 40, 
      discrete.cif = T, # one discrete population
      cif.sigma = 0.25, 
      speed.up = T,
      diff.cif.fraction=0.05, #Dial up GRN effect as far as possible
      unregulated.gene.ratio = 0.05, # Dial down number of random genes as far as possible
      do.velocity=FALSE,
      intrinsic.noise = 1.0 # add some noise
    ))
    plot_gene_module_cor_heatmap(results)


    gex_data <-as.data.frame(results$counts)
    gex_data$gene<-rownames(results$counts)
    gex_data<-gex_data[, c(n_cells+1, 1:n_cells)]

    data_folder<-file.path(outpath, 'data')
    if(!dir.exists(data_folder)){
      dir.create(data_folder, recursive = T)
    }
    data_file<- file.path(data_folder, paste0('data_', i, '.tsv' ))

    fwrite(gex_data, file=data_file, sep = '\t', row.names = F)


    # save plot verifying there is one single cluster
    plot_file<- file.path(data_folder, paste0('data_', i, '_plot.pdf' ))
    plot_data<-plot_tsne(results$counts, results$cell_meta$pop)+theme_bw()+xlab('TSNE1')+ylab('TSNE2')
    ggsave(plot_data, file = plot_file, height = 15, width = 16, units = 'cm')

    # save plot verifying there is one single cluster

    #plot_file<- file.path(data_folder, paste0('data_', i, '_heatmap.pdf' ))
    #ggsave(data_plot, file = plot_file, height = 15, width = 16, units = 'cm')

  }
  return(results)
}

collectri<-decoupleR::get_collectri()


dataset <- create_datasets(collectri, '/home/bionets-og86asub/Documents/GRN-FinDeR/data/sc_simulated_data/5_sources', num_sources = 5, max_out_degree = 50, n_datasets = 10, n_cells = 500)
dataset <- create_datasets(collectri, '/home/bionets-og86asub/Documents/GRN-FinDeR/data/sc_simulated_data/10_sources', num_sources = 10, max_out_degree = 40, n_datasets = 10, n_cells = 500)
dataset <- create_datasets(collectri, '/home/bionets-og86asub/Documents/GRN-FinDeR/data/sc_simulated_data/20_sources', num_sources = 20, max_out_degree = 20, n_datasets = 10, n_cells = 500)
