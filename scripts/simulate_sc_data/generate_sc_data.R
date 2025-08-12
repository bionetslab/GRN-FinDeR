library(decoupleR)
library(scMultiSim)
library(data.table)
library(ggplot2)
library(dplyr)


#' @title Subsample and Sparsify a Gene Regulatory Network
#'
#' @description
#' This function takes a gene regulatory network and subsamples it to a
#' specified number of source and target nodes. It then sparsifies the
#' network by limiting the number of outgoing connections for each source node.
#'
#' @param network A data frame or data.table representing the network.
#' @param num_sources The desired number of unique source nodes.
#' @param max_out_degree The maximum number of outgoing connections per source.
#'
#' @return A sparsified and subsampled network data.table.
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

  return(sparsified_subnetwork)
}


create_datasets<-function(collectri_net, outpath, num_sources = 100, max_out_degree=20, n_datasets = 1){
  # Number of datasets to simulate
  n_datasets <- 10
  # Simulation parameters for each dataset
  num_sources <- 100
  max_out_degree <- 20 # Maximum outgoing connections per source node



  for(i in 1:n_datasets){
    net_sub <- subsample_and_sparsify_network(
      network = collectri_net,
      num_sources = num_sources,
      max_out_degree = max_out_degree
    )

    net_sub$mor<-5

    network_folder<-file.path(outpath, 'nets')
    if(!dir.exists(network_folder)){
      dir.create(network_folder, recursive = T)
    }
    network_file<- file.path(network_folder, paste0('network_', i, '.tsv' ))

    # save network into file
    fwrite(net_sub, network_file, sep = '\t')

    # take true simulated counts, here this should be sufficient.
    results <- sim_true_counts(list(
      # required options
      GRN = net_sub,
      tree = Phyla1(),
      num.cells = 1000,
      # optional options
      num.cif = 50,
      discrete.cif = T, # one discrete population
      cif.sigma = 0.1,
      speed.up = T
    ))

    gex_data<-as.data.table(results$counts)
    gex_data$gene<-rownames(results$counts)
    gex_data<-gex_data[, c(1001, 1:1000)]

    data_folder<-file.path(outpath, 'data')
    if(!dir.exists(data_folder)){
      dir.create(data_folder, recursive = T)
    }
    data_file<- file.path(data_folder, paste0('data_', i, '.tsv' ))

    fwrite(gex_data, data_file, sep = '\t', row.names = T)


    # save plot verifying there is one single cluster
    plot_file<- file.path(data_folder, paste0('data_', i, '_plot.pdf' ))
    plot_data<-plot_tsne(results$counts, results$cell_meta$pop)+theme_bw()+xlab('TSNE1')+ylab('TSNE2')
    ggsave(plot_data, file = plot_file, height = 15, width = 16, units = 'cm')
  }
}


collectri_net <- decoupleR::get_collectri()
create_datasets(collectri_net, 'Documents/GRN-FinDeR/data/sc_simulated_data', n_datasets = 10)
