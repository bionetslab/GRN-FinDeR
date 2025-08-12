require(ggplot2)
require(data.table)

data.1<-fread('Documents/GRN-FinDeR/results/gtex_up_to_breast/approximate_fdr_grns_random_targets_breast_kidney_testis.csv')
ggplot(data.1[tissue %in% c('Breast', 'Kidney', 'Testis')], aes(x = num_non_tfs, y=f1_001, col=tissue))+geom_line()+scale_x_log10()+theme_bw()

data.1<-data.1[tissue %in% c('Breast', 'Kidney', 'Testis')]
data.1$trial<-'Wasserstein/Random'
data.1$clustering <-'Wasserstein'
data.1$representative_selection<-'Random'


data.2<-fread('Documents/GRN-FinDeR/results/gtex_up_to_breast/approximate_fdr_grns_random_kmeans_targets_breast_kidney_testis.csv')
data.2$trial<-'K-medoid/Random'
data.2$clustering <-'K-Medoid'
data.2$representative_selection<-'Random'

data.3<-fread('Documents/GRN-FinDeR/results/gtex_up_to_breast/approximate_fdr_grns_medoid_kmeans_targets_breast_kidney_testis.csv')
data.3$trial<-'K-Medoid/K-Medoid'
data.3$clustering <-'K-Medoid'
data.3$representative_selection<-'K-Medoid'

data.4<-fread('Documents/GRN-FinDeR/results/gtex_up_to_breast/approximate_fdr_grns_medoid_targets_breast_kidney_testis.csv')
data.4$trial<-'Wasserstein/K-Medoid'
data.4$clustering <-'Wasserstein'
data.4$representative_selection<-'K-Medoid'

data<-rbind(data.2, data.1)


f_005<-ggplot(data, aes(x = num_non_tfs, y=f1_005, col=tissue))+ geom_line(aes(linetype=trial))+
  theme_bw()+
  scale_linetype_manual(name = 'Trial', values = c('Wasserstein/Random' = "solid", 'K-medoid/Random' = "dashed", rep("dotted", 4)))+
  theme(axis.title = element_text(size=20), legend.text = element_text(size=15), legend.position = 'right', legend.box = 'vertical', legend.title = element_text(size=15),
        legend.justification = "right",
        axis.text  = element_text(size=10),
        axis.line = element_line())+
  ylab('F1 Score (p=0.05)')+
  xlab('Number of target clusters')+
  scale_x_continuous(
    trans = 'log10',
    breaks = c(10, 100, 1000, 2000, 10000), # Add 2000 here
    guide = guide_axis_logticks()
  ) +
  labs(
    color = "Tissue", # Changes the title for the 'color' legend
  )

ggsave(f_005, filename = 'Documents/GRN-FinDeR/results/gtex_up_to_breast/f1_005_best_performer.pdf', height = 15, width = 25, units = 'cm')

f1_001<-ggplot(data, aes(x = num_non_tfs, y=f1_001, col=tissue))+ geom_line(aes(linetype=trial))+
  ylab('F1 Score (p=0.01)')+xlab('Number of target clusters')+
  theme_bw()+
  scale_linetype_manual(name = 'Trial', values = c('Wasserstein/Random' = "solid", 'K-medoid/Random' = "dashed", rep("dotted", 4)))+
  theme(axis.title = element_text(size=20), legend.text = element_text(size=15), legend.position = 'right', legend.box = 'vertical', legend.title = element_text(size=15),
        legend.justification = "right",
        axis.text  = element_text(size=10),
        axis.line = element_line())+
  scale_x_continuous(
    trans = 'log10',
    breaks = c(10, 100, 1000, 2000, 10000), # Add 2000 here
    guide = guide_axis_logticks()
  ) +
  labs(
    color = "Tissue", # Changes the title for the 'color' legend
  )
ggsave(f1_001, filename = 'Documents/GRN-FinDeR/results/gtex_up_to_breast/f1_001_best_performer.pdf', height = 15, width = 25, units = 'cm')



mae<-ggplot(data, aes(x = num_non_tfs, y=mae, col=tissue))+ geom_line(aes(linetype=trial))+
  ylab('Mean Absolute Error')+xlab('Number of target clusters')+
  theme_bw()+
  scale_linetype_manual(name = 'Trial', values = c('Wasserstein/Random' = "solid", 'K-medoid/Random' = "dashed", rep("dotted", 4)))+
  theme(axis.title = element_text(size=20), legend.text = element_text(size=15), legend.position = 'right', legend.box = 'vertical', legend.title = element_text(size=15),
        legend.justification = "right",
        axis.text  = element_text(size=10),
        axis.line = element_line())+
  scale_x_continuous(
    trans = 'log10',
    breaks = c(10, 100, 1000, 2000, 10000), # Add 2000 here
    guide = guide_axis_logticks()
  ) +
  labs(
    color = "Tissue", # Changes the title for the 'color' legend
  )

ggsave(mae, filename = 'Documents/GRN-FinDeR/results/gtex_up_to_breast/mae_best_performer.pdf', height = 15, width = 25, units = 'cm')


timesav<-ggplot(data, aes(x = num_non_tfs, y=abs_time_saving/3600, col=tissue))+ geom_line(aes(linetype=trial))+
  ylab('Runtime saved [hours]')+xlab('Number of target clusters')+
  theme_bw()+
  scale_linetype_manual(name = 'Trial', values = c('Wasserstein/Random' = "solid", 'K-medoid/Random' = "dashed", rep("dotted", 4)))+
  theme(axis.title = element_text(size=20), legend.text = element_text(size=15), legend.position = 'right', legend.box = 'vertical', legend.title = element_text(size=15),
        legend.justification = "right",
        axis.text  = element_text(size=10),
        axis.line = element_line())+
  scale_x_continuous(
    trans = 'log10',
    breaks = c(10, 100, 1000, 2000, 10000), # Add 2000 here
    guide = guide_axis_logticks()
  ) +
  labs(
    color = "Tissue", # Changes the title for the 'color' legend
  )

ggsave(timesav, filename = 'Documents/GRN-FinDeR/results/gtex_up_to_breast/time_saved_best_performer.pdf', height = 15, width = 25, units = 'cm')

runtime<-ggplot(data, aes(x = num_non_tfs, y=total_runtime/3600, col=tissue))+ geom_line(aes(linetype=trial))+
  ylab('Runtime [hours]')+xlab('Number of target clusters')+
  theme_bw()+
  scale_linetype_manual(name = 'Trial', values = c('Wasserstein/Random' = "solid", 'K-medoid/Random' = "dashed", rep("dotted", 4)))+
  theme(axis.title = element_text(size=20), legend.text = element_text(size=15), legend.position = 'right', legend.box = 'vertical', legend.title = element_text(size=15),
        legend.justification = "right",
        axis.text  = element_text(size=10),
        axis.line = element_line())+
  scale_x_continuous(
    trans = 'log10',
    breaks = c(10, 100, 1000, 2000, 10000), # Add 2000 here
    guide = guide_axis_logticks()
  ) +
  labs(
    color = "Tissue", # Changes the title for the 'color' legend
  )
ggsave(runtime, filename = 'Documents/GRN-FinDeR/results/gtex_up_to_breast/runtime_best_performer.pdf', height = 15, width = 25, units = 'cm')


ggplot(data, aes(x = num_non_tfs, y=abs_emission_saving, col=tissue))+ geom_line(aes(linetype=trial))+
  ylab('Emissions saved gCO2')+xlab('Number of target clusters')+
  theme_bw()+
  scale_linetype_manual(name = 'Trial', values = c('Wasserstein/Random' = "solid", 'K-medoid/Random' = "dashed", rep("dotted", 4)))+
  theme(axis.title = element_text(size=20), legend.text = element_text(size=15), legend.position = 'right', legend.box = 'vertical', legend.title = element_text(size=15),
        legend.justification = "right",
        axis.text  = element_text(size=10),
        axis.line = element_line())+
  scale_x_continuous(
    trans = 'log10',
    breaks = c(10, 100, 1000, 2000, 10000), # Add 2000 here
    guide = guide_axis_logticks()
  ) +
  labs(
    color = "Tissue", # Changes the title for the 'color' legend
  )



ggplot(data, aes(x = num_non_tfs, y=rel_time_saving, col=tissue))+ geom_line(aes(linetype=trial))+scale_x_log10()+theme_bw()+ylab('Runtime factor')+xlab('Number of target clusters')

datalist<-list()
for (i in 0:9){
  data.1<-fread(paste0('Documents/GRN-FinDeR/results/gtex_up_to_breast/approximate_fdr_grns_random_separate_wasserstein_targets_gt',i,'_breast_kidney_testis.csv'))
  data.1$gt<-i
  datalist[[i+1]]<-data.1
}
data<-rbindlist(datalist)


ggplot(data[tissue %in% c('Breast', 'Testis', 'Kidney')], aes(x = num_non_tfs, y=f1_005, col=tissue))+geom_line(aes(linetype=as.factor(gt)))+scale_x_log10()+theme_bw()


data<-rbind(data.1, data.2, data.3, data.4)
ggplot(data, aes(x = num_non_tfs, y=mae, col=tissue))+ geom_line(aes(linetype=clustering))+ facet_wrap(~representative_selection)+
  ylab('Mean Absolute Error')+xlab('Number of target clusters')+
  theme_bw()+
  scale_linetype_manual(name = 'Trial', values = c('Wasserstein' = "solid", 'K-Medoid' = "dashed", rep("dotted", 4)))+
  theme(axis.title = element_text(size=20), legend.text = element_text(size=15), legend.position = 'right', legend.box = 'vertical', legend.title = element_text(size=15),
        legend.justification = "right",
        axis.text  = element_text(size=10),
        axis.line = element_line())+
  scale_x_continuous(
    trans = 'log10',
    breaks = c(10, 100, 1000, 2000, 10000), # Add 2000 here
    guide = guide_axis_logticks()
  ) +
  labs(
    color = "Tissue", # Changes the title for the 'color' legend
  )

ggplot(data, aes(x = num_non_tfs, y=f1_001, col=tissue))+ geom_line(aes(linetype=clustering))+ facet_wrap(~representative_selection)+
  ylab('Mean Absolute Error')+xlab('Number of target clusters')+
  theme_bw()+
  scale_linetype_manual(name = 'Trial', values = c('Wasserstein' = "solid", 'K-Medoid' = "dashed", rep("dotted", 4)))+
  theme(axis.title = element_text(size=20), legend.text = element_text(size=15), legend.position = 'right', legend.box = 'vertical', legend.title = element_text(size=15),
        legend.justification = "right",
        axis.text  = element_text(size=10),
        axis.line = element_line())+
  scale_x_continuous(
    trans = 'log10',
    breaks = c(10, 100, 1000, 2000, 10000), # Add 2000 here
    guide = guide_axis_logticks()
  ) +
  labs(
    color = "Tissue", # Changes the title for the 'color' legend
  )

ggplot(data, aes(x = num_non_tfs, y=f1_005, col=tissue))+ geom_line(aes(linetype=clustering))+ facet_wrap(~representative_selection)+
  ylab('Mean Absolute Error')+xlab('Number of target clusters')+
  theme_bw()+
  scale_linetype_manual(name = 'Trial', values = c('Wasserstein' = "solid", 'K-Medoid' = "dashed", rep("dotted", 4)))+
  theme(axis.title = element_text(size=20), legend.text = element_text(size=15), legend.position = 'right', legend.box = 'vertical', legend.title = element_text(size=15),
        legend.justification = "right",
        axis.text  = element_text(size=10),
        axis.line = element_line())+
  scale_x_continuous(
    trans = 'log10',
    breaks = c(10, 100, 1000, 2000, 10000), # Add 2000 here
    guide = guide_axis_logticks()
  ) +
  labs(
    color = "Tissue", # Changes the title for the 'color' legend
  )

ggplot(data, aes(x = num_non_tfs, y=total_runtime/3600, col=tissue))+ geom_line(aes(linetype=clustering))+ facet_wrap(~representative_selection)+
  ylab('Mean Absolute Error')+xlab('Number of target clusters')+
  theme_bw()+
  scale_linetype_manual(name = 'Trial', values = c('Wasserstein' = "solid", 'K-Medoid' = "dashed", rep("dotted", 4)))+
  theme(axis.title = element_text(size=20), legend.text = element_text(size=15), legend.position = 'right', legend.box = 'vertical', legend.title = element_text(size=15),
        legend.justification = "right",
        axis.text  = element_text(size=10, angle = 45),
        axis.line = element_line())+
  scale_x_continuous(
    trans = 'log10',
    breaks = c(10, 100, 1000, 2000), # Add 2000 here
    guide = guide_axis_logticks()
  ) +
  labs(
    color = "Tissue", # Changes the title for the 'color' legend
  )

