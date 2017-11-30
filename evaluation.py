# -*- coding: utf-8 -*-
import util
from sklearn import svm
from WLVectorizer import WLVectorizer
from sklearn import metrics
import sys

def evaluate(adjacency_path=None, node_label_folder=None, all_gene_path=None, train_gene_folder=None, train_label_folder=None,
             n_iters=None, n_hops=None, n_clusters=None, svm_paras=None, save_folder=None):
    
    all_genes = util.load_list_from_file(all_gene_path)
    number_svm_parameters = len(svm_paras)

    dict_gene_idx = {}    
    for idx, gene in enumerate(all_genes):
        dict_gene_idx[gene] = idx
    
    graph = util.create_graph(adjacency_path=adjacency_path)
    
    for n_cluster in n_clusters:
        util.node_labeling(g=graph, label_path=node_label_folder+str(n_cluster))
        for n_iter in n_iters:
            
            WLvect= WLVectorizer(r=n_iter)
            iters_features = WLvect.transform([graph])
            M = iters_features[0][0]
            for iter_id in range(1, n_iter + 1):
                M = M + iters_features[iter_id][0]
            print 'Done WL compuation'
            sys.stdout.flush()
            
            for n_hop in n_hops:
                print 'Begining DWL compuation'
                sys.stdout.flush()
                G = util.deepwl(graph=graph, feature_matrix=M, n_hop=n_hop)
                print "Size of G", G.shape
                
                print 'Done DWL compuation'
                sys.stdout.flush()   

                for disease_idx in range(12):
                    list_training_genes = util.load_list_from_file(train_gene_folder + str(disease_idx))
                    list_training_labels = util.load_list_from_file(train_label_folder + str(disease_idx))
                    list_training_labels = [int(e) for e in list_training_labels]
                    list_qscores = [[] for i in range(number_svm_parameters)] 
                
                    for gene_idx, gene in enumerate(list_training_genes):    
                        list_training_genes_del = list_training_genes[:]
                        del list_training_genes_del[gene_idx]
                        training_genes_idx = [dict_gene_idx[g] for g in list_training_genes_del]
                        
                        list_training_labels_del = list_training_labels[:]
                        del list_training_labels_del[gene_idx]
                        
                        
                        unknown_genes_idx = [dict_gene_idx[gene]]
                        for idx in range(len(all_genes)):
                            if (idx not in training_genes_idx) and (idx != dict_gene_idx[gene]):
                                unknown_genes_idx.append(idx)        
                                
                        Mtr = util.extract_submatrix(training_genes_idx,training_genes_idx,G)
                        M_unknown = util.extract_submatrix(unknown_genes_idx,training_genes_idx,G)
                        
                        for idx_svm, svm_para in enumerate(svm_paras):
                            clf = svm.SVC(C = svm_para, kernel='precomputed')
                            clf.fit(Mtr, list_training_labels_del)
                            scores = clf.decision_function(M_unknown)
                            len_scores = len(scores)
                            qscore = float(sum([int(scores[0] > val) for val in scores]))/len_scores
                            list_qscores[idx_svm].append(qscore)
                    # computing auc
                    save_lines = []
                    for qscores_idx, qscores in enumerate(list_qscores):
                        fpr, tpr, thresholds = metrics.roc_curve(list_training_labels, qscores, pos_label= 1)
                        auc = metrics.auc(fpr, tpr)
                        
                        line = str(n_cluster) + "_" + str(n_iter) + "_" + str(n_hop) + "_" + str(qscores_idx) + ":\t" + str(auc) + "\n"
                        save_lines.append(line)
    
                    f = open(save_folder + str(disease_idx),'w')
                    f.writelines(save_lines)
                    f.close()