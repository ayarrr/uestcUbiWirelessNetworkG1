from sklearn.cluster import AgglomerativeClustering
import numpy as np
from utility import construct_doc_matrix


class Evaluator():
    def __init__(self,label):
        self.label=label
    def compute_f1(self,dataset,latent_matrix):
        """
        perform Hierarchy Clustering on doc embedding matrix
        for name disambiguation
        use cluster-level mean F1 for evaluation
        """
        D_matrix = construct_doc_matrix(latent_matrix,dataset.paper_list)
        true_cluster_size =self.label.n_clusfer
        #'ward', 'average', 'complete'
        y_pred = AgglomerativeClustering(n_clusters = true_cluster_size,
                                         linkage = "average",
                                         affinity = "cosine").fit_predict(D_matrix)

        true_label_dict = self.label.detail
        predict_label_dict = {}
        for idx, pred_lbl in enumerate(y_pred):
            pid=dataset.paper_list[idx]
            if pred_lbl not in predict_label_dict:
                predict_label_dict[pred_lbl] = [pid]
            else:
                predict_label_dict[pred_lbl].append(pid)

        # compute cluster-level F1
        # let's denote C(r) as clustering result and T(k) as partition (ground-truth)
        # construct r * k contingency table for clustering purpose 
        r_k_table = []
        for v1 in predict_label_dict.values():
            k_list = []
            for v2 in true_label_dict:
                N_ij = len(set(v1).intersection(v2))
                k_list.append(N_ij)
            r_k_table.append(k_list)
        r_k_matrix = np.array(r_k_table)
        r_num = int(r_k_matrix.shape[0])

        # compute F1 for each row C_i
        sum_f1 = 0.0
        for row in range(r_num):
            row_sum = np.sum(r_k_matrix[row,:])
            if row_sum != 0:
                max_col_index = np.argmax(r_k_matrix[row,:])
                row_max_value = r_k_matrix[row, max_col_index]
                prec = float(row_max_value) / row_sum
                col_sum = np.sum(r_k_matrix[:, max_col_index])
                rec = float(row_max_value) / col_sum
                row_f1 = float(2 * prec * rec) / (prec + rec)
                sum_f1 += row_f1

        average_f1 = float(sum_f1) / r_num
        return average_f1
