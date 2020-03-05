# Author: Peter Tsung

def forward_algorithm(O, HMM_model):
    """
    HMM Forward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O) 
    N = len(pi)
    
    alpha_prob = np.zeros((N, T))
    
    # First Step: Initial alpha_1(i), i = 1,2,...,N
    for i in range(N):
        alpha_prob[i][0] = pi[i] * B[i][O[0]]
        
    # Second Step: calculatee alpha_t+1
    for t in range(1, T):
        for i in range(N):
            sigma_forward = sum(alpha_prob[:, t - 1] * np.array(A)[:, i])
            alpha_prob[i][t] = sigma_forward * B[i][O[t]]
    
    print("=" * 10 + "ALPHA_PROB_MATRIX" + "="*10)
    print(alpha_prob)
    print("=" * 10 + "End" + "="*10) 
    
    # Final Step: Terminate
    return sum(alpha_prob[:, T-1])

def backward_algorithm(O, HMM_model):
    """
    HMM Backward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    
    beta_prob = np.zeros((N, T))
    # First Step: Initial beta_T(i), i = 1,2,...,N
    for i in range(N):
        beta_prob[i][T-1] = 1

    # Second Step: Calculate beta_t(i)
    for t in reversed(range(T - 1)):
        for i in range(N):
            beta_prob[i][t] = sum(np.array(A)[i, :] * np.array(B)[:, observations[t+1]] * beta_prob[:, t+1])
    
    print("=" * 10 + "BETA_PROB_MATRIX" + "="*10)
    print(beta_prob)
    print("=" * 10 + "End" + "="*10) 
    
    # Final Step: Terminate
    prob = sum(pi * np.array(B)[:, observations[0]] * beta_prob[:, 0])
    return prob
    
def Viterbi_algorithm(O, HMM_model):
    """
    Viterbi decoding.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state pro, transition prob, emitting prob)
    Returns:
        best_prob: the probability of the best state sequence
        best_path: the best state sequence
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    best_prob, best_path = 0.0, []
    
    # First Step: Initial delta_mat and psi_mat
    delta_mat = np.zeros((N, T))
    psi_mat = np.zeros((N, T + 1))

    for i in range(N):
        delta_mat[i][0] = pi[i] * B[i][observations[0]]

    # Second Step: Calculate delta_mat from t = 2,3,..., T 
    # And psi_mat from t = 2,3,...,T
    for t in range(1, T):
        for i in range(N):
            single_path_prob_lst = delta_mat[:, t-1] * np.array(A)[:, i]
            delta_mat[i][t] = max(single_path_prob_lst) * B[i][observations[t]]
            psi_mat[i][t] = np.argmax(single_path_prob_lst) + 1

    # Third Step: Calculate in time T, the max probability of single path
    # And the best final node.
    best_path_prob = max(delta_mat[:, -1])
    best_final_node = np.argmax(delta_mat[:, -1]) + 1
    psi_mat[:, -1][best_final_node - 1] = best_final_node
    best_path_mat = np.zeros(T)
    best_path_mat[-1] = best_final_node

    # Final Step: Back tracking the node among T-1, T-2, ..., 1
    for t in reversed(range(T - 1)):
        back_node = int(best_path_mat[t + 1])
        best_path_mat[t] = psi_mat[:, t+1][back_node - 1]
    
    return best_path_prob, best_path_mat    

if __name__ == "__main__":
    color2id = { "RED": 0, "WHITE": 1 }
    # model parameters
    
    # 初始状态概率向量 
    # 表示在时刻t=1处于状态q_i的概率
    pi = [0.2, 0.4, 0.4]
    
    # 状态转移矩阵
    # 表示在时刻t处于状态q_i的条件下在时刻t+1转移到状态q_j的概率
    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    
    # 观测概率矩阵
    # 表示在时刻t处于状态q_j的条件下生成观测v_k的概率
    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    # input
    observations = (1, 1, 0)
    
    # HMM满足两个基本假设
    # 1. 齐次马尔科夫性假设：时刻t的状态只与t-1的状态有关
    # 2. 观测独立性假设：观测只和当前时刻的状态有关
    HMM_model = (pi, A, B)
    
    # process
    observ_prob_forward = forward_algorithm(observations, HMM_model)
#     print(observ_prob_forward)
    
    observ_prob_backward = backward_algorithm(observations, HMM_model)
#     print(observ_prob_backward)
    
    best_prob, best_path = Viterbi_algorithm(observations, HMM_model)
    print(best_prob, best_path)