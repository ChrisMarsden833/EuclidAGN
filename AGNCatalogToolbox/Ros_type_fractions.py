import numpy as np

N_agn_SF=np.array([[75,117,38],[175,370,142],[177 ,368,172],[98,166,70]])
N_SF=N_agn_SF+np.array([[2514,1214,239],[7957,3888,761],[11334,5273,954],[7159,2726,466]])
N_agn_Q=np.array([[12,19,14],[19,72,25],[9 ,25,8],[3,7,6]])
N_Q=N_agn_Q+np.array([[867,830,271],[2594,3354,1107],[769 ,1534,532],[105,272,66]])
N_agn_SB=np.array([[23,7,4],[231,104,18],[125,182,29],[29,86,23]])
N_SB=N_agn_SB+np.array([[5,2,4],[12,7,2],[11,31,7],[0,7,2]])
N_agn_tot=N_agn_SF+N_agn_Q+N_agn_SB
N_tot=N_SF+N_Q+N_SB

np.savez('../IDL_data/type_fractions.npz',factor_SF=factor_SF,factor_Q=factor_Q,factor_SB=factor_SB)

factor_SF=N_agn_SF/N_agn_tot*N_tot/N_SF
factor_Q=N_agn_Q/N_agn_tot*N_tot/N_Q
factor_SB=N_agn_SB/N_agn_tot*N_tot/N_SB

np.savez('../IDL_data/weight_factors.npz',factor_SF=factor_SF,factor_Q=factor_Q,factor_SB=factor_SB)