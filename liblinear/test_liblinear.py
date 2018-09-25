from liblinearutil import *
import pdb


y, x = svm_read_problem('heart_scale.txt')
prob = problem(y, x)
m =  train(prob)
save_model('heart_scale.model',m)

pdb.set_trace()
y, x = svm_read_problem('heart_scale_2.txt')
prob = problem(y, x)
m1 = load_model('heart_scale.model')
m_inc = warm_start_train(prob, m1)