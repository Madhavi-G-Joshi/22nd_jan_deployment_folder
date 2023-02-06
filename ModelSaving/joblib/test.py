import joblib
import pickle

#load the model
model= joblib.load('log_reg.pkl')

#load model with pickle
model= pickle.load(open('log_reg.pkl','rb'))


result = model.predict ([[1,2,3,4,5,6,7,8]])

if result[0]==1:
    print('daibetic')
    
else:
    print('not diabetic')