import numpy as np
import torch
import torch.nn as nn

class CS145_NN(nn.Module):
    def __init__(self):
        super(CS145_NN, self).__init__()
        with open('ohe_symptoms.txt', 'r') as ohe:
            ss = ohe.readlines()
        self.symptom_set = [s[:-1] for s in ss]

        with open('ohe_diseases.txt', 'r') as ohe:
            ds = ohe.readlines()
        self.disease_set = [d[:-1] for d in ds]

        self.ll1 = nn.Linear(len(self.symptom_set), 512) #256
        self.ll2 = nn.Linear(512, 128) #256, 512 for s18: 512, 512
        #self.ll3 = nn.Linear(512, 256) #active for s18
        #self.ll4 = nn.Linear(256, 128)
        self.ll5 = nn.Linear(128, len(self.disease_set)) #128 for s18: 256
        self.relu = nn.ReLU()
        

    def forward(self, x):
        #x = self.transform_test_data(x)
        x = self.relu(self.ll1(x))
        x = self.relu(self.ll2(x))
        #x = F.relu(self.ll3(x))
        #x = F.relu(self.ll4(x))
        x = self.ll5(x)
        return x

    '''
    def transform_test_data(self, X):
        N,d = X.shape
        symptoms = X[:,1:]
        input = np.zeros(shape=(N,131), dtype=int)
        for i in range(N):
            for sym in symptoms[i,:]:
                if sym != '':
                    s = sym
                    if s[0] == ' ':
                        s = s[1:]
                    input[i, self.symptom_set.index(s)] = 1
        return input
    '''

def predict(input, model, device):
    model = model.to(device)
    input = torch.tensor(input).to(device).float()
    model.eval()
    prediction = model(input).argmax(axis=1)
    return prediction

def transform_test_data(X, symptom_set):
    N,d = X.shape
    symptoms = X[:,1:]
    test_input = np.zeros(shape=(N,131), dtype=int)
    #print(symptom_set)
    for i in range(N):
        for sym in symptoms[i,:]:
            if sym != '':
                s = sym
                if s[0] == ' ':
                    s = s[1:]
                test_input[i,symptom_set.index(s)] = 1
    return test_input

def transform_test_label(y, disease_set):
    N, = y.shape
    to_csv = np.empty(shape=(N+1,2),dtype=object)
    to_csv[0] = ['ID','Disease']
    for i in range(N):
        to_csv[i+1] = [str(i+1), disease_set[y[i]]]
        
    return to_csv

if __name__ == '__main__':
    device = torch.device('cpu')
    model_d = torch.load('cs145_m25.pt', map_location=device)
    #print('hello')
    #for param in model_d.parameters():
    #    print(param.data)

    test_data = np.loadtxt('./test.csv', delimiter=',', dtype=str, skiprows=1)
    test_input = transform_test_data(test_data, model_d.symptom_set)
    test_pred = predict(test_input, model_d, device)
    csv_arr = transform_test_label(test_pred, model_d.disease_set)
    np.savetxt('submission27.csv',csv_arr, delimiter=',', fmt='%s')
