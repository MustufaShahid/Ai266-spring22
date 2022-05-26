//mustufashahidassignment3
//updated attempt

import numpy as np

def main():
  print("\nnaive Bayes classification \n")
  data = np.loadtxt(".\\people_data_20.txt",
    dtype=np.str, delimiter=" ")
  print("Data looks like: ")
  for i in range(5):
    print(data[i])
  print(". . . \n")

  nx = 3  # number predictor variables
  nc = 2  # number classes
  N = 20  # data items

  joint_cts = np.zeros((nx,nc), dtype=np.int) 
  y_cts = np.zeros(nc, dtype=np.int)

  X = ['WordPress Developer', 'Karachi', 'Pakistan']
  print("Item to classify: ")
  print(X)

  for i in range(N):
    y = int(data[i][nx])  # class in last column
    y_cts[y] += 1
    for j in range(nx):
      if data[i][j] == X[j]:
        joint_cts[j][y] += 1

  joint_cts += 1  # Laplacian smoothing

  print("\nJoint counts: ")
  print(joint_cts)
  print("\nClass counts: ")
  print(y_cts)

  # compute evidence terms directly
  # e_terms = np.zeros(nc, dtype=np.float32) 
  # for k in range(nc):
  #   v = 1.0
  #   for j in range(nx):
  #     v *= joint_cts[j][k] / (y_cts[k] + nx)
  #   v *= y_cts[k] / N
  #   e_terms[k] = v

  # compute evidence terms using log trick
  e_terms = np.zeros(nc, dtype=np.float32) 
  for k in range(nc):
    v = 0.0
    for j in range(nx):
      v += np.log(joint_cts[j][k]) - np.log(y_cts[k] + nx)
    v += np.log(y_cts[k]) - np.log(N)
    e_terms[k] = np.exp(v)

  np.set_printoptions(4)
  print("\nEvidence terms: ")
  print(e_terms)

  evidence = np.sum(e_terms)
  probs = np.zeros(nc, dtype=np.float32)
  for k in range(nc):
    probs[k] = e_terms[k] / evidence

  print("\nPseudo-probabilities: ")
  print(probs)

  pc = np.argmax(probs)
  print("\nPredicted class: ")
  print(pc)

  print("\nEnd naive Bayes demo ")

if __name__ == "__main__":
  main()
  
  newCSVTest

predT = test.drop(columns=['id'])
predT.head()

predictionOnTest = perceptron.predict(predT)

newCSVTest['target'] = predictionOnTest

newCSVTest.head()
newCSVTest.to_csv('submission.csv', index=False)

