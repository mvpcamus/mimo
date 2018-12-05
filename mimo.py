import sys
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# hyper parameters
random_seed = 0 #random seed for raw data shuffling
num_pairs = 36 #duplication number of each experiment in raw data
num_repeat = 5 #number of data generation (data shuffling)
num_train = 4000 #number of train data in ML data set
draw_graph = True

def read_csv(gshcsv, gssgcsv):
  # read csv data
  gshdata = np.loadtxt(gshcsv, delimiter=';')
  gssgdata = np.loadtxt(gssgcsv, delimiter=';')
  if np.shape(gshdata) != np.shape(gssgdata):
    print('ERROR: different shape of gsh/gssg data')
    exit

  if np.ndim(gshdata) == 2:
    num_exp = np.shape(gshdata)[0]
    num_obs = np.shape(gshdata)[1] - 1
  elif np.ndim(gshdata) == 1:
    gshdata = gshdata[np.newaxis, :]
    gssgdata = gssgdata[np.newaxis, :]
    num_exp = 1 
    num_obs = np.shape(gshdata)[1] - 1
  else:
    print('ERROR: invalid data shape: ', np.shape(gshdata))

  # data generation
  np.random.seed(random_seed)
  rawdata = None
  for j in range(num_repeat):
    for i in range(num_exp):
      gsh_idx = np.random.choice(num_obs, num_pairs, replace=False)
      gssg_idx = np.random.choice(num_obs, num_pairs, replace=False)

      gsh_val = gshdata[i][gsh_idx]
      gssg_val = gssgdata[i][gssg_idx]

      if gshdata[i][-1] != gssgdata[i][-1]:
        print('ERROR: different ratio detected')
        exit
      ratio = np.full(np.shape(gsh_val), gshdata[i][-1])
      if rawdata is None:
        rawdata = np.column_stack([gsh_val, gssg_val, ratio])
      else:
        rawdata = np.vstack([rawdata, np.column_stack([gsh_val, gssg_val, ratio])])

  print('raw data generated:', np.shape(rawdata))
  return rawdata


def train(rawdata):
  # shuffle rawdata
  np.random.shuffle(rawdata)

  # construct ML data set
  x_train = rawdata[:num_train, 0:2]
  y_train = rawdata[:num_train, 2]
  x_test = rawdata[num_train:, 0:2]
  y_test = rawdata[num_train:, 2]

  print('train data:', np.shape(y_train)[0], ', test data:', np.shape(y_test)[0])

  if draw_graph:
    ax = plt.axes(projection='3d')
    colorize = dict(c=y_test, cmap=plt.cm.get_cmap('rainbow', 256))
    ax.scatter3D(x_test[:,0], x_test[:,1], y_test, **colorize)
    ax.view_init(azim=-155, elev=20)
    plt.show()

  # model description
  model = Sequential()
  model.add(Dense(32, input_dim=2, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dense(64, activation='relu'))
  model.add(Dense(32, activation='relu'))
  model.add(Dense(16, activation='relu'))
  model.add(Dense(1))

  model.compile(loss='mse', optimizer='rmsprop')

  # train
  model.fit(x_train, y_train, epochs=3000, batch_size=50)
  print('train END')

  # test
  loss = model.evaluate(x_test, y_test, batch_size=50)
  print('------------------------------')
  print('test loss: ' + str(loss))

  model.save('2nd_model.h5')


def test(valdata):
  x_val = valdata[:, 0:2]
  y_val = valdata[:, 2]
  model = load_model('2nd_model.h5')
  loss = model.evaluate(x_val, y_val)
  print('------------------------------')
  print('validation loss: ' + str(loss))
  y_hat = model.predict(x_val)
  print(y_hat)
  print('mean:', np.mean(y_hat))

def graph():
  model = load_model('2nd_model.h5')  
#  x1 = np.arange(1300, 10000, 50)
#  x2 = np.arange(2700, 8800, 50)
  x1 = np.arange(1000, 10000, 25)
  x2 = np.arange(1000, 10000, 25)
  x_hat = np.array(np.meshgrid(x1, x2)).T.reshape(-1, 2)
  y_hat = model.predict(x_hat)
  x_diff = np.array(abs(x_hat[:,0] - x_hat[:,1])).reshape(-1, 1)
  result = np.hstack((x_hat, y_hat, x_diff))
  result = result[result[:,2] > 0]
  result = result[result[:,2] < 2.1]
  result = result[result[:,3] < 3000]

  np.savetxt('graph.csv', result[:,0:3], delimiter=',', fmt="%f")

  ax = plt.axes(projection='3d')
  colorize = dict(c=result[:,2], cmap=plt.cm.get_cmap('rainbow', 256))
  ax.view_init(azim=-155, elev=20)
  ax.scatter3D(result[:,0], result[:,1], result[:,2], **colorize)
  plt.show()
#  for azim in range(0, 360, 1):
#    ax.view_init(azim=azim, elev=20)
#    plt.savefig("./movie/rotation_%d.png" % azim)

if __name__ == '__main__':
  try:
    sys_arg = sys.argv[1]
  except:
    print('Usage hint: need argument train or test')
    exit()

  if sys_arg == 'train':
    data = read_csv('./gshimo.csv', './gssgimo.csv')
    train(data)
  elif sys_arg == 'test':
    data = read_csv('./gshimo_val.csv', './gssgimo_val.csv')
    test(data)
  elif sys_arg == 'graph':
    graph()
  else:
    print('Invalid argument ', sys_arg)

