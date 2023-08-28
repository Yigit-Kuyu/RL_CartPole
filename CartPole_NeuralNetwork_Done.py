# importing the dependencies
import gym
from gym import wrappers
import numpy as np
import random
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler # for normalization
from IPython.display import clear_output
import matplotlib.pyplot as plt

# Link:
# https://github.com/adibyte95/CartPole-OpenAI-GYM


'''
NOTE
action:
0 for left 
1 for right
'''
checkpoint = ModelCheckpoint('model/model_dnn.h5', monitor='val_loss', verbose=1, save_best_only=True)
no_of_timesteps = 500
min_score = 100


def plot_res(values, title=''):
    ''' Plot the reward curve and histogram of results over time.'''
    # Update the window after each episode
    clear_output(wait=True)

    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    f.suptitle(title)
    ax[0].plot(values, label='score per run')
    ax[0].axhline(195, c='red', ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x, p(x), "--", label='trend')
    except:
        print('')

    # Plot the histogram of results
    ax[1].hist(values[-50:])
    ax[1].axvline(195, c='red', label='goal')
    ax[1].set_xlabel('Scores per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    plt.show()


# Not:
# Aşağıdaki dataset eksik oluşturuluyor, dataset her bir state (X) icib sadece tek bir action (Y)'a gore
# olusturuluyor. Fakat her bir state için olası actionların her birini vermesi gerekirdu.
# Çünkü ML model 2 output lu sen Y değeri olarak tek output lu değer sokuyorsun ve iki output predict
# etmesini bekliyorsun.

# generate the training data
def generate_training_data(no_of_episodes):
    print('generating training data')
    # initize the environment
    env = gym.make('CartPole-v1').env
    X = []
    y = []
    left = 0
    right = 0

    for i_episode in range(no_of_episodes):
        prev_observation = env.reset()
        score = 0
        X_memory = []
        y_memory = []
        for t in range(no_of_timesteps):
            action = random.randrange(0, 2) # left or right

            ## debugging code
            '''
            if action == 0:
                left = left + 1
            else:
                right = right + 1
            '''
            # observation or state refers the same thing.
            new_observation, reward, done, info = env.step(action) # observation: [position, velocity, angle, angular velocity]
            score = score + reward # cumulative reward
            X_memory.append(prev_observation)
            y_memory.append(action)
            prev_observation = new_observation
            if done:
                if score > min_score:
                    for data in X_memory:
                        X.append(data)
                    for data in y_memory:
                        y.append(data)
                    print('episode : ', i_episode, ' score : ', score)
                break
        env.reset()
    # debugging code
    '''
    print('left : ', left)
    print('right: ',right)
    '''
    # converting them into numpy array
    X_copy=X.copy()
    X = np.asarray(X)
    y = np.asarray(y)

    # saving the numpy array
    np.save('X', X)
    np.save('y', y)

    # printing the size
    print('shape of X: ', X.shape)
    print('shape of target labels', y.shape)


# defines the model to be trained
def get_model():
    model = Sequential()
    model.add(Dense(128, input_dim=4)) # # Input layer=state_size
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(.5))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(.5))

    #model.add(Dense(1)) # Kapatildi
    #model.add(Activation('sigmoid')) # Kapatildi
    
    num_of_actions=2
    model.add(Dense(num_of_actions))  # two neurons in the output layer, each represents the q value for one of the action
    model.add(Activation('sigmoid'))


    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    model.optimizer.learning_rate = 0.2
    return model


# trains the model
# The model is being trained to predict the actions (Y) based on the state values (X), 
def train_model(model):
    # loading the training data from the disk
    X = np.load('X.npy')
    y = np.load('y.npy')

    # Normalization
    scaler = MinMaxScaler() # The state values (X) are normalized using MinMaxScaler() to ensure that all input values are within a similar range. 
    scaler.fit(X)
    X_normalized = scaler.transform(X)

    # making train test split
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=.3, random_state=42) 
    # test_size=.3 indicates that 30% of the data will be used for testing, while 70% will be used for training
    # When you set random_state=42, you ensure that the same random split is obtained each time you run the code.

    print('X_train: ', X_train.shape)
    print('y_train:', y_train.shape)
    print('X_test: ', X_test.shape)
    print('y_test: ', y_test.shape)
    # training the model
    model.fit(X_train, y_train, validation_data=[X_test, y_test], verbose=1,
              callbacks=[checkpoint],
              epochs=50, batch_size=100, shuffle=True) # orj: batch_size=10000,
    # verbose=1 prints training progress to the console.
    # the [checkpoint] callback is used to save the model's weights.
    # batch_size: The number of samples used in each training batch. This helps in efficient gradient updates during training.
    
    # returns the model
    return model


# testing the model
def testing(model):
    # model = load_model('model/model.h5')
    #env = gym.make('CartPole-v0').env # orj
    env = gym.make('CartPole-v1').env
    #env = wrappers.Monitor(env, 'nn_files', force=True)
    env.render(mode='human')
    observation = env.reset()
    no_of_rounds = 10
    max_rounds = no_of_rounds
    min_score = 1000000
    max_score = -1
    avg_score = 0
    final = []
    title='simple neural network'
    # playing a number of games
    while (no_of_rounds > 0):
        # initial score
        score = 0
        action = 0
        prev_obs = []
        while (True):
            #env.render() # Kapatildi
            if len(prev_obs) == 0:
                action = random.randrange(0, 2)
            else:
                data = np.asarray(prev_obs)
                data = np.reshape(data, (1, 4))
                # sigmoid function ile output verilen model [0,1] arasındadır output daima, aslında bir probabilty verir
                # Q_values action probabilty'i verir, sigmoid activation function'dan dolayı
                
                Q_values= model.predict(data) # current state'teki her bir action icin bir Q degeri, cunku NN 2 outputlu
                # argmax: gives the maximum index
                action=np.argmax(Q_values) # Choosen action
                '''
                #  Kapatildi
                output = model.predict(data)
                # checking if the required action is left or right
                if output[0][0] >= .5:
                    action = 1
                elif output[0][0] < .5:
                    action = 0
                '''
            new_observation, reward, done, info = env.step(action) # observation: [position, velocity, angle, angular velocity]
            prev_obs = new_observation
            # calculating total reward
            score = score + reward

            if done:
                final.append(score)
                # if the game is over
                print('game over!! your score is :  ', score)
                if score > max_score:
                    max_score = score
                elif score < min_score:
                    min_score = score
                avg_score += score
                env.reset()
                break
        no_of_rounds = no_of_rounds - 1
        # stats about scores
        if no_of_rounds == 0:
            print('avg score : ', avg_score / max_rounds)
            print('max score: ', max_score)
            print('min score: ', min_score)
            plot_res(final, title)

# calling the functions
generate_training_data(50000)
model = get_model()
model = train_model(model)
testing(model)