import tkinter
import gym
import numpy as np
import threading

#-------------------- Functions ----------------------------------

def closeit():
    global env2
    env2.reset()
    env2.close()
    show_chk.configure(state="active")
    train_btn.configure(state="active")
    msg_lbl.configure(text="Ready to Train",fg="green")
    
def mTrain():
    global env2,state
    
   
    #--------------------------------------------------------------
    show=var.get()
    show_chk.configure(state="disabled")
    train_btn.configure(state="disabled")
    msg_lbl.configure(text="Model is Training...",fg="orange")
    win.update()
    for episode in range(total_episodes):
        state = env.reset()[0] # Reset the environment and init state is given
        step = 0
        done = False
        if show:
            print("EPISODE: ",episode)
        for step in range(max_steps):
            #random action
            action = env.action_space.sample()
            env.render()
            new_state, reward, done,truncated, info = env.step(action)
            #Somehow, the environment does not give negative rewards for game over, so fix it:
            if done and reward == 0:
                reward = -5
            if new_state == state:
                reward = -1
            if show:
                print("NEW STATE:",new_state,"REWARD:",reward)
            #qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
            qtable[state, action]=reward + gamma * np.max(qtable[new_state, :])
            
            if show:
                print("QTABLE AT",state,qtable[state])
            state = new_state
            if done:
                if show:
                    print("GAME OVER.\n\n")
                break
        if show:
            print("new QTABLE")
            print(qtable)

    env.reset()
    env.close()
    eval_btn.configure(state="active")
    
#    ----------- game display -------
    env2 = gym.make('FrozenLake-v1', desc=None,map_name="4x4", is_slippery=False,render_mode='human')
    state = env2.reset()[0]
    env2.render()
    
    

#-----------------------------------
    
    msg_lbl.configure(text="ready to run result",fg="green")

def runResult():
    global env2,state
    msg_lbl.configure(text="Game Result is running...",fg="orange")
    eval_btn.configure(state="disabled")
    win.update()
    
   
    step = 0
    done = False
    print("****************************************************")
    for step in range(max_steps):
            env2.render()
            action = np.argmax(qtable[state,:])
            new_state, reward, done,truncated, info = env2.step(action)
            if done:
                break
            state = new_state
    
    threading.Timer(2.0, closeit).start()

#-------------------- Create window ------------------------------
win=tkinter.Tk()
win.geometry("300x150")
win.title("QLearning Project")
win.resizable(False,False)

var=tkinter.IntVar()
show_chk=tkinter.Checkbutton(win,text="Show training info in console",variable=var, pady=5)
show_chk.pack()



train_btn=tkinter.Button(win,text="Train model",width=10,command=mTrain)
train_btn.pack()

eval_btn=tkinter.Button(win,text="Run Result",width=10,state="disabled",command=runResult)
eval_btn.pack()

msg_lbl=tkinter.Label(win,text="Ready to Train",fg="green",pady=5)
msg_lbl.pack()

 #-------------- set environment and variables -------------------------
   
env = gym.make('FrozenLake-v1', desc=None,map_name="4x4", is_slippery=False,render_mode='ansi')
action_size = env.action_space.n  # 4 action
state_size = env.observation_space.n #16 state
qtable = np.zeros((state_size, action_size))
    
    #print("Init Qtable:\n ")
    #print(qtable)
    
total_episodes = 300        # Total episodes
learning_rate = 0.8         # Learning rate
max_steps = 20              # Max steps per episode
gamma = 0.5                 # Discounting rate

#-----------------------------------------------------------------------

win.mainloop()

