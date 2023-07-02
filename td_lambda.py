import matplotlib.pyplot as plt
import numpy as np
import random



#references

# https://edstem.org/us/courses/39617/discussion/3174150
# https://edstem.org/us/courses/39617/discussion/3192259
# https://edstem.org/us/courses/39617/discussion/3197977

def random_walk():
    #[A,B,C,D,E,F,G]
    #outputs walk, result
    # used to generate figures 3, 4, 5
    # ideal predictions to terminate at G:
    # 1/6 - B
    # 1/3 - C
    # 1/2 - D
    # 2/3 - E
    # 5/6 - F
    
    x = 3
    x_basis = np.zeros(5, dtype=bool)
    walk = []
    random.seed()

    while x > 0 and x < 6:
        x_basis[x-1] = True
        walk.append(x_basis.copy())
        x_basis[x-1] = False
                
        x+=1
        if random.random() < 0.5:
            x-=2
    
    result = int(x/6) # returns 0 or 1
    

    return walk, float(result)


def td_lambda_paper(episodes, set_length, lda, alpha, w, convergence = True):

    #w = weight vector, replaces V
    # Pt = predictions
    #Pt = sum(w(i)xt(i))
    # xt = vector x1,x2,x3...xm,z 
    # z is result
    # lda = lambda
 
    for T in range(episodes):    
        
        delta_w = np.zeros(5) #change in weights to zero
                
        #training sets
        for set in range(set_length):
              
            walk, z = random_walk()
            
            #propagate prediction states
            P=[]
            for q in walk:
                P.append(np.sum(w*q))
            P.append(z)
                    
            for t in range(len(walk)): #step through each P
                
                summer = np.zeros(5)
                for k in range(t+1):
                    summer += walk[k]*(lda**(t-k)) # sum for equation 4
                
                delta_w += alpha * (P[t+1]-P[t]) * summer # equation 4
                        
        w+=delta_w #update w after 10 sets training
        np.clip(w,0,1,out=w)

        if convergence and np.sum(np.absolute(delta_w)) < 0.01:
            break
       
    return(w)


def td_lambda_class(episodes, lda, alpha, gma = 1):
    
    # version of TD(lambda) that is given in the class video
    # specific to random walk
    # episodes = # of episodes (VT)
    # V = inital expected values of the discounted return
    # lda = lambda
    # gma = gamma
    
    # RETURNS
    # V
    
    #every episode is a value set, with a series of states
     
    #gma = 1

    V = []
    for k in range(5):
        V.append(random.random())
    V = np.array(V)

    for T in range(episodes):    
        
        pV = V.copy()       #VT = VT-1
        e = np.zeros(5)     #eligibility to zero
        walk, result = random_walk()
        
        rewards = np.zeros(len(walk))
        rewards[-1] = result

        for t in range(1,len(walk)): #steps through each item 
            
            e = np.where(walk[t-1], e+1, e) #after step, add 1
            V += alpha*(result+ gma*pV*walk[t] - pV*walk[t-1])*e #are there rewards? rewards[t] +
            e *= lda*gma

    return(V)
        


def main():

    ideal_weights = np.array([1/6, 1/3, 1/2, 2/3, 5/6])
    
    lambdas = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]

    episodes = 100

    
    #Figure 3

    errors = []
    for lda in lambdas:
        
        best_error = 10000
        
        for alpha in range(0,65,5):
            #initalize weight function random values
            random.seed(47)
            w = np.random.rand(5)
            
            #tester = td_lambda_class(episodes, lda, alpha/100)
            tester = td_lambda_paper(episodes, 10, lda, alpha/100, w)
            
            #print(lda,', ',alpha)
            
            RMSerror = ( np.sum((tester - ideal_weights)*(tester - ideal_weights)) / 5 )**0.5
            if RMSerror < best_error:
                best_error = RMSerror

        errors.append(best_error) #RMS
        

    plt.plot(lambdas, errors)
    plt.xlabel('Lambda')
    plt.ylabel('RMS error using best alpha')
    plt.title("Figure 3")
    plt.show()
    


    #Figure 4
    
    ladas = [0,0.3,0.8,1]
    
    errors = np.zeros((4,13))

    alpha = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    
    for lda in range(4):
        if lda < 3:
            for a in range(13):
                random.seed(47)
                w = np.random.rand(5)

                tester = td_lambda_paper(episodes, 10, ladas[lda], alpha[a], w)
                #print(tester)
                errors[lda,a] = (np.sum( ((ideal_weights - tester)**2) )/5)**0.5
        else:
            for a in range(9):
                random.seed(47)
                w = np.random.rand(5)

                tester = td_lambda_paper(episodes, 10, ladas[lda], alpha[a], w)
                #print(tester)
                errors[lda,a] = (np.sum( ((ideal_weights - tester)**2) )/5)**0.5

        #print(errors)

    plt.plot(alpha, errors[0,:], 'r', label="lambda 0.0")
    plt.plot(alpha, errors[1,:], 'b', label="lambda 0.3")
    plt.plot(alpha, errors[2,:], 'g', label="lambda 0.8")
    plt.plot(alpha[:9], errors[3,:9], 'm', label="lambda 1.0")


    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.title("Figure 4")
    plt.legend(loc="lower right")
    plt.show()



    #Figure 5

    errors = []
    for lda in lambdas:
        
        best_error = 10000
        
        for alpha in range(0,65,5):
            
            w = np.ones(5)*0.5
            #tester = td_lambda_class(episodes, lda, alpha/100)
            tester = td_lambda_paper(episodes, 1, lda, alpha/100, w, False)
            
            #print(lda,', ',alpha)
            
            RMSerror = ( np.sum((tester - ideal_weights)*(tester - ideal_weights)) / 5 )**0.5
            if RMSerror < best_error:
                best_error = RMSerror

        errors.append(best_error) #RMS
        

    plt.plot(lambdas, errors)
    plt.xlabel('Lambda')
    plt.ylabel('RMS error using best alpha')
    plt.title("Figure 5")
    plt.show()








    







if __name__ == "__main__":

    main()







    



