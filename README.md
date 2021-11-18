# GRU Implementation 

To get a brief idea of GRU theory and the structure of a GRU structure refer to the below image 


![image](https://user-images.githubusercontent.com/80246631/142428572-c3e66703-9c34-4476-867e-49bf4d4cdd57.png)

The structure of the GRU allows it to adaptively capture dependencies from large sequences of data without discarding information from earlier parts of the sequence. This is achieved through its gating units, similar to the ones in LSTMs, which solve the vanishing/exploding gradient problem of traditional RNNs. These gates are responsible for regulating the information to be kept or discarded at each time step. We’ll dive into the specifics of how these gates work and how they overcome the above issues later in this article.

![image](https://user-images.githubusercontent.com/80246631/142428836-9da29028-3119-4d6e-8d81-67a2da87fa2d.png)

Other than its internal gating mechanisms, the GRU functions just like an RNN, where sequential input data is consumed by the GRU cell at each time step along with the memory, or otherwise known as the hidden state. The hidden state is then re-fed into the RNN cell together with the next input data in the sequence. This process continues like a relay system, producing the desired output.

### Reset Gate (Short term memory)

In the first step, we’ll be creating the Reset gate. This gate is derived and calculated using both the hidden state from the previous time step and the input data at the current time step. Mathematically, this is achieved by multiplying the previous hidden state and current input with their respective weights and summing them before passing the sum through a sigmoid function. The sigmoid function will transform the values to fall between 0 and 1, allowing the gate to filter between the less-important and more-important information in the subsequent steps.

![image](https://user-images.githubusercontent.com/80246631/142429105-1dfa8a11-a36f-440b-9e56-26228a21db9a.png)

The previous hidden state will first be multiplied by a trainable weight and will then undergo an element-wise multiplication (Hadamard product) with the reset vector. This operation will decide which information is to be kept from the previous time steps together with the new inputs. At the same time, the current input will also be multiplied by a trainable weight before being summed with the product of the reset vector and previous hidden state above. Lastly, a non-linear activation tanh function will be applied to the final result to obtain r in the equation below.

### Update Gate (Long Term memory)

Similarly, we’ll have to create the Update gate. Just like the Reset gate, the gate is computed using the previous hidden state and current input data.

![image](https://user-images.githubusercontent.com/80246631/142429413-b9a133f2-1241-428d-ae3e-9593815f40dd.png)

Both the Update and Reset gate vectors are created using the same formula, but, the weights multiplied with the input and hidden state are unique to each gate, which means that  the final vectors for each gate are different. This allows the gates to serve their specific purposes.

![image](https://user-images.githubusercontent.com/80246631/142429465-e979e1d1-e9fd-4e5c-a641-daa8ef95cd00.png)

The Update vector will then undergo element-wise multiplication with the previous hidden state to obtain u in our equation below, which will be used to compute our final output later.

![image](https://user-images.githubusercontent.com/80246631/142429498-a7f91f52-80df-4d46-804c-ad311a669383.png)

The Update vector will also be used in another operation later when obtaining our final output. The purpose of the Update gate here is to help the model determine how much of the past information stored in the previous hidden state needs to be retained for the future.

### Combining the outputs

In the last step, we will be reusing the Update gate and obtaining the updated hidden state.

![image](https://user-images.githubusercontent.com/80246631/142429768-e0fe3da6-56ec-49b7-b078-f9aa65bd7d1e.png)

This time, we will be taking the element-wise inverse version of the same Update vector (1 - Update gate) and doing an element-wise multiplication with our output from the Reset gate, r. The purpose of this operation is for the Update gate to determine what portion of the new information should be stored in the hidden state.

Lastly, the result from the above operations will be summed with our output from the Update gate in the previous step, u. This will give us our new and updated hidden state.

![image](https://user-images.githubusercontent.com/80246631/142429820-0dfe2d1d-ea7e-49fb-89ad-9b0320b253ca.png)

We can use this new hidden state as our output for that time step as well by passing it through a linear activation layer.
 

## Design decisions:

1. We remove the new line markers during pre-processing. `response.text.split("\n")[0:]`
2. Followed by this, we remove the punctuation from the corpus to prepare for tokenization. `str.maketrans("", "", string.punctuation)`
3. For optimization, we used a standard ADAM optimizer, which works on the intuition on RMS prop + two beta values. 
4. Weights have been initialized using the Xavier GLOROT scheme. I tried random weights too, but the loss didn't decrease. So only GLOROT worked. 
5. Cross Entropy Loss is used, and perplexity is used just as a metric (not for ADAM optimization).
6. We find that after 100 epochs with the current layer we obtain a perplexity of _12.249737_. `np.min(train_perplexity)`

## Evaluation: 

We use a regular cross entropy loss, and a perplexity metric. 

![image](https://user-images.githubusercontent.com/20723780/138415573-64ad2e7c-b1f0-44d1-8628-4f099e4aba4c.png)

The cross entropy loss _maximizes_ the probability of the given next true word, and in theory, perplexity is implemented as the exponential of the cross entropy loss. Perplexity in this scenario can be easy to understand from a _human_ perspective, as whenever we try to predict the next words, we have a choice between *n* words, where *n* denotes the perplexity. 

Furthermore, the cross entropy is a very textbook loss for problems/formulations such as these. We use it over regression based losses like the squared error loss as we want to perform Convex optimization, that enables us to efficiently train a deep network. The slide below from Grosse et. al from Stanford University talks about the intuition nicely. 

![image](https://user-images.githubusercontent.com/20723780/138416248-eddf6e62-eeef-4ccb-8b96-013c42ada084.png)

## Sample text generated after a few epochs: 

 ```
decembers bareness everywhere and medromio of ephesus 
i could not die a food to see the world i am weary sir 
i am not so far i am a strumpet in my mind no more with them 
i will not say you speak to me 
 ```
 ## Install Requirements: 
 
 We use the high level keras api of tensorflow 2 to create our model: 
 
 ```
 pip install -r requirements.txt
 ```
 
 ## Usage:
 
 ```
 conda install -c conda-forge jupyterlab
 jupyter-lab
 #open Shakespeare_text_generator_using_GRU.ipynb
 ```
 
 
 ## Format code to PEP-8 standards (Important for contributing to the repo): 
 
 This repository is strictly based on *PEP-8* standards. To assert PEP-8 standards after editing your own code, use the following: 
 
 ```
 black  shakespeare_text_generator_using_gru.py
 ```
 
 If you would like to replace the dataset, just replace the text. I would NOT recommend NLP processing in this sort of a problem where you remove stop words, URLs, etc as we do not want incorrect semantics to be learned. 
 
 NOTE: Since the data used is large Colab Pro was used for access to the additional ram it offers. 

### Reference: 

1. https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be
2. https://arxiv.org/abs/1412.3555 (Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling, Junyoung et al.)
3. https://www.analyticsvidhya.com/blog/2021/03/introduction-to-gated-recurrent-unit-gru/
4. https://towardsdatascience.com/perplexity-in-language-models-87a196019a94
