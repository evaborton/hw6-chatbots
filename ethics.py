"""
Please answer the following ethics and reflection questions.

We are expecting at least three complete sentences for each question for full credit.

Each question is worth 2.5 points.
"""

######################################################################################
"""
QUESTION 1 - Anthropomorphizing

Is there potential for users of your chatbot possibly anthropomorphize
(attribute human characteristics to an object) it?
What are some possible ramifications of anthropomorphizing chatbot systems?
Can you think of any ways that chatbot designers could ensure that users can easily
distinguish the chatbot responses from those of a human?
"""

Q1_your_answer = """

There is pottential for users of our chatbot to possibly anthropomorphize it. 
A possible ramifications of anthropomorphizing chatbot systems include false confidence of 
users in these systems. The stakes are not very high in our movie recommendation chatbot, 
but for a system who is highly human-like (ChatGPT) and also tends to hallucinate facts, 
false confidence in the chatbot can be dangerous. Another possible problem is creating frustration 
for users. If the chatbot fails to understand common sense (despite "speaking" like a human), the 
user might get frustrated and fail to understand how to correct the chatbot in a way that it will 
understand. (For example, repeating the same thing you just said to a human while emphasizing 
certain words might help a human understand you, but not a chatbot). 
Chatbot designers could ensure that users can easily distinguish the chatbot responses 
from those of a human by introducing the chatbot to users carefully and being explicit 
about what the chatbot can and cannot do. During interactions with users, the chatbot 
should also present its responses as objectively as possible, ideally including proper sources. 

"""

######################################################################################

"""
QUESTION 2 - Privacy Leaks

One of the potential harms for chatbots is collecting and then subsequently leaking
(advertly or inadvertently) private information. Does your chatbot have risk of doing so?
Can you think of ways that designers of the chatbot can help to mitigate this risk?
"""

Q2_your_answer = """

Our chatbot probably doesn't have much of a risk of privacy leaks, since the movies
people like is not generally considered personal information. It also doesn't store
anything but movies and sentiments, so even if a user inexplicably decides to tell the
chatbot personal information, it won't collect or store it. However, if a chatbot did
collect and store personal information, care should be taken to encrypt the information
and make sure that the chatbot is not trained on it in a way that it could regurgitate
that information to other users.

"""

######################################################################################

"""
QUESTION 3 - Effects on Labor

Advances in dialogue systems, and the language technologies based on them, could lead to the automation of
tasks that are currently done by paid human workers, such as responding to customer-service queries,
translating documents or writing computer code. These could displace workers and lead to widespread unemployment.
What do you think different stakeholders -- elected government officials, employees at technology companies,
citizens -- should do in anticipation of these risks and/or in response to these real-world harms?
"""

Q3_your_answer = """

On the government side of things, regulations should be put in place requiring a certain number or
percentage of human workers in these fields as a form of necessary quality control, since technology
will always fail at some point. At the current point we are at in NLP, it is in the best interest of
citizens to support companies that continue to use human workers for these tasks, since humans are
currently better at the tasks given as examples above than language models are. Employees at technology
companies should be honest with themselves and their customers about the capabilities and limitations
of their models— for example, do not market machine translation as equal in quality to human translation
when that is not true at the moment.

"""

"""
QUESTION 4 - Refelection

You just built a frame-based dialogue system using a combination of rule-based and machine learning
approaches. Congratulations! What are the advantages and disadvantages of this paradigm
compared to an end-to-end deep learning approach, e.g. ChatGPT?
"""

Q4_your_answer = """

The main advatage of this paradigm is that it is intepretable. It is also modular, which makes it easy to 
implement and debug. However, this system is not very generalizable and high performing as an end-to-end 
deep learning approach such as ChatGPT. 

"""
