import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import collections
import itertools

seed = 42
N = 1000
K = 4
P = 0.2

min_number_skills = 1
max_number_skills = 4

np.random.seed(seed)
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})


skills = sorted([
    "Python", "C", "C++",
    "Assembly", "Java", "C#",
    # "JavaScript", "Go", "Rust"
])


# prior probabilities
pS = np.random.random(len(skills))
pS = pS / np.sum(pS)
print(pS)

# a posteriori probabilities
pSC = np.zeros((len(skills), len(skills)))
for a, b in itertools.combinations(range(len(skills)), 2):
    value = np.random.random(1)
    pSC[a, b] = value
    pSC[b, a] = value


users_skills = []

print("skills", skills)
print("pS", pS)
print("pSC", pSC)
print("--")

for i in range(N):
    user_skills = list()

    new_skill_added = True

    while new_skill_added:
        new_skill_added = False

        if len(user_skills) <= 0:
            first_skill = np.random.choice(skills, p=pS)
            user_skills.append(first_skill)
            new_skill_added = True
        else:
            possible_new_skills = list(set(skills) - set(user_skills))

            p = []
            for skill in skills:
                i = skills.index(skill)
                if skill not in user_skills:
                    p_i = pS[i]
                    for user_skill in user_skills:
                        j = skills.index(user_skill)
                        p_i *= pSC[i, j]
                    p.append(p_i)
                else:
                    p.append(0.0)

            for i, skill in enumerate(skills):
                value = np.random.random()
                if value < p[i]:
                    user_skills.append(skills[i])
                    new_skill_added = True

    users_skills.append(user_skills)

print(users_skills[0:10])

# Verify the probabilities
every_skills = list(itertools.chain(*users_skills))
counted = collections.Counter(every_skills)
counted_skills = [counted[s] for s in skills]
print(counted_skills / np.sum(counted_skills))

G = nx.watts_strogatz_graph(N, K, P, seed)
# print(G.nodes)
# nx.draw(G)
# plt.savefig("test.png")
